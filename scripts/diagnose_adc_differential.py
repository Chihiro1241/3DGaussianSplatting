"""Run a controlled official/local ADC comparison through a target iteration."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from gaussian_splatting.config import load_config
from gaussian_splatting.data.dataset import load_dataset
from gaussian_splatting.model.gaussian_model import GaussianModel as LocalGaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
    reset_gaussian_opacity,
    run_density_control_event,
)
from gaussian_splatting.training.losses import total_loss
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import (
    PositionLearningRateScheduler,
    compute_scene_extent,
)
from gaussian_splatting.training.trainer import camera_to


def _local_model(official: object) -> LocalGaussianModel:
    model = LocalGaussianModel(
        means_world=official._xyz.detach().clone(),
        raw_quaternions=official._rotation.detach().clone(),
        raw_scales=official._scaling.detach().clone(),
        raw_opacities=official._opacity.detach().clone(),
        sh_dc=official._features_dc.detach().clone(),
        sh_rest=official._features_rest.detach().clone(),
    ).cuda()
    model.set_active_sh_degree(0)
    return model


def _summary(accum: torch.Tensor, denom: torch.Tensor, threshold: float) -> dict[str, float | int]:
    accum = accum.detach().reshape(-1).float()
    denom = denom.detach().reshape(-1).float()
    observed = denom > 0
    averaged = torch.zeros_like(accum)
    averaged[observed] = accum[observed] / denom[observed]

    def stats(values: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(values.mean().item()) if values.numel() else 0.0,
            "median": float(values.median().item()) if values.numel() else 0.0,
            "max": float(values.max().item()) if values.numel() else 0.0,
        }

    return {
        "gaussian_count": int(accum.numel()),
        "observed_count": int(observed.sum().item()),
        "xyz_gradient_accum_mean_observed": float(accum[observed].mean().item()),
        "xyz_gradient_accum_median_observed": float(accum[observed].median().item()),
        "denom_mean_observed": float(denom[observed].mean().item()),
        "denom_mean_all": float(denom.mean().item()),
        "averaged_gradient_mean_observed": stats(averaged[observed])["mean"],
        "averaged_gradient_median_observed": stats(averaged[observed])["median"],
        "averaged_gradient_max_observed": stats(averaged[observed])["max"],
        "threshold_exceeded_count": int((observed & (averaged >= threshold)).sum().item()),
    }


def _camera_sequence(names: list[str], iterations: int) -> list[str]:
    rng = random.Random(0)
    sequence: list[str] = []
    stack: list[str] = []
    for _ in range(iterations):
        if not stack:
            stack = list(names)
        sequence.append(stack.pop(rng.randint(0, len(stack) - 1)))
    return sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()
    if args.iterations < 600 or args.iterations % 100 != 0:
        parser.error("--iterations must be a multiple of 100 and at least 600")
    sys.path.insert(0, str(args.official_root.resolve()))
    from gaussian_renderer import render as official_render  # type: ignore[import-not-found]
    from scene.dataset_readers import readColmapSceneInfo  # type: ignore[import-not-found]
    from scene.gaussian_model import GaussianModel as OfficialGaussianModel  # type: ignore[import-not-found]
    from utils.camera_utils import cameraList_from_camInfos  # type: ignore[import-not-found]
    from utils.loss_utils import l1_loss as official_l1_loss  # type: ignore[import-not-found]
    from utils.loss_utils import ssim as official_ssim  # type: ignore[import-not-found]

    torch.manual_seed(0)
    config = load_config(args.config)
    local_dataset = load_dataset(args.data, config, splits=("train",), image_directory="images")
    local_extent = compute_scene_extent(local_dataset.train)
    local_cameras = {
        Path(str(camera.image_name)).stem: camera_to(
            camera, device=torch.device("cuda"), dtype=torch.float32
        )
        for camera in local_dataset.train
    }
    scene_info = readColmapSceneInfo(str(args.data.resolve()), "images", True)
    official_cameras = {
        camera.image_name: camera
        for camera in cameraList_from_camInfos(
            scene_info.train_cameras,
            1.0,
            SimpleNamespace(resolution=-1, data_device="cuda"),
        )
    }
    names = sorted(set(official_cameras) & set(local_cameras))
    sequence = _camera_sequence(names, args.iterations)

    official_model = OfficialGaussianModel(3)
    official_model.create_from_pcd(scene_info.point_cloud, scene_info.nerf_normalization["radius"])
    opt_args = SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=1.6e-4,
        position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=2.5e-3,
        opacity_lr=5.0e-2,
        scaling_lr=5.0e-3,
        rotation_lr=1.0e-3,
    )
    official_model.training_setup(opt_args)
    local_model = _local_model(official_model)
    local_optimizer = create_optimizer(local_model, config, position_lr_scale=local_extent)
    local_scheduler = PositionLearningRateScheduler(
        local_optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial * local_extent,
        final_learning_rate=config.training.position_lr_final * local_extent,
    )
    local_statistics = ScreenSpaceDensityStatistics.for_model(local_model)
    local_renderer = GaussianRenderer(config.rendering, backend="cuda").cuda()
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    official_event: dict[str, float | int] | None = None
    local_event_summary: dict[str, float | int] | None = None
    official_before: dict[str, float | int] | None = None
    local_before: dict[str, float | int] | None = None
    threshold = 0.0002

    for iteration, name in enumerate(sequence, start=1):
        official_model.update_learning_rate(iteration)
        local_scheduler.step(iteration)
        if iteration % 1000 == 0:
            official_model.oneupSHdegree()
            local_model.one_up_sh_degree()
        official_model.optimizer.zero_grad(set_to_none=True)
        local_optimizer.zero_grad(set_to_none=True)

        official_pkg = official_render(official_cameras[name], official_model, pipe, background)
        official_image = official_pkg["render"]
        official_gt = official_cameras[name].original_image
        official_loss = 0.8 * official_l1_loss(official_image, official_gt) + 0.2 * (
            1.0 - official_ssim(official_image, official_gt)
        )
        official_loss.backward()
        official_visible = official_pkg["visibility_filter"]
        official_model.max_radii2D[official_visible] = torch.maximum(
            official_model.max_radii2D[official_visible], official_pkg["radii"][official_visible]
        )
        official_model.add_densification_stats(official_pkg["viewspace_points"], official_visible)

        local_render = local_renderer(local_model, local_cameras[name], retain_screen_grad=True)
        local_loss = total_loss(
            local_render.image,
            local_cameras[name].image,
            lambda_dssim=config.loss.lambda_dssim,
            window_size=config.loss.ssim_window_size,
            sigma=config.loss.ssim_sigma,
            k1=config.loss.ssim_k1,
            k2=config.loss.ssim_k2,
        ).total
        local_loss.backward()
        local_statistics.accumulate(local_render)

        if iteration > 500 and iteration % 100 == 0:
            official_before = _summary(
                official_model.xyz_gradient_accum, official_model.denom, threshold
            )
            local_before = _summary(
                local_statistics.position_gradient_accumulator,
                local_statistics.position_gradient_denominator,
                threshold,
            )
            with torch.no_grad():
                official_average = official_model.xyz_gradient_accum / official_model.denom
                official_average[official_average.isnan()] = 0.0
                official_high = official_average.reshape(-1) >= threshold
                official_large = official_model.get_scaling.max(dim=1).values > (
                    official_model.percent_dense * official_model.spatial_lr_scale
                )
                official_clone_count = int((official_high & ~official_large).sum().item())
                official_split_count = int((official_high & official_large).sum().item())

            size_pruning = iteration > 3000
            size_threshold = 20.0 if size_pruning else None
            world_threshold = (
                0.1 * official_model.spatial_lr_scale if size_pruning else None
            )
            prune_calls: list[dict[str, int]] = []
            original_prune_points = official_model.prune_points

            def record_prune(prune_mask: torch.Tensor) -> None:
                with torch.no_grad():
                    low_opacity = official_model.get_opacity.squeeze(-1) < 0.005
                    large_screen = (
                        official_model.max_radii2D > size_threshold
                        if size_threshold is not None
                        else torch.zeros_like(low_opacity)
                    )
                    large_world = (
                        official_model.get_scaling.max(dim=1).values > world_threshold
                        if world_threshold is not None
                        else torch.zeros_like(low_opacity)
                    )
                    prune_calls.append(
                        {
                            "low_opacity_prune": int(low_opacity.sum().item()),
                            "large_screen_prune": int(large_screen.sum().item()),
                            "large_world_prune": int(large_world.sum().item()),
                            "total_prune": int(prune_mask.sum().item()),
                        }
                    )
                original_prune_points(prune_mask)

            official_model.prune_points = record_prune
            event_rng_state = torch.cuda.get_rng_state()
            try:
                official_model.densify_and_prune(
                    threshold,
                    0.005,
                    official_model.spatial_lr_scale,
                    size_threshold,
                )
            finally:
                official_model.prune_points = original_prune_points
            official_rng_after = torch.cuda.get_rng_state()
            final_official_prune = prune_calls[-1]
            official_event = {
                **official_before,
                "high_gradient_count": int(official_high.sum().item()),
                "clone_count": official_clone_count,
                "split_parent_count": official_split_count,
                "children_count": 2 * official_split_count,
                **final_official_prune,
                "gaussian_count_after": int(official_model.get_xyz.shape[0]),
            }

            torch.cuda.set_rng_state(event_rng_state)
            local_result = run_density_control_event(
                local_model,
                local_optimizer,
                local_statistics,
                gradient_threshold=threshold,
                densify_world_scale_threshold=0.01 * local_extent,
                prune_opacity_threshold=0.005,
                prune_screen_radius_threshold=size_threshold,
                prune_world_scale_threshold=(
                    0.1 * local_extent if size_pruning else None
                ),
            )
            torch.cuda.set_rng_state(official_rng_after)
            local_event_summary = {
                **local_before,
                "high_gradient_count": local_result.num_high_gradient,
                "clone_count": local_result.num_cloned,
                "split_parent_count": local_result.num_split_parents,
                "children_count": local_result.num_children_created,
                "low_opacity_prune": local_result.num_low_opacity,
                "large_screen_prune": local_result.num_large_screen,
                "large_world_prune": local_result.num_large_world,
                "total_prune": local_result.num_pruned_total,
                "gaussian_count_after": local_result.num_gaussians_after,
            }

        if iteration % 3000 == 0:
            official_model.reset_opacity()
            reset_gaussian_opacity(local_model, local_optimizer, maximum_opacity=0.01)

        if iteration < args.iterations:
            official_model.optimizer.step()
            local_optimizer.step()

    if official_event is None or local_event_summary is None:
        raise RuntimeError("target iteration did not execute a density-control event")
    result = {
        "conditions": {
            "iterations": args.iterations,
            "camera_sequence": "identical explicit official-style stack/pop sequence, seed 0",
            "camera_count": len(names),
            "resolution": [1332, 876],
            "resolution_warmup": False,
            "reason_warmup_disabled": "isolate ADC using official default native-resolution path",
            "threshold": threshold,
            "initial_state_identical": True,
        },
        "official": official_event,
        "ours": local_event_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
