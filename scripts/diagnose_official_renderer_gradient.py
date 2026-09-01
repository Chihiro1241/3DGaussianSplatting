"""Compare the paper-era GraphDeco renderer/gradients with this implementation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from gaussian_splatting.config import load_config
from gaussian_splatting.data.dataset import load_dataset
from gaussian_splatting.model.gaussian_model import GaussianModel as LocalGaussianModel
from gaussian_splatting.model.initialization import initialize_gaussian_model
from gaussian_splatting.renderer.cuda_rasterizer import (
    _camera_matrices,
    prepare_cuda_rasterization,
    rasterize_gaussians_cuda,
)
from gaussian_splatting.training.losses import total_loss
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import (
    PositionLearningRateScheduler,
    compute_scene_extent,
)
from gaussian_splatting.training.trainer import camera_to


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float()
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()) if values.numel() else 0.0,
        "mean": float(values.mean().item()) if values.numel() else 0.0,
        "median": float(values.median().item()) if values.numel() else 0.0,
        "max": float(values.max().item()) if values.numel() else 0.0,
    }


def _gradient_comparison(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | None]:
    a_flat = a.detach().double().reshape(-1)
    b_flat = b.detach().double().reshape(-1)
    norm_a = torch.linalg.vector_norm(a_flat)
    norm_b = torch.linalg.vector_norm(b_flat)
    denominator = norm_a * norm_b
    cosine = None if denominator.item() == 0.0 else float(torch.dot(a_flat, b_flat).div(denominator).item())
    return {
        "official_norm": float(norm_a.item()),
        "ours_norm": float(norm_b.item()),
        "norm_ratio_ours_over_official": None if norm_a.item() == 0.0 else float((norm_b / norm_a).item()),
        "cosine_similarity": cosine,
        "max_abs_difference": float((a_flat - b_flat).abs().max().item()),
    }


def _raw_local_model(official: object) -> LocalGaussianModel:
    model = LocalGaussianModel(
        means_world=official._xyz.detach().clone(),
        raw_quaternions=official._rotation.detach().clone(),
        raw_scales=official._scaling.detach().clone(),
        raw_opacities=official._opacity.detach().clone(),
        sh_dc=official._features_dc.detach().clone(),
        sh_rest=official._features_rest.detach().clone(),
    ).cuda()
    model.set_active_sh_degree(int(official.active_sh_degree))
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    from gaussian_renderer import render as official_render  # type: ignore[import-not-found]
    from scene.dataset_readers import readColmapSceneInfo  # type: ignore[import-not-found]
    from scene.gaussian_model import GaussianModel as OfficialGaussianModel  # type: ignore[import-not-found]
    from utils.camera_utils import cameraList_from_camInfos  # type: ignore[import-not-found]
    from utils.loss_utils import l1_loss as official_l1_loss  # type: ignore[import-not-found]
    from utils.loss_utils import ssim as official_ssim  # type: ignore[import-not-found]

    torch.manual_seed(0)
    config = load_config(args.config)
    local_dataset = load_dataset(args.data, config, splits=("train", "test"), image_directory="images")
    scene_info = readColmapSceneInfo(str(args.data.resolve()), "images", True)
    official_cameras = cameraList_from_camInfos(
        scene_info.train_cameras,
        1.0,
        SimpleNamespace(resolution=-1, data_device="cuda"),
    )
    official_camera = sorted(official_cameras, key=lambda camera: camera.image_name)[0]
    local_camera_cpu = next(
        camera
        for camera in local_dataset.train
        if Path(str(camera.image_name)).stem == official_camera.image_name
    )
    local_camera = camera_to(
        local_camera_cpu, device=torch.device("cuda"), dtype=torch.float32
    )

    official_model = OfficialGaussianModel(3)
    official_model.create_from_pcd(
        scene_info.point_cloud, scene_info.nerf_normalization["radius"]
    )
    independently_initialized_local = initialize_gaussian_model(
        local_dataset.initial_points.cuda(),
        local_dataset.initial_colors.cuda(),
        config,
    )
    local_model = _raw_local_model(official_model)
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    official_pkg = official_render(official_camera, official_model, pipe, background)
    local_preparation = prepare_cuda_rasterization(
        local_model.transformed_parameters(), local_camera, config.rendering, 0
    )
    local_image, local_projected, local_visible = rasterize_gaussians_cuda(
        local_model.transformed_parameters(), local_camera, local_preparation
    )
    official_image = official_pkg["render"]
    difference = official_image - local_image
    mse = difference.square().mean()
    official_visible = official_pkg["visibility_filter"]
    official_view, official_full = (
        official_camera.world_view_transform,
        official_camera.full_proj_transform,
    )
    local_view, local_full = _camera_matrices(local_camera, local_model.means_world)

    gradient_names = {
        "xyz": "means_world",
        "sh_dc": "sh_dc",
        "sh_rest": "sh_rest",
        "opacity": "raw_opacities",
        "scale": "raw_scales",
        "quaternion": "raw_quaternions",
    }
    official_loss = 0.8 * official_l1_loss(official_image, official_camera.original_image) + 0.2 * (
        1.0 - official_ssim(official_image, official_camera.original_image)
    )
    official_loss.backward()
    official_gradients = {
        "xyz": official_model._xyz.grad.detach().clone(),
        "sh_dc": official_model._features_dc.grad.detach().clone(),
        "sh_rest": official_model._features_rest.grad.detach().clone(),
        "opacity": official_model._opacity.grad.detach().clone(),
        "scale": official_model._scaling.grad.detach().clone(),
        "quaternion": official_model._rotation.grad.detach().clone(),
    }
    local_loss = total_loss(
        local_image,
        local_camera.image,
        lambda_dssim=config.loss.lambda_dssim,
        window_size=config.loss.ssim_window_size,
        sigma=config.loss.ssim_sigma,
        k1=config.loss.ssim_k1,
        k2=config.loss.ssim_k2,
    ).total
    local_loss.backward()
    local_gradients = {
        label: getattr(local_model, local_name).grad.detach().clone()
        for label, local_name in gradient_names.items()
    }

    official_model.training_setup(
        SimpleNamespace(
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
    )
    official_lr_iteration_1 = official_model.update_learning_rate(1)
    local_extent = compute_scene_extent(local_dataset.train)
    local_optimizer = create_optimizer(local_model, config, position_lr_scale=local_extent)
    local_scheduler = PositionLearningRateScheduler(
        local_optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial * local_extent,
        final_learning_rate=config.training.position_lr_final * local_extent,
    )
    local_lr_iteration_1 = local_scheduler.step(1)

    result = {
        "camera": {
            "image_name": official_camera.image_name,
            "width": int(official_camera.image_width),
            "height": int(official_camera.image_height),
            "ground_truth_max_abs_difference": float(
                (official_camera.original_image - local_camera.image).abs().max().item()
            ),
            "viewmatrix_max_abs_difference": float((official_view - local_view).abs().max().item()),
            "projmatrix_max_abs_difference": float((official_full - local_full).abs().max().item()),
            "camera_center_max_abs_difference": float(
                (official_camera.camera_center - local_camera.camera_center_world).abs().max().item()
            ),
            "tanfovx_official": float(math.tan(official_camera.FoVx * 0.5)),
            "tanfovx_ours": float(local_camera.width / (2.0 * local_camera.fx)),
            "tanfovy_official": float(math.tan(official_camera.FoVy * 0.5)),
            "tanfovy_ours": float(local_camera.height / (2.0 * local_camera.fy)),
        },
        "initial_state": {
            "gaussian_count": int(official_model.get_xyz.shape[0]),
            "spatial_lr_scale_official": float(scene_info.nerf_normalization["radius"]),
            "spatial_lr_scale_ours": float(local_extent),
            "raw_scale": _distribution(official_model._scaling),
            "actual_scale": _distribution(official_model.get_scaling),
            "independent_local_max_abs_difference": {
                "xyz": float(
                    (official_model._xyz - independently_initialized_local.means_world)
                    .abs()
                    .max()
                    .item()
                ),
                "sh_dc": float(
                    (official_model._features_dc - independently_initialized_local.sh_dc)
                    .abs()
                    .max()
                    .item()
                ),
                "sh_rest": float(
                    (official_model._features_rest - independently_initialized_local.sh_rest)
                    .abs()
                    .max()
                    .item()
                ),
                "raw_opacity": float(
                    (official_model._opacity - independently_initialized_local.raw_opacities)
                    .abs()
                    .max()
                    .item()
                ),
                "raw_scale": float(
                    (official_model._scaling - independently_initialized_local.raw_scales)
                    .abs()
                    .max()
                    .item()
                ),
                "quaternion": float(
                    (official_model._rotation - independently_initialized_local.raw_quaternions)
                    .abs()
                    .max()
                    .item()
                ),
            },
        },
        "renderer": {
            "mae": float(difference.abs().mean().item()),
            "rmse": float(torch.sqrt(mse).item()),
            "psnr_official_vs_ours": float((-10.0 * torch.log10(mse)).item()) if mse.item() else float("inf"),
            "max_abs_pixel_difference": float(difference.abs().max().item()),
            "official_visible_count": int(official_visible.sum().item()),
            "ours_visible_count": int(local_visible.sum().item()),
            "visibility_xor_count": int(torch.logical_xor(official_visible, local_visible).sum().item()),
            "official_radii": _distribution(official_pkg["radii"][official_visible]),
            "ours_radii": _distribution(local_projected.radii),
        },
        "loss": {
            "official": float(official_loss.item()),
            "ours": float(local_loss.item()),
        },
        "gradients": {
            name: _gradient_comparison(official_gradients[name], local_gradients[name])
            for name in gradient_names
        },
        "optimizer": {
            "official": {
                "betas": list(official_model.optimizer.defaults["betas"]),
                "epsilon": float(official_model.optimizer.defaults["eps"]),
                "spatial_lr_scale": float(official_model.spatial_lr_scale),
                "position_lr_group_initial": float(1.6e-4 * official_model.spatial_lr_scale),
                "position_lr_iteration_1": float(official_lr_iteration_1),
                "sh_dc_lr": 2.5e-3,
                "sh_rest_lr": 2.5e-3 / 20.0,
                "opacity_lr": 5.0e-2,
                "scale_lr": 5.0e-3,
                "rotation_lr": 1.0e-3,
            },
            "ours": {
                "betas": list(local_optimizer.defaults["betas"]),
                "epsilon": float(local_optimizer.defaults["eps"]),
                "spatial_lr_scale": float(local_extent),
                "position_lr_group_initial": float(config.training.position_lr_initial * local_extent),
                "position_lr_iteration_1": float(local_lr_iteration_1),
                "sh_dc_lr": float(config.training.sh_dc_lr),
                "sh_rest_lr": float(config.training.sh_rest_lr),
                "opacity_lr": float(config.training.opacity_lr),
                "scale_lr": float(config.training.scale_lr),
                "rotation_lr": float(config.training.quaternion_lr),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
