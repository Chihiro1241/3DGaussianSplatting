"""Train the reference PyTorch 3D Gaussian Splatting implementation."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from gaussian_splatting.config import Config, load_config, resolve_device, resolve_dtype
from gaussian_splatting.data import (
    load_dataset,
)
from gaussian_splatting.io.checkpoint import (
    load_checkpoint,
    model_from_checkpoint_state,
    read_checkpoint,
    validate_resume_config,
)
from gaussian_splatting.model import (
    GaussianModel,
    generate_initial_points,
    initialize_gaussian_model,
)
from gaussian_splatting.renderer import GaussianRenderer
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
)
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.screen_radius_diagnostics import (
    ScreenRadiusDiagnostic,
)
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.schedules import compute_scene_extent
from gaussian_splatting.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset scene directory")
    parser.add_argument("--config", type=Path, required=True, help="complete YAML configuration")
    parser.add_argument("--output", type=Path, required=True, help="new or resumed run directory")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="allow a benchmark runner-created output directory for a fresh run",
    )
    parser.add_argument(
        "--image-directory",
        default="images",
        help="COLMAP image subdirectory (paper Mip-NeRF360 uses images_4/images_2)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="resume from a .pt checkpoint instead of initializing a new model",
    )
    parser.add_argument("--diagnose-screen-radius-start", type=int, default=None)
    parser.add_argument("--diagnose-screen-radius-end", type=int, default=None)
    parser.add_argument(
        "--profile-training",
        action="store_true",
        help="profile synchronized wall-clock time for the initial training steps",
    )
    parser.add_argument("--profile-warmup", type=int, default=5, metavar="N")
    parser.add_argument("--profile-steps", type=int, default=10, metavar="N")
    parser.add_argument(
        "--milestone-iterations",
        type=int,
        nargs="*",
        default=(),
        metavar="N",
        help="iterations whose numbered checkpoints must be retained",
    )
    parser.add_argument(
        "--stop-after-iteration",
        type=int,
        default=None,
        metavar="N",
        help="stop cleanly at N while preserving the configured full-run schedules",
    )
    parser.add_argument(
        "--disable-training-evaluation",
        action="store_true",
        help="defer all evaluation to the separate native-resolution evaluator",
    )
    parser.add_argument(
        "--render-backend",
        choices=("reference", "cuda"),
        default="reference",
        help="Gaussian rasterization backend (default: reference)",
    )
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _density_statistics_for_model(
    model: GaussianModel,
    config: Config,
) -> ScreenSpaceDensityStatistics | None:
    """Create ADC statistics only when the configured feature is enabled."""
    if not config.features.adaptive_density_control:
        return None
    return ScreenSpaceDensityStatistics.for_model(model)


def _prepare_output(
    path: Path, *, resume: bool, config: object, allow_existing: bool = False
) -> None:
    if path.exists() and not resume and not allow_existing:
        raise FileExistsError(
            f"output directory already exists and output.exist_policy=error: {path}"
        )
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=resume or allow_existing)
    config_path = path / "config.yaml"
    serialized = yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True)
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"resolved config already exists: {config_path}")
        existing_config = load_config(config_path)
        if existing_config != config:
            raise ValueError(
                f"existing resolved config does not match the resume config: {config_path}"
            )
    else:
        config_path.write_text(serialized, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.diagnose_screen_radius_start is None) != (
        args.diagnose_screen_radius_end is None
    ):
        raise ValueError(
            "screen-radius diagnostic start and end must be supplied together"
        )
    if args.profile_training and args.profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if args.profile_training and args.profile_steps <= 0:
        raise ValueError("--profile-steps must be positive")
    config = load_config(args.config)
    if args.stop_after_iteration is not None and not (
        0 < args.stop_after_iteration <= config.training.iterations
    ):
        raise ValueError("stop-after iteration must lie within the configured training range")
    if (
        args.output.exists()
        and args.resume is None
        and not args.allow_existing_output
    ):
        raise FileExistsError(
            f"output directory already exists and output.exist_policy=error: {args.output}"
        )
    device = resolve_device(config.runtime)
    dtype = resolve_dtype(config.runtime)
    _seed_everything(config.runtime.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = load_dataset(
        args.data,
        config,
        splits=("train", "val"),
        image_directory=args.image_directory,
    )
    train_cameras = dataset.train
    train_cameras_quarter = None
    train_cameras_half = None
    if config.features.resolution_warmup:
        if config.data.resolution_scale != 1.0:
            raise ValueError(
                "resolution warm-up requires data.resolution_scale=1.0 so evaluation is native"
            )
        quarter_config = replace(
            config, data=replace(config.data, resolution_scale=0.25)
        )
        half_config = replace(
            config, data=replace(config.data, resolution_scale=0.5)
        )
        train_cameras_quarter = load_dataset(
            args.data,
            quarter_config,
            splits=("train",),
            load_points=False,
            image_directory=args.image_directory,
        ).train
        train_cameras_half = load_dataset(
            args.data,
            half_config,
            splits=("train",),
            load_points=False,
            image_directory=args.image_directory,
        ).train
    if dataset.format == "colmap":
        evaluation_cameras = []
    elif dataset.val:
        evaluation_cameras = dataset.val
    else:
        evaluation_cameras = load_dataset(
            args.data,
            config,
            splits=("test",),
            load_points=False,
            image_directory=args.image_directory,
        ).test
    if args.disable_training_evaluation:
        evaluation_cameras = []

    checkpoint_state = None
    if args.resume is None:
        if dataset.initial_points is not None and dataset.initial_colors is not None:
            points = dataset.initial_points.to(device=device, dtype=dtype)
            colors = dataset.initial_colors.to(device=device, dtype=dtype)
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.runtime.seed)
            points, colors = generate_initial_points(dataset.scene_center, config, generator)
        model = initialize_gaussian_model(points, colors, config)
        start_iteration = 0
        camera_order = None
        camera_cursor = 0
        best_mean_psnr = None
    else:
        checkpoint_state = read_checkpoint(args.resume, map_location="cpu")
        validate_resume_config(checkpoint_state["config"], config)
        model = model_from_checkpoint_state(
            checkpoint_state,
            config,
            device=device,
            dtype=dtype,
        )
        start_iteration = int(checkpoint_state["iteration"])
        camera_order = list(checkpoint_state["camera_order"])
        camera_cursor = int(checkpoint_state["camera_cursor"])
        best_mean_psnr = checkpoint_state["best_mean_psnr"]

    model = model.to(device=device, dtype=dtype)
    spatial_lr_scale = compute_scene_extent(train_cameras)
    optimizer = create_optimizer(model, config, position_lr_scale=spatial_lr_scale)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial * spatial_lr_scale,
        final_learning_rate=config.training.position_lr_final * spatial_lr_scale,
    )
    density_statistics = _density_statistics_for_model(model, config)
    if checkpoint_state is not None:
        load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            density_statistics=density_statistics,
            map_location=device,
            restore_random_state=True,
        )

    _prepare_output(
        args.output,
        resume=args.resume is not None,
        allow_existing=args.allow_existing_output,
        config=config,
    )
    renderer = GaussianRenderer(config.rendering, backend=args.render_backend)
    trainer = Trainer(
        model=model,
        renderer=renderer,
        train_cameras=train_cameras,
        train_cameras_quarter=train_cameras_quarter,
        train_cameras_half=train_cameras_half,
        evaluation_cameras=evaluation_cameras,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=args.output,
        start_iteration=start_iteration,
        camera_order=camera_order,
        camera_cursor=camera_cursor,
        best_mean_psnr=best_mean_psnr,
        density_statistics=density_statistics,
        screen_radius_diagnostic=(
            ScreenRadiusDiagnostic(
                args.output / "diagnostics",
                start=args.diagnose_screen_radius_start,
                end=args.diagnose_screen_radius_end,
            )
            if args.diagnose_screen_radius_start is not None
            and args.diagnose_screen_radius_end is not None
            else None
        ),
        profile_training=args.profile_training,
        profile_warmup=args.profile_warmup,
        profile_steps=args.profile_steps,
        milestone_iterations=tuple(args.milestone_iterations),
        stop_iteration=args.stop_after_iteration,
    )
    try:
        trainer.train()
    except BaseException:
        trainer.write_training_telemetry(status="FAILED")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
