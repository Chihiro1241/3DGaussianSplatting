"""Train the reference PyTorch 3D Gaussian Splatting implementation."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from gaussian_splatting.config import load_config, resolve_device, resolve_dtype
from gaussian_splatting.data import (
    load_blender_dataset,
    read_camera_poses_json,
    split_cameras,
)
from gaussian_splatting.io.checkpoint import (
    load_checkpoint,
    model_from_checkpoint_state,
    read_checkpoint,
    validate_resume_config,
)
from gaussian_splatting.model import generate_initial_points, initialize_gaussian_model
from gaussian_splatting.renderer import GaussianRenderer
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Blender dataset directory")
    parser.add_argument("--config", type=Path, required=True, help="complete YAML configuration")
    parser.add_argument("--output", type=Path, required=True, help="new or resumed run directory")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="resume from a .pt checkpoint instead of initializing a new model",
    )
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_output(path: Path, *, resume: bool, config: object) -> None:
    if path.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists and output.exist_policy=error: {path}"
        )
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=resume)
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
    config = load_config(args.config)
    if args.output.exists() and args.resume is None:
        raise FileExistsError(
            f"output directory already exists and output.exist_policy=error: {args.output}"
        )
    device = resolve_device(config.runtime)
    dtype = resolve_dtype(config.runtime)
    _seed_everything(config.runtime.seed)

    cameras = load_blender_dataset(args.data, config)
    train_cameras, evaluation_cameras = split_cameras(cameras, config.data.test_every)

    checkpoint_state = None
    if args.resume is None:
        metadata = read_camera_poses_json(args.data / config.data.camera_file)
        generator = torch.Generator(device=device)
        generator.manual_seed(config.runtime.seed)
        points, colors = generate_initial_points(metadata["target"], config, generator)
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
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    if checkpoint_state is not None:
        load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            restore_random_state=True,
        )

    _prepare_output(args.output, resume=args.resume is not None, config=config)
    renderer = GaussianRenderer(config.rendering)
    trainer = Trainer(
        model=model,
        renderer=renderer,
        train_cameras=train_cameras,
        evaluation_cameras=evaluation_cameras,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=args.output,
        start_iteration=start_iteration,
        camera_order=camera_order,
        camera_cursor=camera_cursor,
        best_mean_psnr=best_mean_psnr,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
