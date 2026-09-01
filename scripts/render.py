"""Render train or test views from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from gaussian_splatting.config import config_from_mapping, resolve_device, resolve_dtype
from gaussian_splatting.data import load_dataset
from gaussian_splatting.evaluation.runner import render_camera_set
from gaussian_splatting.io.checkpoint import model_from_checkpoint_state, read_checkpoint
from gaussian_splatting.renderer import GaussianRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-directory", default="images")
    parser.add_argument(
        "--render-backend",
        choices=("reference", "cuda"),
        default="reference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = read_checkpoint(args.checkpoint, map_location="cpu")
    config = config_from_mapping(state["config"])
    device = resolve_device(config.runtime)
    dtype = resolve_dtype(config.runtime)
    model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
    dataset = load_dataset(
        args.data,
        config,
        splits=(args.split,),
        load_points=False,
        image_directory=args.image_directory,
    )
    selected = getattr(dataset, args.split)
    render_camera_set(
        model,
        GaussianRenderer(config.rendering, backend=args.render_backend),
        selected,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
