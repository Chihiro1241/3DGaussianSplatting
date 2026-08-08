"""Render and evaluate camera collections for command-line workflows."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from gaussian_splatting.data.camera import Camera
from gaussian_splatting.evaluation.metrics import mean_psnr, psnr
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.trainer import EvaluationResult, camera_to


def save_rgb_image(
    image: torch.Tensor,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a ``(3,H,W)`` RGB tensor as an 8-bit PNG."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite rendered image: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(pixels, mode="RGB").save(destination)


def render_camera_set(
    model: GaussianModel,
    renderer: GaussianRenderer,
    cameras: list[Camera],
    output_directory: str | Path,
) -> list[Path]:
    """Render cameras in order and save one PNG for each image name."""

    destination = Path(output_directory)
    written: list[Path] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            runtime_camera = camera_to(
                camera,
                device=model.means_world.device,
                dtype=model.means_world.dtype,
                include_image=False,
            )
            image = renderer(model, runtime_camera).image.clamp(0.0, 1.0)
            source_name = camera.image_name or f"view_{index:03d}.png"
            output_path = destination / Path(source_name).with_suffix(".png").name
            save_rgb_image(image, output_path)
            written.append(output_path)
    if was_training:
        model.train()
    return written


def evaluate_camera_set(
    model: GaussianModel,
    renderer: GaussianRenderer,
    cameras: list[Camera],
    output_json: str | Path,
) -> EvaluationResult:
    """Compute per-image PSNR and write the required evaluation JSON."""

    if not cameras:
        raise ValueError("at least one evaluation camera is required")
    per_image: dict[str, float] = {}
    values: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            runtime_camera = camera_to(
                camera,
                device=model.means_world.device,
                dtype=model.means_world.dtype,
                include_image=True,
            )
            if runtime_camera.image is None:
                raise ValueError("evaluation cameras must contain target images")
            rendered = renderer(model, runtime_camera).image.clamp(0.0, 1.0)
            value = psnr(rendered, runtime_camera.image)
            name = camera.image_name or f"view_{index:03d}.png"
            per_image[name] = float(value.item())
            values.append(value)
    if was_training:
        model.train()
    average = float(mean_psnr(values).item())
    destination = Path(output_json)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation JSON: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"images": per_image, "mean_psnr": average}, indent=2) + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(per_image_psnr=per_image, mean_psnr=average)


__all__ = ["evaluate_camera_set", "render_camera_set", "save_rgb_image"]
