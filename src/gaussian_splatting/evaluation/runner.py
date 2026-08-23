"""Render and evaluate camera collections for command-line workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from gaussian_splatting.data.camera import Camera
from gaussian_splatting.evaluation.metrics import LPIPSMetric, mean_psnr, psnr
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.losses import ssim
from gaussian_splatting.training.trainer import camera_to


@dataclass(frozen=True)
class ImageEvaluationMetrics:
    psnr: float
    ssim: float
    lpips: float


@dataclass(frozen=True)
class EvaluationMetricsResult:
    images: dict[str, ImageEvaluationMetrics]
    mean_psnr: float
    mean_ssim: float
    mean_lpips: float

    @property
    def per_image_psnr(self) -> dict[str, float]:
        """Retain convenient access to the former per-image PSNR mapping."""

        return {name: metrics.psnr for name, metrics in self.images.items()}


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
    *,
    ssim_window_size: int = 11,
    ssim_sigma: float = 1.5,
    ssim_k1: float = 0.01,
    ssim_k2: float = 0.03,
    lpips_metric: LPIPSMetric | None = None,
) -> EvaluationMetricsResult:
    """Compute per-image PSNR, SSIM, and VGG LPIPS and write JSON."""

    if not cameras:
        raise ValueError("at least one evaluation camera is required")
    per_image: dict[str, ImageEvaluationMetrics] = {}
    psnr_values: list[torch.Tensor] = []
    ssim_values: list[torch.Tensor] = []
    lpips_values: list[torch.Tensor] = []
    if lpips_metric is None:
        lpips_metric = LPIPSMetric(device=model.means_world.device)
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
            psnr_value = psnr(rendered, runtime_camera.image)
            ssim_value = ssim(
                rendered,
                runtime_camera.image,
                window_size=ssim_window_size,
                sigma=ssim_sigma,
                k1=ssim_k1,
                k2=ssim_k2,
            )
            lpips_value = lpips_metric(rendered, runtime_camera.image)
            name = camera.image_name or f"view_{index:03d}.png"
            per_image[name] = ImageEvaluationMetrics(
                psnr=float(psnr_value.item()),
                ssim=float(ssim_value.item()),
                lpips=float(lpips_value.item()),
            )
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)
    if was_training:
        model.train()
    result = EvaluationMetricsResult(
        images=per_image,
        mean_psnr=float(mean_psnr(psnr_values).item()),
        mean_ssim=float(torch.stack(ssim_values).mean().item()),
        mean_lpips=float(torch.stack(lpips_values).mean().item()),
    )
    destination = Path(output_json)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation JSON: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "images": {
                    name: asdict(metrics) for name, metrics in result.images.items()
                },
                "mean_psnr": result.mean_psnr,
                "mean_ssim": result.mean_ssim,
                "mean_lpips": result.mean_lpips,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "EvaluationMetricsResult",
    "ImageEvaluationMetrics",
    "evaluate_camera_set",
    "render_camera_set",
    "save_rgb_image",
]
