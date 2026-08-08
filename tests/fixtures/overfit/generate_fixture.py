"""Regenerate the deterministic single-view overfit fixture.

Run from the repository root with the project installed, or with
``PYTHONPATH=src python tests/fixtures/overfit/generate_fixture.py``.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.config import load_config
from gaussian_splatting.renderer.renderer import GaussianRenderer


FIXTURE_DIRECTORY = Path(__file__).resolve().parent
SEED = 20260807
Y00 = 1.0 / (2.0 * math.sqrt(math.pi))


def _model_from_arrays(arrays: dict[str, np.ndarray]) -> GaussianModel:
    return GaussianModel(
        means_world=torch.from_numpy(arrays["means_world"]),
        raw_quaternions=torch.from_numpy(arrays["raw_quaternions"]),
        raw_scales=torch.from_numpy(arrays["raw_scales"]),
        raw_opacities=torch.from_numpy(arrays["raw_opacities"]),
        sh_dc=torch.from_numpy(arrays["sh_dc"]),
        sh_rest=torch.from_numpy(arrays["sh_rest"]),
    )


def _camera() -> Camera:
    values = json.loads((FIXTURE_DIRECTORY / "camera.json").read_text(encoding="utf-8"))
    return Camera(
        rotation_cw=torch.tensor(values["rotation_cw"], dtype=torch.float32),
        translation_cw=torch.tensor(values["translation_cw"], dtype=torch.float32),
        camera_center_world=torch.tensor(
            values["camera_center_world"], dtype=torch.float32
        ),
        fx=values["fx"],
        fy=values["fy"],
        cx=values["cx"],
        cy=values["cy"],
        width=values["width"],
        height=values["height"],
        image_name=values["image_name"],
    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    target_means = np.array(
        [
            [-0.34, -0.20, 2.00],
            [0.18, -0.12, 1.58],
            [-0.10, 0.29, 2.35],
            [0.35, 0.25, 2.05],
        ],
        dtype=np.float32,
    )
    target_scales = np.array(
        [
            [0.28, 0.16, 0.18],
            [0.20, 0.27, 0.16],
            [0.24, 0.14, 0.25],
            [0.18, 0.24, 0.15],
        ],
        dtype=np.float32,
    )
    target_quaternions = np.array(
        [
            [1.00, 0.10, 0.00, 0.05],
            [0.94, 0.00, 0.25, 0.08],
            [0.91, -0.18, 0.05, 0.22],
            [0.96, 0.12, -0.20, 0.00],
        ],
        dtype=np.float32,
    )
    target_opacities = np.array([[0.72], [0.64], [0.78], [0.68]], dtype=np.float32)
    target_colors = np.array(
        [
            [0.90, 0.18, 0.12],
            [0.12, 0.78, 0.22],
            [0.16, 0.28, 0.94],
            [0.88, 0.72, 0.10],
        ],
        dtype=np.float32,
    )
    target = {
        "means_world": target_means,
        "raw_quaternions": target_quaternions,
        "raw_scales": np.log(target_scales).astype(np.float32),
        "raw_opacities": np.log(
            target_opacities / (1.0 - target_opacities)
        ).astype(np.float32),
        "sh_dc": ((target_colors - 0.5) / Y00).astype(np.float32)[:, None, :],
        "sh_rest": np.zeros((4, 15, 3), dtype=np.float32),
    }

    initial_colors = np.array(
        [
            [0.42, 0.38, 0.52],
            [0.48, 0.46, 0.36],
            [0.36, 0.52, 0.48],
            [0.55, 0.38, 0.42],
        ],
        dtype=np.float32,
    )
    initial_means = target_means + rng.normal(0.0, 0.10, target_means.shape).astype(
        np.float32
    )
    initial = {
        "means_world": initial_means,
        "raw_quaternions": np.array(
            [[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32
        ),
        "raw_scales": np.log(target_scales * 1.25).astype(np.float32),
        "raw_opacities": np.full(
            (4, 1), math.log(0.35 / 0.65), dtype=np.float32
        ),
        "sh_dc": ((initial_colors - 0.5) / Y00).astype(np.float32)[:, None, :],
        "sh_rest": np.zeros((4, 15, 3), dtype=np.float32),
    }

    np.savez_compressed(FIXTURE_DIRECTORY / "target_gaussians.npz", **target)
    np.savez_compressed(FIXTURE_DIRECTORY / "initial_gaussians.npz", **initial)

    config = load_config(FIXTURE_DIRECTORY / "config.yaml")
    target_model = _model_from_arrays(target)
    with torch.no_grad():
        image = GaussianRenderer(config.rendering)(target_model, _camera()).image
    pixels = (
        image.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(pixels, mode="RGB").save(FIXTURE_DIRECTORY / "ground_truth.png")


if __name__ == "__main__":
    main()

