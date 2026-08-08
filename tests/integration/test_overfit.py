from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gaussian_splatting.config import load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures" / "overfit"


def _load_model(path: Path) -> GaussianModel:
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return GaussianModel(
        means_world=torch.from_numpy(arrays["means_world"]),
        raw_quaternions=torch.from_numpy(arrays["raw_quaternions"]),
        raw_scales=torch.from_numpy(arrays["raw_scales"]),
        raw_opacities=torch.from_numpy(arrays["raw_opacities"]),
        sh_dc=torch.from_numpy(arrays["sh_dc"]),
        sh_rest=torch.from_numpy(arrays["sh_rest"]),
    )


def _load_camera() -> Camera:
    values = json.loads((FIXTURE_DIRECTORY / "camera.json").read_text(encoding="utf-8"))
    with Image.open(FIXTURE_DIRECTORY / "ground_truth.png") as image_file:
        image_array = np.asarray(image_file.convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 0, 1)))
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
        image=image,
        image_name=values["image_name"],
    )


def test_fixture_ground_truth_matches_saved_target_gaussians() -> None:
    config = load_config(FIXTURE_DIRECTORY / "config.yaml")
    target_model = _load_model(FIXTURE_DIRECTORY / "target_gaussians.npz")
    camera = _load_camera()

    with torch.no_grad():
        regenerated = GaussianRenderer(config.rendering)(target_model, camera).image

    quantized = regenerated.clamp(0.0, 1.0).mul(255.0).round().div(255.0)
    torch.testing.assert_close(quantized, camera.image, rtol=0.0, atol=0.0)


def test_single_view_overfit_reduces_loss_and_stays_finite(tmp_path: Path) -> None:
    config = load_config(FIXTURE_DIRECTORY / "config.yaml")
    random.seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)
    torch.manual_seed(config.runtime.seed)

    model = _load_model(FIXTURE_DIRECTORY / "initial_gaussians.npz")
    camera = _load_camera()
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    trainer = Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[camera],
        evaluation_cameras=[camera],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path,
        camera_order=[0],
    )

    losses: list[float] = []
    for iteration in range(1, config.training.iterations + 1):
        result = trainer.train_step(camera, iteration)
        loss_value = float(result.loss.total.detach().item())
        assert np.isfinite(loss_value)
        assert torch.isfinite(result.render.image).all()
        for parameter in model.parameters():
            assert torch.isfinite(parameter).all()
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        losses.append(loss_value)

    initial_mean = float(np.mean(losses[:50]))
    final_mean = float(np.mean(losses[-50:]))
    assert final_mean < 0.5 * initial_mean, (
        f"expected final mean loss < 50% of initial mean; "
        f"got initial={initial_mean:.8f}, final={final_mean:.8f}"
    )

