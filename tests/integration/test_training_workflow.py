from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.evaluation.runner import evaluate_camera_set
from gaussian_splatting.io.checkpoint import (
    load_checkpoint,
    model_from_checkpoint_state,
    read_checkpoint,
    save_checkpoint,
)
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer
from scripts.train import _prepare_output


ROOT = Path(__file__).resolve().parents[2]


def _tiny_config() -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        runtime=replace(config.runtime, device="cpu"),
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(
            config.training,
            iterations=4,
            log_interval=4,
            evaluation_interval=4,
            checkpoint_interval=4,
        ),
        features=replace(
            config.features,
            adaptive_density_control=False,
            opacity_reset=False,
        ),
        output=replace(config.output, save_rendered_images=False),
    )


def _tiny_model() -> GaussianModel:
    return GaussianModel(
        means_world=torch.tensor([[-0.10, 0.00, 2.0], [0.15, 0.08, 2.3]]),
        raw_quaternions=torch.tensor(
            [[1.0, 0.10, 0.05, 0.00], [1.0, 0.00, 0.20, -0.10]]
        ),
        raw_scales=torch.log(
            torch.tensor([[0.12, 0.08, 0.10], [0.10, 0.14, 0.08]])
        ),
        raw_opacities=torch.logit(torch.tensor([[0.35], [0.45]])),
        sh_dc=torch.tensor(
            [[[0.10, -0.05, 0.00]], [[-0.08, 0.02, 0.12]]]
        ),
        sh_rest=torch.zeros((2, 15, 3)),
    )


def _camera(image_value: float, name: str) -> Camera:
    image = torch.full((3, 7, 7), image_value)
    image[0].add_(0.05)
    image[2].sub_(0.05)
    return Camera(
        rotation_cw=torch.eye(3),
        translation_cw=torch.zeros(3),
        camera_center_world=torch.zeros(3),
        fx=8.0,
        fy=8.0,
        cx=3.0,
        cy=3.0,
        width=7,
        height=7,
        image=image,
        image_name=name,
    )


def _training_objects(
    model: GaussianModel, config: Config
) -> tuple[torch.optim.Adam, PositionLearningRateScheduler]:
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    return optimizer, scheduler


def test_checkpoint_render_round_trip_and_evaluation_json(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = _tiny_model()
    camera = _camera(0.25, "view_a.png")
    renderer = GaussianRenderer(config.rendering)
    optimizer, scheduler = _training_objects(model, config)
    with torch.no_grad():
        before = renderer(model, camera).image

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        iteration=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        camera_order=[0],
        camera_cursor=0,
        best_mean_psnr=None,
    )
    state = read_checkpoint(checkpoint, map_location="cpu")
    restored = model_from_checkpoint_state(
        state, config, device="cpu", dtype=torch.float32
    )
    with torch.no_grad():
        after = renderer(restored, camera).image

    assert float((before - after).abs().max()) <= 1.0e-6

    output_json = tmp_path / "metrics" / "evaluation.json"
    evaluation = evaluate_camera_set(
        restored, renderer, [camera], output_json
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["images"] == evaluation.per_image_psnr
    assert payload["mean_psnr"] == evaluation.mean_psnr
    assert np.isfinite(evaluation.mean_psnr)


def test_numbered_training_outputs_refuse_unconditional_overwrite(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    model = _tiny_model()
    camera = _camera(0.25, "view_a.png")
    optimizer, scheduler = _training_objects(model, config)
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

    trainer.validate(iteration=1)
    with pytest.raises(FileExistsError, match="validation metrics"):
        trainer.validate(iteration=1)

    trainer.save_checkpoint(iteration=1)
    with pytest.raises(FileExistsError, match="numbered checkpoint"):
        trainer.save_checkpoint(iteration=1)


def test_resume_output_rejects_mismatched_existing_resolved_config(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    output = tmp_path / "run"
    _prepare_output(output, resume=False, config=config)
    mismatched = replace(
        config,
        runtime=replace(config.runtime, seed=config.runtime.seed + 1),
    )

    with pytest.raises(ValueError, match="does not match"):
        _prepare_output(output, resume=True, config=mismatched)


def test_checkpoint_resume_reproduces_next_camera_loss_and_update(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    config = _tiny_config()
    cameras = [_camera(0.20, "view_0.png"), _camera(0.35, "view_1.png")]
    model = _tiny_model()
    optimizer, scheduler = _training_objects(model, config)
    trainer = Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=cameras,
        evaluation_cameras=cameras,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path / "original",
        camera_order=[0, 1],
        camera_cursor=2,
        best_mean_psnr=12.5,
    )
    trainer.train_step(cameras[1], iteration=1)

    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        iteration=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        camera_order=trainer.camera_order,
        camera_cursor=trainer.camera_cursor,
        best_mean_psnr=trainer.best_mean_psnr,
    )

    original_index, original_camera = trainer._next_camera()
    original_result = trainer.train_step(original_camera, iteration=2)
    original_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    state = read_checkpoint(checkpoint, map_location="cpu")
    restored_model = model_from_checkpoint_state(
        state, config, device="cpu", dtype=torch.float32
    )
    restored_optimizer, restored_scheduler = _training_objects(
        restored_model, config
    )
    loaded = load_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location="cpu",
        restore_random_state=True,
    )
    resumed = Trainer(
        model=restored_model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=cameras,
        evaluation_cameras=cameras,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        config=config,
        output_directory=tmp_path / "resumed",
        start_iteration=int(loaded["iteration"]),
        camera_order=list(loaded["camera_order"]),
        camera_cursor=int(loaded["camera_cursor"]),
        best_mean_psnr=loaded["best_mean_psnr"],
    )
    resumed_index, resumed_camera = resumed._next_camera()
    resumed_result = resumed.train_step(resumed_camera, iteration=2)

    assert resumed_index == original_index
    assert resumed.best_mean_psnr == trainer.best_mean_psnr == 12.5
    torch.testing.assert_close(
        resumed_result.loss.total,
        original_result.loss.total,
        rtol=0.0,
        atol=1.0e-6,
    )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(
            value, original_parameters[name], rtol=0.0, atol=1.0e-6
        )

    original_evaluation = trainer.validate(iteration=2)
    resumed_evaluation = resumed.validate(iteration=2)
    original_should_update = original_evaluation.mean_psnr > trainer.best_mean_psnr
    resumed_should_update = resumed_evaluation.mean_psnr > resumed.best_mean_psnr
    assert original_should_update == resumed_should_update
    assert resumed_evaluation.mean_psnr == pytest.approx(
        original_evaluation.mean_psnr, abs=1.0e-6
    )
    if original_should_update:
        trainer.best_mean_psnr = original_evaluation.mean_psnr
        resumed.best_mean_psnr = resumed_evaluation.mean_psnr
        trainer._save_best_checkpoint(iteration=2)
        resumed._save_best_checkpoint(iteration=2)
        assert read_checkpoint(
            tmp_path / "original" / "checkpoints" / "best.pt",
            map_location="cpu",
        )["best_mean_psnr"] == pytest.approx(
            read_checkpoint(
                tmp_path / "resumed" / "checkpoints" / "best.pt",
                map_location="cpu",
            )["best_mean_psnr"],
            abs=1.0e-6,
        )


@pytest.mark.parametrize(
    ("script", "required_option"),
    [
        ("train.py", "--resume"),
        ("render.py", "--checkpoint"),
        ("evaluate.py", "--output"),
    ],
)
def test_cli_help_imports(script: str, required_option: str) -> None:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_path
        else os.pathsep.join((source_path, existing_path))
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert required_option in completed.stdout
