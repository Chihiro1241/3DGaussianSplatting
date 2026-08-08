from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from gaussian_splatting.config import load_config
from gaussian_splatting.io.checkpoint import (
    load_checkpoint,
    model_from_checkpoint_state,
    read_checkpoint,
    save_checkpoint,
)
from gaussian_splatting.io.ply_io import load_gaussians_ply, save_gaussians_ply
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler


ROOT = Path(__file__).resolve().parents[2]


def _model(dtype: torch.dtype = torch.float32) -> GaussianModel:
    torch.manual_seed(0)
    count = 4
    return GaussianModel(
        means_world=torch.randn((count, 3), dtype=dtype),
        raw_quaternions=torch.randn((count, 4), dtype=dtype),
        raw_scales=torch.randn((count, 3), dtype=dtype),
        raw_opacities=torch.randn((count, 1), dtype=dtype),
        sh_dc=torch.randn((count, 1, 3), dtype=dtype),
        sh_rest=torch.randn((count, 15, 3), dtype=dtype),
    )


def test_create_optimizer_has_documented_parameter_groups() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    model = _model()
    optimizer = create_optimizer(model, config)
    assert [group["name"] for group in optimizer.param_groups] == [
        "means_world",
        "sh_dc",
        "sh_rest",
        "raw_opacities",
        "raw_scales",
        "raw_quaternions",
    ]
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1.0e-15
    assert optimizer.defaults["weight_decay"] == 0.0


def test_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    model = _model()
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(optimizer)
    scheduler.step(17)
    path = tmp_path / "checkpoint.pt"

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    save_checkpoint(
        path,
        iteration=17,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        camera_order=[2, 0, 1],
        camera_cursor=1,
        best_mean_psnr=23.5,
    )
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_random = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(())

    with torch.no_grad():
        model.means_world.add_(100.0)
    random.random()
    np.random.rand()
    torch.rand(())
    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location="cpu",
    )
    assert state["iteration"] == 17
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0.0, atol=0.0)
    assert random.random() == expected_random
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(()), expected_torch, rtol=0.0, atol=0.0)


def test_model_from_checkpoint_avoids_new_random_initialization(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    model = _model(torch.float64)
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(optimizer)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        iteration=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        camera_order=[0],
        camera_cursor=0,
        best_mean_psnr=None,
    )
    state = read_checkpoint(path, map_location="cpu")
    restored = model_from_checkpoint_state(
        state, config, device="cpu", dtype=torch.float64
    )
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0.0, atol=0.0)


def test_ply_round_trip_preserves_all_raw_parameters(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "point_cloud.ply"
    save_gaussians_ply(path, model)
    restored = load_gaussians_ply(path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=1e-5, atol=1e-6)

