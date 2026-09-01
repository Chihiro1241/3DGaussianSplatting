from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.profiling import (
    CORE_STAGES,
    TrainingProfiler,
    WallClockIterationTimer,
)
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[2]


def _config() -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(config.training, iterations=1),
        features=replace(
            config.features,
            adaptive_density_control=False,
            opacity_reset=False,
        ),
    )


def _model() -> GaussianModel:
    return GaussianModel(
        means_world=torch.tensor([[-0.1, 0.0, 2.0], [0.15, 0.08, 2.3]]),
        raw_quaternions=torch.tensor(
            [[1.0, 0.1, 0.05, 0.0], [1.0, 0.0, 0.2, -0.1]]
        ),
        raw_scales=torch.log(torch.tensor([[0.12, 0.08, 0.1], [0.1, 0.14, 0.08]])),
        raw_opacities=torch.logit(torch.tensor([[0.35], [0.45]])),
        sh_dc=torch.tensor([[[0.1, -0.05, 0.0]], [[-0.08, 0.02, 0.12]]]),
        sh_rest=torch.zeros((2, 15, 3)),
    )


def _camera() -> Camera:
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
        image=torch.full((3, 7, 7), 0.25),
        image_name="tiny.png",
    )


def _trainer(model: GaussianModel, output: Path, *, profile: bool) -> Trainer:
    config = _config()
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=1,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    camera = _camera()
    return Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[camera],
        evaluation_cameras=[],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=output,
        camera_order=[0],
        profile_training=profile,
        profile_warmup=0,
        profile_steps=1,
    )


def test_profiled_step_preserves_loss_render_and_parameter_update(tmp_path: Path) -> None:
    plain_model = _model()
    profiled_model = _model()
    plain = _trainer(plain_model, tmp_path / "plain", profile=False)
    profiled = _trainer(profiled_model, tmp_path / "profiled", profile=True)

    plain_result = plain.train_step(_camera(), 1)
    profiled_result = profiled.train_step(_camera(), 1)

    torch.testing.assert_close(profiled_result.loss.total, plain_result.loss.total)
    torch.testing.assert_close(profiled_result.render.image, plain_result.render.image)
    for plain_parameter, profiled_parameter in zip(
        plain_model.parameters(), profiled_model.parameters(), strict=True
    ):
        torch.testing.assert_close(profiled_parameter, plain_parameter)

    profile_path = tmp_path / "profiled" / "profiling" / "training_profile.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["environment"]["device"] == "cpu"
    assert payload["environment"]["gpu_name"] is None
    assert payload["gaussian_counts"] == [2]
    assert payload["resolutions"] == [{"height": 7, "width": 7}]
    assert payload["iterations"][0]["visible_gaussian_count"] <= 2
    assert payload["summaries"]["normal_iterations"]["iteration_count"] == 1
    assert payload["rasterization_share_percent"] >= 0.0
    for stage in (*CORE_STAGES, "other", "total"):
        assert payload["iterations"][0]["stage_times_ms"][stage] >= 0.0


def test_disabled_cpu_profiling_never_synchronizes_cuda(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_synchronize(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA synchronization must not run")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_synchronize)
    trainer = _trainer(_model(), tmp_path, profile=False)
    trainer.train_step(_camera(), 1)


def test_wall_clock_timer_synchronizes_only_for_cuda(monkeypatch) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))

    cpu_timer = WallClockIterationTimer(torch.device("cpu"))
    for stage in CORE_STAGES:
        with cpu_timer.measure(stage):
            pass
    cpu_timer.finish()
    assert calls == []

    cuda_timer = WallClockIterationTimer(torch.device("cuda:0"))
    for stage in CORE_STAGES:
        with cuda_timer.measure(stage):
            pass
    cuda_timer.finish()
    assert len(calls) == 14
    assert all(device == torch.device("cuda:0") for device in calls)


def test_warmup_and_density_event_are_excluded_from_normal_summary(
    tmp_path: Path,
) -> None:
    profiler = TrainingProfiler(
        output_directory=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        warmup_steps=1,
        profile_steps=1,
    )
    stage_times = {stage: 1.0 for stage in CORE_STAGES}
    stage_times.update(other=2.0, total=8.0)
    for iteration, density_event in ((1, False), (2, True), (3, False)):
        profiler.record_iteration(
            iteration=iteration,
            gaussian_count=2,
            visible_gaussian_count=2,
            width=7,
            height=7,
            density_control_event=density_event,
            opacity_reset_event=False,
            stage_times=dict(stage_times),
        )

    payload = json.loads(profiler.output_path.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    assert summaries["normal_iterations"]["iteration_count"] == 1
    assert summaries["density_control_event_iterations"]["iteration_count"] == 1
    assert payload["iterations"][0]["phase"] == "warmup"
    assert payload["rasterization_share_percent"] == 12.5
