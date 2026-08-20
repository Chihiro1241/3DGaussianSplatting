from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import Tensor

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.io.checkpoint import (
    load_checkpoint,
    model_from_checkpoint_state,
    read_checkpoint,
)
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GaussianModel,
)
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
)
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer
from scripts.train import _density_statistics_for_model


ROOT = Path(__file__).resolve().parents[2]


def _config(
    *,
    adaptive_density_control: bool,
    opacity_reset: bool,
    iterations: int = 3,
    densify_from: int = 1,
    densify_until: int = 4,
    densification_interval: int = 2,
    gradient_threshold: float = 0.0,
    percent_dense: float = 1.0,
    opacity_reset_interval: int = 2,
    log_interval: int = 10,
) -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        runtime=replace(config.runtime, device="cpu"),
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(
            config.training,
            iterations=iterations,
            log_interval=log_interval,
            evaluation_interval=iterations,
            checkpoint_interval=iterations,
        ),
        density_control=replace(
            config.density_control,
            densify_from_iteration=densify_from,
            densify_until_iteration=densify_until,
            densification_interval=densification_interval,
            position_gradient_threshold=gradient_threshold,
            percent_dense=percent_dense,
            prune_opacity_threshold=0.0,
            opacity_reset_interval=opacity_reset_interval,
            opacity_reset_maximum=0.01,
        ),
        features=replace(
            config.features,
            adaptive_density_control=adaptive_density_control,
            opacity_reset=opacity_reset,
        ),
        output=replace(config.output, save_rendered_images=False),
    )


def _model() -> GaussianModel:
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


def _camera(center_x: float, name: str) -> Camera:
    center = torch.tensor([center_x, 0.0, 0.0])
    image = torch.full((3, 7, 7), 0.25)
    image[0].add_(0.05)
    image[2].sub_(0.05)
    return Camera(
        rotation_cw=torch.eye(3),
        translation_cw=-center,
        camera_center_world=center,
        fx=8.0,
        fy=8.0,
        cx=3.0,
        cy=3.0,
        width=7,
        height=7,
        image=image,
        image_name=name,
    )


def _cameras() -> list[Camera]:
    return [_camera(-1.0, "left.png"), _camera(1.0, "right.png")]


def _training_objects(
    model: GaussianModel,
    config: Config,
) -> tuple[torch.optim.Adam, PositionLearningRateScheduler]:
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    return optimizer, scheduler


def _trainer(
    output: Path,
    model: GaussianModel,
    config: Config,
    *,
    optimizer: torch.optim.Adam | None = None,
    scheduler: PositionLearningRateScheduler | None = None,
    statistics: ScreenSpaceDensityStatistics | None = None,
    start_iteration: int = 0,
    camera_order: list[int] | None = None,
    camera_cursor: int = 0,
) -> Trainer:
    if optimizer is None or scheduler is None:
        optimizer, scheduler = _training_objects(model, config)
    cameras = _cameras()
    return Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=cameras,
        evaluation_cameras=[cameras[0]],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=output,
        start_iteration=start_iteration,
        camera_order=[0, 1] if camera_order is None else camera_order,
        camera_cursor=camera_cursor,
        density_statistics=statistics,
    )


def _optimizer_states(
    trainer: Trainer,
) -> dict[str, dict[str, object]]:
    return {
        name: trainer.optimizer.state[getattr(trainer.model, name)]
        for name in GAUSSIAN_PARAMETER_NAMES
    }


def _run_steps(trainer: Trainer, first: int, last: int) -> None:
    for iteration in range(first, last + 1):
        _, camera = trainer._next_camera()
        trainer.train_step(camera, iteration)


def test_real_density_event_replaces_parameters_and_next_step_succeeds(
    tmp_path: Path,
) -> None:
    torch.manual_seed(101)
    config = _config(
        adaptive_density_control=True,
        opacity_reset=False,
        percent_dense=1.0,
    )
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = _trainer(tmp_path, model, config, statistics=statistics)

    trainer.train_step(trainer.train_cameras[0], iteration=1)
    old_parameters = model.gaussian_parameter_dict()
    event_step = trainer.train_step(trainer.train_cameras[0], iteration=2)

    assert event_step.density_control_result is not None
    assert event_step.density_control_result.num_cloned == 2
    assert event_step.density_control_result.num_split_parents == 0
    assert model.num_gaussians == 4
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        assert parameter is not old_parameters[name]
        assert parameter.grad is None
        group = next(
            group for group in trainer.optimizer.param_groups
            if group["name"] == name
        )
        assert group["params"][0] is parameter
        state = trainer.optimizer.state[parameter]
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq"].shape == parameter.shape
    assert trainer.scheduler._position_group()["lr"] == (
        event_step.position_learning_rate
    )
    assert torch.count_nonzero(statistics.position_gradient_accumulator).item() == 0
    assert torch.count_nonzero(statistics.position_gradient_denominator).item() == 0
    assert torch.count_nonzero(statistics.max_screen_radius).item() == 0

    next_step = trainer.train_step(trainer.train_cameras[0], iteration=3)

    assert torch.isfinite(next_step.loss.total)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.count_nonzero(statistics.position_gradient_denominator).item() > 0


def test_real_opacity_reset_discards_old_gradient_and_training_continues(
    tmp_path: Path,
) -> None:
    config = _config(
        adaptive_density_control=False,
        opacity_reset=True,
        opacity_reset_interval=2,
    )
    model = _model()
    trainer = _trainer(tmp_path, model, config)

    trainer.train_step(trainer.train_cameras[0], iteration=1)
    old_opacity = model.raw_opacities
    assert old_opacity.grad is not None
    old_step = trainer.optimizer.state[old_opacity]["step"].detach().clone()
    reset_step = trainer.train_step(trainer.train_cameras[0], iteration=2)

    assert reset_step.opacity_reset_result is not None
    assert reset_step.density_control_result is None
    assert model.raw_opacities is not old_opacity
    assert model.raw_opacities.grad is None
    assert old_opacity not in trainer.optimizer.state
    opacity_state = trainer.optimizer.state[model.raw_opacities]
    assert torch.count_nonzero(opacity_state["exp_avg"]).item() == 0
    assert torch.count_nonzero(opacity_state["exp_avg_sq"]).item() == 0
    torch.testing.assert_close(
        opacity_state["step"], old_step, rtol=0.0, atol=0.0
    )
    assert torch.all(torch.sigmoid(model.raw_opacities) <= 0.01).item()

    continuation = trainer.train_step(trainer.train_cameras[0], iteration=3)

    assert torch.isfinite(continuation.loss.total)
    assert model.raw_opacities.grad is not None


def test_event_logging_and_trainer_checkpoint_capture_post_event_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(102)
    config = _config(
        adaptive_density_control=True,
        opacity_reset=True,
        percent_dense=1.0,
        log_interval=10,
    )
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = _trainer(tmp_path, model, config, statistics=statistics)

    trainer.train()

    records = [
        json.loads(line)
        for line in (tmp_path / "train_log.jsonl").read_text().splitlines()
    ]
    assert [record["iteration"] for record in records] == [2, 3]
    event_record = records[0]
    assert event_record["density_control_event"] is True
    assert event_record["density_num_gaussians_before"] == 2
    assert event_record["density_num_gaussians_after"] == 4
    assert event_record["density_num_cloned"] == 2
    assert event_record["density_num_split_parents"] == 0
    assert event_record["density_num_children_created"] == 0
    assert event_record["density_num_pruned_total"] == 0
    assert event_record["density_num_low_opacity"] == 0
    assert event_record["density_num_large_screen"] == 0
    assert event_record["density_num_large_world"] == 0
    assert event_record["opacity_reset"] is True
    assert event_record["opacity_num_clamped"] == 4
    assert event_record["opacity_reset_maximum"] == pytest.approx(0.01)
    assert event_record["gaussian_count"] == 4
    assert "density_control_event" not in records[1]
    assert "opacity_reset" not in records[1]

    checkpoint = read_checkpoint(
        tmp_path / "checkpoints" / "latest.pt",
        map_location="cpu",
    )
    density_state = checkpoint["density_statistics_state"]
    assert isinstance(density_state, dict)
    assert density_state["position_gradient_accumulator"].shape == (4,)
    assert torch.count_nonzero(
        density_state["position_gradient_denominator"]
    ).item() > 0
    best_checkpoint = read_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
    )
    assert isinstance(best_checkpoint["density_statistics_state"], dict)
    dynamic_model = model_from_checkpoint_state(
        checkpoint,
        config,
        device="cpu",
        dtype=torch.float32,
    )
    dynamic_statistics = _density_statistics_for_model(dynamic_model, config)
    assert dynamic_model.num_gaussians == 4
    assert dynamic_statistics is not None
    assert dynamic_statistics.num_gaussians == 4


@pytest.mark.parametrize(
    ("adaptive_density_control", "opacity_reset", "expected_field"),
    [
        (True, False, "density_control_event"),
        (False, True, "opacity_reset"),
    ],
)
def test_density_event_and_opacity_reset_independently_force_logging(
    tmp_path: Path,
    adaptive_density_control: bool,
    opacity_reset: bool,
    expected_field: str,
) -> None:
    config = _config(
        adaptive_density_control=adaptive_density_control,
        opacity_reset=opacity_reset,
        percent_dense=1.0,
        log_interval=10,
    )
    model = _model()
    statistics = _density_statistics_for_model(model, config)
    trainer = _trainer(
        tmp_path,
        model,
        config,
        statistics=statistics,
    )

    trainer.train()

    records = [
        json.loads(line)
        for line in (tmp_path / "train_log.jsonl").read_text().splitlines()
    ]
    assert [record["iteration"] for record in records] == [2, 3]
    assert records[0][expected_field] is True
    other_field = (
        "opacity_reset"
        if expected_field == "density_control_event"
        else "density_control_event"
    )
    assert other_field not in records[0]


def test_fixed_mode_log_schema_cadence_and_checkpoint_remain_unchanged(
    tmp_path: Path,
) -> None:
    config = _config(
        adaptive_density_control=False,
        opacity_reset=False,
        log_interval=2,
    )
    model = _model()
    trainer = _trainer(tmp_path, model, config)

    trainer.train()

    records = [
        json.loads(line)
        for line in (tmp_path / "train_log.jsonl").read_text().splitlines()
    ]
    assert [record["iteration"] for record in records] == [2, 3]
    for record in records:
        assert "density_control_event" not in record
        assert "opacity_reset" not in record
    checkpoint = read_checkpoint(
        tmp_path / "checkpoints" / "latest.pt",
        map_location="cpu",
    )
    assert checkpoint["density_statistics_state"] is None


def test_split_resume_from_mid_window_is_deterministic(tmp_path: Path) -> None:
    config = _config(
        adaptive_density_control=True,
        opacity_reset=False,
        percent_dense=0.01,
    )

    random.seed(110)
    np.random.seed(111)
    torch.manual_seed(112)
    continuous_model = _model()
    continuous_statistics = _density_statistics_for_model(
        continuous_model, config
    )
    assert continuous_statistics is not None
    continuous = _trainer(
        tmp_path / "continuous",
        continuous_model,
        config,
        statistics=continuous_statistics,
    )
    _run_steps(continuous, 1, 3)
    continuous_rng = torch.get_rng_state().clone()

    random.seed(110)
    np.random.seed(111)
    torch.manual_seed(112)
    interrupted_model = _model()
    interrupted_statistics = _density_statistics_for_model(
        interrupted_model, config
    )
    assert interrupted_statistics is not None
    interrupted = _trainer(
        tmp_path / "interrupted",
        interrupted_model,
        config,
        statistics=interrupted_statistics,
    )
    _run_steps(interrupted, 1, 1)
    interrupted.save_checkpoint(iteration=1)
    checkpoint_path = (
        tmp_path / "interrupted" / "checkpoints" / "iteration_00000001.pt"
    )
    state = read_checkpoint(checkpoint_path, map_location="cpu")
    assert torch.count_nonzero(
        state["density_statistics_state"]["position_gradient_denominator"]
    ).item() > 0

    random.seed(900)
    np.random.seed(901)
    torch.manual_seed(902)
    resumed_model = model_from_checkpoint_state(
        state,
        config,
        device="cpu",
        dtype=torch.float32,
    )
    resumed_optimizer, resumed_scheduler = _training_objects(
        resumed_model, config
    )
    resumed_statistics = _density_statistics_for_model(resumed_model, config)
    assert resumed_statistics is not None
    loaded = load_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        density_statistics=resumed_statistics,
        map_location="cpu",
        restore_random_state=True,
    )
    resumed = _trainer(
        tmp_path / "resumed",
        resumed_model,
        config,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        statistics=resumed_statistics,
        start_iteration=int(loaded["iteration"]),
        camera_order=list(loaded["camera_order"]),
        camera_cursor=int(loaded["camera_cursor"]),
    )
    _run_steps(resumed, 2, 3)

    assert continuous.model.num_gaussians == resumed.model.num_gaussians == 4
    for name, expected in continuous.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name], expected, rtol=0.0, atol=0.0
        )
    continuous_states = _optimizer_states(continuous)
    resumed_states = _optimizer_states(resumed)
    for name, expected_state in continuous_states.items():
        for key, expected in expected_state.items():
            actual = resumed_states[name][key]
            if isinstance(expected, Tensor):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            else:
                assert actual == expected
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()
    for name, expected in continuous.density_statistics.state_dict().items():
        torch.testing.assert_close(
            getattr(resumed.density_statistics, name),
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert resumed.camera_order == continuous.camera_order
    assert resumed.camera_cursor == continuous.camera_cursor
    assert torch.equal(torch.get_rng_state(), continuous_rng)


def test_script_statistics_factory_matches_new_and_dynamic_resume_models() -> None:
    fixed = _config(
        adaptive_density_control=False,
        opacity_reset=False,
    )
    adaptive = _config(
        adaptive_density_control=True,
        opacity_reset=False,
    )
    model = _model()

    assert _density_statistics_for_model(model, fixed) is None
    statistics = _density_statistics_for_model(model, adaptive)

    assert statistics is not None
    assert statistics.num_gaussians == model.num_gaussians
    assert statistics.dtype == model.means_world.dtype
    assert statistics.device == model.means_world.device
