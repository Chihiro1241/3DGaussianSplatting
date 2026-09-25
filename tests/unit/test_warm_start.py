"""The warm-start overrides used by the 4D iteration-budget experiments.

Two properties matter throughout: a carried-over frame gets exactly the
behaviour ``config.warm_start`` asks for, and a configuration that says nothing
about warm start -- which is every configuration the static reproduction uses --
behaves as it always did.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from gaussian_splatting.config import (
    DEFAULT_WARM_START,
    ConfigError,
    config_from_mapping,
    load_config,
)
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import (
    FixedPositionLearningRateScheduler,
    PositionLearningRateScheduler,
)
from trainer_4d import (
    FrameHandoff,
    assert_frame_state,
    assert_frame_state_is_reset,
    build_frame_training_state,
    install_adam_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIRECTORY / "default.yaml"
WARM_START_CONFIG = CONFIG_DIRECTORY / "neu3d" / "warmstart_sweep.yaml"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_every_shipped_configuration_still_loads() -> None:
    """The optional section must not have broken the strict loader."""

    paths = sorted(CONFIG_DIRECTORY.rglob("*.yaml"))
    assert paths, "no configuration files were found"
    for path in paths:
        config = load_config(path)
        # asdict is also the checkpoint reconstruction path.
        assert config_from_mapping(asdict(config)) == config, path


def test_omitted_warm_start_section_takes_the_documented_defaults() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.warm_start == DEFAULT_WARM_START
    assert config.warm_start.position_lr_mode == "exponential"
    assert config.warm_start.adam_state == "reset"


def test_warm_start_sweep_configuration_disables_every_varying_feature() -> None:
    """The sweep's own configuration must hold everything but the budget fixed."""

    config = load_config(WARM_START_CONFIG)

    assert config.features.adaptive_density_control is False
    assert config.features.opacity_reset is False
    assert config.features.progressive_sh_degree is False
    assert config.features.resolution_warmup is False
    assert config.warm_start.position_lr_mode == "fixed"
    # A per-frame checkpoint interval larger than any budget we sweep keeps one
    # checkpoint per frame however many iterations the command line asks for.
    assert config.training.checkpoint_interval > 30_000


def _mapping_with_warm_start(**overrides: object) -> dict[str, object]:
    values = asdict(load_config(DEFAULT_CONFIG))
    values["warm_start"] = {
        "position_lr_mode": "fixed",
        "position_lr_fixed": 1.6e-5,
        "adam_state": "reset",
        **overrides,
    }
    return values


def test_warm_start_section_is_checked_as_strictly_as_the_others() -> None:
    values = _mapping_with_warm_start()
    del values["warm_start"]["adam_state"]
    with pytest.raises(ConfigError, match="missing keys: adam_state"):
        config_from_mapping(values)

    with pytest.raises(ConfigError, match="unknown keys: typo"):
        config_from_mapping(_mapping_with_warm_start(typo=1))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"position_lr_mode": "cosine"}, "position_lr_mode"),
        ({"position_lr_fixed": -1.0}, "position_lr_fixed"),
        ({"position_lr_fixed": float("inf")}, "position_lr_fixed"),
        ({"adam_state": "keep"}, "adam_state"),
    ],
)
def test_warm_start_values_are_validated(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_mapping(_mapping_with_warm_start(**overrides))


# --------------------------------------------------------------------------
# the fixed position schedule
# --------------------------------------------------------------------------


def _named_optimizer() -> tuple[torch.nn.Parameter, torch.optim.Adam]:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "name": "means_world", "lr": 1.6e-4}]
    )
    return parameter, optimizer


def test_fixed_schedule_is_constant_and_budget_independent() -> None:
    """The whole point: the budget must stop moving the learning rate."""

    _, optimizer = _named_optimizer()
    short = FixedPositionLearningRateScheduler(
        optimizer, learning_rate=2.5e-4, total_iterations=100
    )
    long = FixedPositionLearningRateScheduler(
        optimizer, learning_rate=2.5e-4, total_iterations=30_000
    )

    rates = {short.step(i) for i in (0, 1, 50, 100, 10_000)}
    assert rates == {2.5e-4}
    assert long.step(1) == short.step(1)
    assert optimizer.param_groups[0]["lr"] == 2.5e-4


def test_fixed_schedule_rejects_an_unusable_optimizer_at_construction() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    unnamed = torch.optim.Adam([{"params": [parameter], "lr": 1.0}])
    with pytest.raises(ValueError, match="means_world"):
        FixedPositionLearningRateScheduler(unnamed, learning_rate=1e-4)


@pytest.mark.parametrize("learning_rate", [0.0, -1.0, float("nan")])
def test_fixed_schedule_rejects_an_unusable_rate(learning_rate: float) -> None:
    _, optimizer = _named_optimizer()
    with pytest.raises(ValueError, match="learning_rate"):
        FixedPositionLearningRateScheduler(optimizer, learning_rate=learning_rate)


def test_fixed_schedule_round_trips_its_own_state() -> None:
    _, optimizer = _named_optimizer()
    scheduler = FixedPositionLearningRateScheduler(
        optimizer, learning_rate=2.5e-4, total_iterations=500
    )
    scheduler.step(321)

    restored = FixedPositionLearningRateScheduler(
        optimizer, learning_rate=2.5e-4, total_iterations=500
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.current_iteration == 321


def test_fixed_schedule_refuses_an_exponential_scheduler_state() -> None:
    """Loading the wrong schedule's state would silently change the run."""

    _, optimizer = _named_optimizer()
    exponential = PositionLearningRateScheduler(optimizer, total_iterations=500)
    fixed = FixedPositionLearningRateScheduler(
        optimizer, learning_rate=2.5e-4, total_iterations=500
    )
    with pytest.raises(ValueError, match="fixed position schedule"):
        fixed.load_state_dict(exponential.state_dict())


def test_exponential_schedule_is_unchanged() -> None:
    """Guard the static path against drift in this file's neighbours."""

    _, optimizer = _named_optimizer()
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=30_000,
        initial_learning_rate=1.6e-4,
        final_learning_rate=1.6e-6,
    )
    assert scheduler.step(0) == pytest.approx(1.6e-4)
    assert scheduler.step(30_000) == pytest.approx(1.6e-6)
    assert set(scheduler.state_dict()) == {
        "current_iteration",
        "total_iterations",
        "initial_learning_rate",
        "final_learning_rate",
    }


# --------------------------------------------------------------------------
# frame construction
# --------------------------------------------------------------------------


def _cameras(count: int = 3) -> list[Camera]:
    cameras = []
    for index in range(count):
        center = torch.tensor([float(index), 0.0, 2.0])
        cameras.append(
            Camera(
                rotation_cw=torch.eye(3),
                translation_cw=-center,
                camera_center_world=center,
                fx=50.0, fy=50.0, cx=16.0, cy=16.0,
                width=32, height=32,
                image=None,
                image_name=f"cam{index:02d}.png",
            )
        )
    return cameras


def _model(count: int = 7) -> GaussianModel:
    generator = torch.Generator().manual_seed(0)
    return GaussianModel(
        means_world=torch.randn(count, 3, generator=generator),
        raw_quaternions=torch.randn(count, 4, generator=generator),
        raw_scales=torch.randn(count, 3, generator=generator) * 0.1,
        raw_opacities=torch.zeros(count, 1),
        sh_dc=torch.randn(count, 1, 3, generator=generator) * 0.1,
        sh_rest=torch.zeros(count, 15, 3),
        epsilon_q=1.0e-8,
        sh_degree=3,
    )


def _config(**warm_start: object):
    values = asdict(load_config(DEFAULT_CONFIG))
    values["features"] = dict(
        values["features"],
        adaptive_density_control=False,
        opacity_reset=False,
        progressive_sh_degree=False,
        resolution_warmup=False,
    )
    values["training"] = dict(values["training"], iterations=100)
    values["warm_start"] = {
        "position_lr_mode": "exponential",
        "position_lr_fixed": 1.6e-5,
        "adam_state": "reset",
        **warm_start,
    }
    return config_from_mapping(values)


def _stepped_handoff(count: int = 7) -> FrameHandoff:
    """A hand-off whose Adam moments are genuinely non-zero."""

    model = _model(count)
    optimizer = create_optimizer(model, _config(), position_lr_scale=1.0)
    for _ in range(4):
        optimizer.zero_grad()
        sum(parameter.sum() for parameter in model.parameters()).backward()
        optimizer.step()
    return FrameHandoff.from_model(
        model, source_frame=1, device="cpu", optimizer=optimizer
    )


def _build(config, handoff: FrameHandoff | None = None, **kwargs):
    return build_frame_training_state(
        config=config,
        train_cameras=_cameras(),
        device="cpu",
        dtype=torch.float32,
        handoff=handoff,
        **kwargs,
    )


def test_hand_off_without_an_optimizer_carries_no_moments() -> None:
    handoff = FrameHandoff.from_model(_model(), source_frame=1, device="cpu")
    assert handoff.optimizer_state is None


def test_hand_off_captures_every_named_parameter_group() -> None:
    handoff = _stepped_handoff()
    assert set(handoff.optimizer_state) == {
        "means_world", "sh_dc", "sh_rest",
        "raw_opacities", "raw_scales", "raw_quaternions",
    }
    for entry in handoff.optimizer_state.values():
        assert set(entry) == {"step", "exp_avg", "exp_avg_sq"}


def test_hand_off_rejects_moments_that_do_not_match_its_parameters() -> None:
    """A changed Gaussian count must be caught, not quietly mis-installed."""

    handoff = _stepped_handoff()
    with pytest.raises(ValueError, match="exp_avg"):
        replace(
            handoff,
            parameters={
                name: value[:5].clone() for name, value in handoff.parameters.items()
            },
        )


def test_carried_over_frame_uses_the_fixed_learning_rate() -> None:
    state = _build(_config(position_lr_mode="fixed"), _stepped_handoff())

    expected = 1.6e-5 * state.scene_extent
    assert state.position_learning_rate == pytest.approx(expected)
    assert isinstance(state.scheduler, FixedPositionLearningRateScheduler)
    assert state.scheduler.step(1) == pytest.approx(expected)
    assert state.scheduler.step(100) == pytest.approx(expected)


def test_the_budget_does_not_move_the_fixed_learning_rate() -> None:
    handoff = _stepped_handoff()
    values = asdict(_config(position_lr_mode="fixed"))

    rates = set()
    for iterations in (100, 1_000, 5_000):
        config = config_from_mapping(
            dict(values, training=dict(values["training"], iterations=iterations))
        )
        state = _build(config, handoff)
        rates.add(state.scheduler.step(1))
    assert len(rates) == 1


def test_first_frame_ignores_every_warm_start_override() -> None:
    """Frame 1 is an ordinary static run and must stay one."""

    state = _build(
        _config(position_lr_mode="fixed", adam_state="carry"),
        handoff=None,
        initial_points=torch.randn(9, 3),
        initial_colors=torch.rand(9, 3),
    )

    assert state.carried_over is False
    assert state.adam_state == "reset"
    assert state.position_learning_rate is None
    assert isinstance(state.scheduler, PositionLearningRateScheduler)
    # Check the untouched state before stepping the schedule moves it.
    assert_frame_state_is_reset(state)
    assert state.scheduler.step(1) != state.scheduler.step(100)


def test_reset_is_the_default_and_leaves_the_optimizer_empty() -> None:
    state = _build(_config(), _stepped_handoff())

    assert state.adam_state == "reset"
    assert len(state.optimizer.state) == 0
    assert_frame_state_is_reset(state)


def test_carry_installs_the_previous_frame_moments_unchanged() -> None:
    handoff = _stepped_handoff()
    state = _build(_config(adam_state="carry"), handoff)

    assert state.adam_state == "carry"
    assert len(state.optimizer.state) == len(handoff.optimizer_state)
    names = {
        group["params"][0]: group["name"] for group in state.optimizer.param_groups
    }
    for parameter, entry in state.optimizer.state.items():
        expected = handoff.optimizer_state[names[parameter]]
        assert torch.equal(entry["exp_avg"], expected["exp_avg"])
        assert torch.equal(entry["exp_avg_sq"], expected["exp_avg_sq"])
        assert torch.equal(entry["step"], expected["step"])
    assert_frame_state(state)


def test_the_strict_assertion_still_rejects_a_carry() -> None:
    state = _build(_config(adam_state="carry"), _stepped_handoff())
    with pytest.raises(RuntimeError, match="Adam moments must be empty"):
        assert_frame_state_is_reset(state)


def test_carry_requires_a_hand_off_that_actually_holds_moments() -> None:
    bare = FrameHandoff.from_model(_model(), source_frame=1, device="cpu")
    with pytest.raises(ValueError, match="requires a hand-off carrying"):
        _build(_config(adam_state="carry"), bare)


def test_carry_requires_density_control_to_be_off() -> None:
    """Densification would leave the inherited moments without a parameter."""

    values = asdict(_config(adam_state="carry"))
    values["features"] = dict(values["features"], adaptive_density_control=True)
    with pytest.raises(ValueError, match="adaptive_density_control"):
        _build(config_from_mapping(values), _stepped_handoff())


def test_installing_mismatched_moments_is_refused() -> None:
    model = _model(5)
    optimizer = create_optimizer(model, _config(), position_lr_scale=1.0)
    moments = {"means_world": {"exp_avg": torch.zeros(7, 3)}}
    with pytest.raises(ValueError, match="fixed Gaussian count"):
        install_adam_state(optimizer, moments)


def test_a_carried_over_frame_keeps_its_gaussian_count_and_sh_degree() -> None:
    handoff = _stepped_handoff(count=11)
    state = _build(_config(position_lr_mode="fixed"), handoff)

    assert state.num_gaussians == 11 == handoff.num_gaussians
    # Progressive SH is off, so the frame renders at the full degree from its
    # very first iteration rather than climbing back up to it.
    assert state.model.active_sh_degree == 3
    assert state.density_statistics is None
