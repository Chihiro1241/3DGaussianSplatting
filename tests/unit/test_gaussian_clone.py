from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from gaussian_splatting.config import load_config
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GaussianModel,
)
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
    clone_gaussians,
)
from gaussian_splatting.training.optimizer import create_optimizer


ROOT = Path(__file__).resolve().parents[2]


def _model(
    scales: list[list[float]],
    *,
    dtype: torch.dtype = torch.float32,
) -> GaussianModel:
    count = len(scales)
    row = torch.arange(count, dtype=dtype).unsqueeze(1)
    means_world = torch.cat((row, row + 0.25, row + 0.5), dim=1)
    raw_quaternions = torch.cat(
        (
            torch.ones((count, 1), dtype=dtype),
            row + 1.0,
            row + 2.0,
            row + 3.0,
        ),
        dim=1,
    )
    if count == 0:
        scale_tensor = torch.empty((0, 3), dtype=dtype)
    else:
        scale_tensor = torch.tensor(scales, dtype=dtype)
    raw_scales = torch.log(scale_tensor)
    raw_opacities = torch.linspace(
        -1.0, 1.0, count, dtype=dtype
    ).unsqueeze(1)
    sh_dc = row.reshape(count, 1, 1).expand(-1, 1, 3).clone() + 20.0
    sh_rest = row.reshape(count, 1, 1).expand(-1, 15, 3).clone() + 40.0
    return GaussianModel(
        means_world=means_world,
        raw_quaternions=raw_quaternions,
        raw_scales=raw_scales,
        raw_opacities=raw_opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
    )


def _optimizer(model: GaussianModel) -> torch.optim.Adam:
    return create_optimizer(model, load_config(ROOT / "configs" / "default.yaml"))


def _statistics(
    model: GaussianModel,
    mean_gradients: list[float],
) -> ScreenSpaceDensityStatistics:
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.position_gradient_accumulator.copy_(
        torch.tensor(mean_gradients, dtype=model.means_world.dtype)
    )
    statistics.position_gradient_denominator.fill_(1)
    statistics.max_screen_radius.copy_(
        torch.arange(10, 10 + model.num_gaussians, dtype=torch.int64)
    )
    return statistics


def _initialize_adam_state(
    model: GaussianModel, optimizer: torch.optim.Adam
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def _groups(optimizer: torch.optim.Adam) -> dict[str, dict[str, object]]:
    return {str(group["name"]): group for group in optimizer.param_groups}


def _values(model: GaussianModel) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.gaussian_parameter_dict().items()
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clone_selects_intersection_copies_raw_rows_and_preserves_order(
    dtype: torch.dtype,
) -> None:
    model = _model(
        [[0.5, 0.7, 0.8], [2.0, 1.0, 1.0], [0.4, 0.5, 0.6], [1.0, 1.0, 1.0]],
        dtype=dtype,
    )
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 2.0, 0.5, 1.0])
    old_values = _values(model)
    old_accumulator = statistics.position_gradient_accumulator.clone()
    old_denominator = statistics.position_gradient_denominator.clone()
    old_radius = statistics.max_screen_radius.clone()
    parent_indices = torch.tensor([0, 3])
    assert model.raw_scales[1].detach().amax().item() < 1.0

    result = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert result.num_gaussians_before == 4
    assert result.num_gaussians_after == 6
    assert result.num_cloned == 2
    assert model.num_gaussians == 6
    assert len(optimizer.state) == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        torch.testing.assert_close(
            parameter[:4], old_values[name], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            parameter[4:], old_values[name][parent_indices], rtol=0.0, atol=0.0
        )
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert parameter.is_leaf
    torch.testing.assert_close(
        statistics.position_gradient_accumulator[:4], old_accumulator
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator[:4], old_denominator
    )
    torch.testing.assert_close(statistics.max_screen_radius[:4], old_radius)
    assert torch.count_nonzero(
        statistics.position_gradient_accumulator[4:]
    ).item() == 0
    assert torch.count_nonzero(
        statistics.position_gradient_denominator[4:]
    ).item() == 0
    assert torch.count_nonzero(statistics.max_screen_radius[4:]).item() == 0


def test_clone_zero_initializes_child_adam_moments_and_preserves_parent_state() -> None:
    model = _model(
        [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [2.0, 2.0, 2.0]]
    )
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [0.0, 2.0, 2.0])
    old_parameters = model.gaussian_parameter_dict()
    old_states = {
        name: {
            state_name: value.detach().clone()
            if isinstance(value, Tensor)
            else value
            for state_name, value in optimizer.state[parameter].items()
        }
        for name, parameter in old_parameters.items()
    }

    result = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert result.num_cloned == 1
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        state = optimizer.state[parameter]
        assert old_parameters[name] not in optimizer.state
        assert _groups(optimizer)[name]["params"][0] is parameter
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state[moment][:3],
                old_states[name][moment],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                state[moment][3:],
                torch.zeros_like(parameter[3:]),
                rtol=0.0,
                atol=0.0,
            )
            assert state[moment].shape == parameter.shape
        torch.testing.assert_close(
            state["step"], old_states[name]["step"], rtol=0.0, atol=0.0
        )


def test_clone_noop_preserves_all_references() -> None:
    model = _model([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [0.0, 0.5])
    parameters = model.gaussian_parameter_dict()
    optimizer_states = {
        name: optimizer.state[parameter]
        for name, parameter in parameters.items()
    }
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )

    result = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert result.num_gaussians_before == 2
    assert result.num_gaussians_after == 2
    assert result.num_cloned == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        assert optimizer.state[parameters[name]] is optimizer_states[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


def test_two_clone_operations_preserve_index_alignment() -> None:
    model = _model([[0.5, 0.5, 0.5]] * 3)
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [0.0, 2.0, 0.0])
    original = _values(model)

    first = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )
    assert first.num_cloned == 1
    statistics.position_gradient_accumulator.zero_()
    statistics.position_gradient_denominator.fill_(1)
    statistics.position_gradient_accumulator[0] = 3.0

    second = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert second.num_gaussians_before == 4
    assert second.num_gaussians_after == 5
    assert second.num_cloned == 1
    for name in GAUSSIAN_PARAMETER_NAMES:
        expected = torch.cat(
            (original[name], original[name][1:2], original[name][0:1]), dim=0
        )
        torch.testing.assert_close(
            getattr(model, name), expected, rtol=0.0, atol=0.0
        )
        parameter = getattr(model, name)
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape
    assert statistics.num_gaussians == 5


@pytest.mark.parametrize(
    "thresholds,expected_exception,message",
    [
        (
            {"gradient_threshold": True, "world_scale_threshold": 1.0},
            TypeError,
            "real number",
        ),
        (
            {"gradient_threshold": float("nan"), "world_scale_threshold": 1.0},
            ValueError,
            "finite",
        ),
        (
            {"gradient_threshold": float("inf"), "world_scale_threshold": 1.0},
            ValueError,
            "finite",
        ),
        (
            {"gradient_threshold": -1.0, "world_scale_threshold": 1.0},
            ValueError,
            "non-negative",
        ),
        (
            {"gradient_threshold": 1.0, "world_scale_threshold": False},
            TypeError,
            "real number",
        ),
        (
            {"gradient_threshold": 1.0, "world_scale_threshold": float("nan")},
            ValueError,
            "finite",
        ),
        (
            {"gradient_threshold": 1.0, "world_scale_threshold": float("inf")},
            ValueError,
            "finite",
        ),
        (
            {"gradient_threshold": 1.0, "world_scale_threshold": 0.0},
            ValueError,
            "positive",
        ),
        (
            {"gradient_threshold": 1.0, "world_scale_threshold": -1.0},
            ValueError,
            "positive",
        ),
    ],
)
def test_invalid_thresholds_fail_before_mutation(
    thresholds: dict[str, object],
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model([[0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0])
    parameters = model.gaussian_parameter_dict()
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )

    with pytest.raises(expected_exception, match=message):
        clone_gaussians(model, optimizer, statistics, **thresholds)

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


@pytest.mark.parametrize("mismatch", ["count", "dtype", "device"])
def test_incompatible_statistics_fail_before_mutation(mismatch: str) -> None:
    model = _model([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    if mismatch == "count":
        statistics = ScreenSpaceDensityStatistics(
            3, dtype=torch.float32, device="cpu"
        )
    elif mismatch == "dtype":
        statistics = ScreenSpaceDensityStatistics(
            2, dtype=torch.float64, device="cpu"
        )
    else:
        statistics = ScreenSpaceDensityStatistics(
            2, dtype=torch.float32, device="meta"
        )
    parameters = model.gaussian_parameter_dict()

    with pytest.raises((TypeError, ValueError), match=mismatch):
        clone_gaussians(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            world_scale_threshold=1.0,
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]


def test_statistics_commit_failure_rolls_back_append_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 0.0])
    parameters = model.gaussian_parameter_dict()
    optimizer_states = {
        name: optimizer.state[parameter]
        for name, parameter in parameters.items()
    }
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )

    def fail_commit(state: object) -> None:
        raise RuntimeError("intentional statistics commit failure")

    monkeypatch.setattr(statistics, "_commit_state", fail_commit)

    with pytest.raises(RuntimeError, match="intentional statistics"):
        clone_gaussians(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            world_scale_threshold=1.0,
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        assert optimizer.state[parameters[name]] is optimizer_states[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


def test_clone_on_zero_gaussian_model_is_a_safe_noop() -> None:
    model = _model([])
    optimizer = _optimizer(model)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    parameters = model.gaussian_parameter_dict()
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )

    result = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=0.0,
        world_scale_threshold=1.0,
    )

    assert result.num_gaussians_before == 0
    assert result.num_gaussians_after == 0
    assert result.num_cloned == 0
    assert len(optimizer.state) == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]
