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
    reset_gaussian_opacity,
)
from gaussian_splatting.training.optimizer import (
    create_optimizer,
    replace_named_gaussian_parameter,
)


ROOT = Path(__file__).resolve().parents[2]


def _model(
    opacities: list[float],
    *,
    dtype: torch.dtype = torch.float32,
) -> GaussianModel:
    count = len(opacities)
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
    raw_scales = torch.cat(
        (row, row + 0.1, row + 0.2), dim=1
    )
    raw_opacities = torch.logit(
        torch.tensor(opacities, dtype=dtype).reshape(count, 1)
    )
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


def _initialize_adam_state(
    model: GaussianModel, optimizer: torch.optim.Adam
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def _groups(optimizer: torch.optim.Adam) -> dict[str, dict[str, object]]:
    return {str(group["name"]): group for group in optimizer.param_groups}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_basic_reset_clamps_only_above_boundary_and_detaches_old_gradient(
    dtype: torch.dtype,
) -> None:
    model = _model([0.001, 0.01, 0.2], dtype=dtype)
    optimizer = _optimizer(model)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.position_gradient_accumulator.copy_(
        torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    )
    statistics.position_gradient_denominator.copy_(torch.tensor([4, 5, 6]))
    statistics.max_screen_radius.copy_(torch.tensor([7, 8, 9]))
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )
    statistic_values = tuple(value.clone() for value in statistic_references)
    parameters = model.gaussian_parameter_dict()
    old_raw_values = model.raw_opacities.detach().clone()
    model.raw_opacities.sum().backward()
    assert model.raw_opacities.grad is not None
    torch.manual_seed(30)
    rng_state = torch.get_rng_state().clone()

    result = reset_gaussian_opacity(
        model,
        optimizer,
        maximum_opacity=0.01,
    )

    assert result.num_gaussians == 3
    assert result.num_clamped == 1
    assert result.maximum_opacity == 0.01
    assert model.num_gaussians == 3
    assert model.raw_opacities is not parameters["raw_opacities"]
    assert model.raw_opacities.is_leaf
    assert model.raw_opacities.grad is None
    torch.testing.assert_close(
        torch.sigmoid(model.raw_opacities),
        torch.tensor([[0.001], [0.01], [0.01]], dtype=dtype),
    )
    torch.testing.assert_close(
        model.raw_opacities[:2], old_raw_values[:2], rtol=0.0, atol=0.0
    )
    for name in GAUSSIAN_PARAMETER_NAMES:
        if name != "raw_opacities":
            assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is getattr(model, name)
    assert parameters["raw_opacities"] not in optimizer.state
    assert len(optimizer.state) == 0
    for actual, expected_reference, expected_value in zip(
        (
            statistics.position_gradient_accumulator,
            statistics.position_gradient_denominator,
            statistics.max_screen_radius,
        ),
        statistic_references,
        statistic_values,
        strict=True,
    ):
        assert actual is expected_reference
        torch.testing.assert_close(actual, expected_value, rtol=0.0, atol=0.0)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_reset_zeros_all_opacity_moments_and_preserves_other_groups() -> None:
    model = _model([0.001, 0.01, 0.2])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    parameters = model.gaussian_parameter_dict()
    states = {
        name: optimizer.state[parameter]
        for name, parameter in parameters.items()
    }
    state_values = {
        name: {
            key: value.detach().clone() if isinstance(value, Tensor) else value
            for key, value in state.items()
        }
        for name, state in states.items()
    }
    opacity_state = states["raw_opacities"]
    opacity_state["max_exp_avg_sq"] = torch.full_like(
        parameters["raw_opacities"], 3.0
    )
    old_opacity_step = opacity_state["step"].detach().clone()

    reset_gaussian_opacity(model, optimizer, maximum_opacity=0.01)

    new_opacity = model.raw_opacities
    new_opacity_state = optimizer.state[new_opacity]
    assert _groups(optimizer)["raw_opacities"]["params"][0] is new_opacity
    assert parameters["raw_opacities"] not in optimizer.state
    for moment in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert new_opacity_state[moment].shape == new_opacity.shape
        assert torch.count_nonzero(new_opacity_state[moment]).item() == 0
    torch.testing.assert_close(
        new_opacity_state["step"], old_opacity_step, rtol=0.0, atol=0.0
    )

    for name in GAUSSIAN_PARAMETER_NAMES:
        if name == "raw_opacities":
            continue
        parameter = parameters[name]
        assert getattr(model, name) is parameter
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert optimizer.state[parameter] is states[name]
        for key, expected in state_values[name].items():
            actual = optimizer.state[parameter][key]
            if isinstance(expected, Tensor):
                assert actual is states[name][key]
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            else:
                assert actual == expected


def test_num_clamped_zero_and_repeated_reset_still_replace_and_zero_state() -> None:
    model = _model([0.001, 0.005])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    first_old_parameter = model.raw_opacities

    first = reset_gaussian_opacity(model, optimizer, maximum_opacity=0.01)
    first_new_parameter = model.raw_opacities
    first_step = optimizer.state[first_new_parameter]["step"].detach().clone()
    optimizer.state[first_new_parameter]["exp_avg"].fill_(2.0)
    optimizer.state[first_new_parameter]["exp_avg_sq"].fill_(3.0)

    second = reset_gaussian_opacity(model, optimizer, maximum_opacity=0.01)

    assert first.num_clamped == 0
    assert second.num_clamped == 0
    assert first_new_parameter is not first_old_parameter
    assert model.raw_opacities is not first_new_parameter
    assert first_new_parameter not in optimizer.state
    assert torch.count_nonzero(
        optimizer.state[model.raw_opacities]["exp_avg"]
    ).item() == 0
    assert torch.count_nonzero(
        optimizer.state[model.raw_opacities]["exp_avg_sq"]
    ).item() == 0
    torch.testing.assert_close(
        optimizer.state[model.raw_opacities]["step"],
        first_step,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "maximum,expected_exception,message",
    [
        (True, TypeError, "real number"),
        (float("nan"), ValueError, "finite"),
        (float("inf"), ValueError, "finite"),
        (0.0, ValueError, "strictly"),
        (-0.1, ValueError, r"\[0, 1\]"),
        (1.0, ValueError, "strictly"),
        (1.1, ValueError, r"\[0, 1\]"),
    ],
)
def test_invalid_maximum_fails_before_mutation_or_rng_consumption(
    maximum: object,
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model([0.2])
    optimizer = _optimizer(model)
    parameters = model.gaussian_parameter_dict()
    torch.manual_seed(31)
    rng_state = torch.get_rng_state().clone()

    with pytest.raises(expected_exception, match=message):
        reset_gaussian_opacity(model, optimizer, maximum_opacity=maximum)

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]


def test_unrepresentable_maximum_reports_logit_saturation_without_mutation() -> None:
    model = _model([0.2], dtype=torch.float32)
    optimizer = _optimizer(model)
    opacity = model.raw_opacities

    with pytest.raises(ValueError, match="logit saturation"):
        reset_gaussian_opacity(model, optimizer, maximum_opacity=1e-100)

    assert model.raw_opacities is opacity
    assert _groups(optimizer)["raw_opacities"]["params"][0] is opacity


@pytest.mark.parametrize("invalid_state", ["shape", "dtype", "device"])
def test_invalid_opacity_adam_state_is_rejected_before_mutation(
    invalid_state: str,
) -> None:
    model = _model([0.1, 0.2])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    opacity = model.raw_opacities
    group_parameter = _groups(optimizer)["raw_opacities"]["params"][0]
    if invalid_state == "shape":
        optimizer.state[opacity]["exp_avg"] = torch.zeros((1, 1))
    elif invalid_state == "dtype":
        optimizer.state[opacity]["exp_avg"] = torch.zeros(
            opacity.shape, dtype=torch.float64
        )
    else:
        optimizer.state[opacity]["exp_avg"] = torch.zeros(
            opacity.shape, device="meta"
        )

    with pytest.raises((TypeError, ValueError), match=invalid_state):
        reset_gaussian_opacity(model, optimizer, maximum_opacity=0.01)

    assert model.raw_opacities is opacity
    assert _groups(optimizer)["raw_opacities"]["params"][0] is group_parameter


def test_named_parameter_transaction_rolls_back_callback_failure() -> None:
    model = _model([0.1, 0.2])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    parameters = model.gaussian_parameter_dict()
    states = {
        name: optimizer.state[parameter]
        for name, parameter in parameters.items()
    }

    def fail_commit() -> None:
        raise RuntimeError("intentional parameter commit failure")

    with pytest.raises(RuntimeError, match="intentional parameter"):
        replace_named_gaussian_parameter(
            model,
            optimizer,
            "raw_opacities",
            model.raw_opacities.detach().clone(),
            commit_callback=fail_commit,
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = parameters[name]
        assert getattr(model, name) is parameter
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert optimizer.state[parameter] is states[name]


def test_reset_zero_gaussian_model_replaces_empty_opacity_parameter() -> None:
    model = _model([])
    optimizer = _optimizer(model)
    old_opacity = model.raw_opacities
    torch.manual_seed(32)
    rng_state = torch.get_rng_state().clone()

    result = reset_gaussian_opacity(model, optimizer, maximum_opacity=0.01)

    assert result.num_gaussians == 0
    assert result.num_clamped == 0
    assert model.num_gaussians == 0
    assert model.raw_opacities is not old_opacity
    assert model.raw_opacities.shape == (0, 1)
    assert model.raw_opacities.grad is None
    assert _groups(optimizer)["raw_opacities"]["params"][0] is model.raw_opacities
    assert len(optimizer.state) == 0
    assert torch.equal(torch.get_rng_state(), rng_state)
