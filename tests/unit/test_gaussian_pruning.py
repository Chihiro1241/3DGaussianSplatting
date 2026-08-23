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
    prune_gaussians,
)
from gaussian_splatting.training.optimizer import create_optimizer


ROOT = Path(__file__).resolve().parents[2]


def _model(
    opacities: list[float],
    scales: list[list[float]] | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> GaussianModel:
    count = len(opacities)
    if scales is None:
        scales = [[1.0, 1.0, 1.0] for _ in range(count)]
    row = torch.arange(count, dtype=dtype).unsqueeze(1)
    means_world = torch.cat((row, row + 0.25, row + 0.5), dim=1)
    raw_quaternions = torch.cat(
        (torch.ones((count, 1), dtype=dtype), row.repeat(1, 3)), dim=1
    )
    raw_scales = torch.log(torch.tensor(scales, dtype=dtype))
    raw_opacities = torch.logit(
        torch.tensor(opacities, dtype=dtype).unsqueeze(1)
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


def _statistics(model: GaussianModel) -> ScreenSpaceDensityStatistics:
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    count = model.num_gaussians
    statistics.position_gradient_accumulator.copy_(
        torch.arange(1, count + 1, dtype=model.means_world.dtype)
    )
    statistics.position_gradient_denominator.copy_(
        torch.arange(11, 11 + count, dtype=torch.int64)
    )
    statistics.max_screen_radius.copy_(
        torch.arange(21, 21 + count, dtype=torch.int64)
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


def _parameter_values(model: GaussianModel) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.gaussian_parameter_dict().items()
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_low_opacity_prunes_strictly_below_threshold_and_keeps_boundary(
    dtype: torch.dtype,
) -> None:
    model = _model([0.1, 0.5, 0.8], dtype=dtype)
    optimizer = _optimizer(model)
    statistics = _statistics(model)
    before = _parameter_values(model)
    keep_mask = torch.tensor([False, True, True])

    result = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=0.5,
    )

    assert result.num_gaussians_before == 3
    assert result.num_gaussians_after == 2
    assert result.num_pruned_total == 1
    assert result.num_low_opacity == 1
    assert result.num_large_screen == 0
    assert result.num_large_world == 0
    assert len(optimizer.state) == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        torch.testing.assert_close(
            getattr(model, name), before[name][keep_mask], rtol=0.0, atol=0.0
        )
        assert _groups(optimizer)[name]["params"][0] is getattr(model, name)


def test_screen_radius_pruning_and_none_disables_the_criterion() -> None:
    model = _model([0.8, 0.8, 0.8])
    optimizer = _optimizer(model)
    statistics = _statistics(model)
    statistics.max_screen_radius.copy_(torch.tensor([9, 11, 10]))
    original_parameters = model.gaussian_parameter_dict()

    disabled = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=0.0,
        screen_radius_threshold=None,
    )

    assert disabled.num_pruned_total == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is original_parameters[name]

    enabled = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=0.0,
        screen_radius_threshold=10.0,
    )

    assert enabled.num_pruned_total == 1
    assert enabled.num_large_screen == 1
    torch.testing.assert_close(
        statistics.max_screen_radius, torch.tensor([9, 10])
    )


def test_world_scale_pruning_uses_parameterized_scale_not_raw_scale() -> None:
    model = _model(
        [0.8, 0.8, 0.8],
        scales=[[0.5, 0.9, 0.7], [1.0, 2.0, 1.0], [1.4, 1.2, 1.0]],
    )
    optimizer = _optimizer(model)
    statistics = _statistics(model)
    assert model.raw_scales[1].max().item() < 1.5

    result = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=0.0,
        world_scale_threshold=1.5,
    )

    assert result.num_pruned_total == 1
    assert result.num_large_world == 1
    torch.testing.assert_close(
        torch.exp(model.raw_scales).amax(dim=-1), torch.tensor([0.9, 1.4])
    )


def test_combined_union_preserves_parameters_adam_moments_and_statistics() -> None:
    model = _model(
        [0.1, 0.8, 0.1, 0.8],
        scales=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [3.0, 1.0, 1.0],
        ],
    )
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model)
    statistics.max_screen_radius.copy_(torch.tensor([12, 2, 12, 2]))
    old_parameters = model.gaussian_parameter_dict()
    old_values = _parameter_values(model)
    old_moments = {
        name: {
            moment: optimizer.state[parameter][moment].detach().clone()
            for moment in ("exp_avg", "exp_avg_sq")
        }
        for name, parameter in old_parameters.items()
    }
    old_accumulator = statistics.position_gradient_accumulator.clone()
    old_denominator = statistics.position_gradient_denominator.clone()
    old_radius = statistics.max_screen_radius.clone()
    keep_mask = torch.tensor([False, True, False, False])

    result = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=0.5,
        screen_radius_threshold=10,
        world_scale_threshold=1.5,
    )

    assert result.num_gaussians_before == 4
    assert result.num_gaussians_after == 1
    assert result.num_pruned_total == 3
    assert result.num_low_opacity == 2
    assert result.num_large_screen == 2
    assert result.num_large_world == 2
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        torch.testing.assert_close(
            parameter, old_values[name][keep_mask], rtol=0.0, atol=0.0
        )
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert old_parameters[name] not in optimizer.state
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][moment],
                old_moments[name][moment][keep_mask],
                rtol=0.0,
                atol=0.0,
            )
    torch.testing.assert_close(
        statistics.position_gradient_accumulator,
        old_accumulator[keep_mask],
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator,
        old_denominator[keep_mask],
    )
    torch.testing.assert_close(
        statistics.max_screen_radius,
        old_radius[keep_mask],
    )


def test_noop_preserves_parameter_optimizer_and_statistics_identity() -> None:
    model = _model([0.8, 0.9])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model)
    parameter_references = model.gaussian_parameter_dict()
    state_references = {
        name: optimizer.state[parameter]
        for name, parameter in parameter_references.items()
    }
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )

    result = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=0.0,
        screen_radius_threshold=None,
        world_scale_threshold=None,
    )

    assert result.num_pruned_total == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameter_references[name]
        assert _groups(optimizer)[name]["params"][0] is parameter_references[name]
        assert optimizer.state[parameter_references[name]] is state_references[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


@pytest.mark.parametrize(
    "thresholds,expected_exception,message",
    [
        ({"opacity_threshold": True}, TypeError, "real number"),
        ({"opacity_threshold": float("nan")}, ValueError, "finite"),
        ({"opacity_threshold": float("inf")}, ValueError, "finite"),
        ({"opacity_threshold": -0.1}, ValueError, r"\[0, 1\]"),
        ({"opacity_threshold": 1.1}, ValueError, r"\[0, 1\]"),
        (
            {"opacity_threshold": 0.0, "screen_radius_threshold": False},
            TypeError,
            "real number",
        ),
        (
            {"opacity_threshold": 0.0, "screen_radius_threshold": -1.0},
            ValueError,
            "non-negative",
        ),
        (
            {"opacity_threshold": 0.0, "screen_radius_threshold": float("nan")},
            ValueError,
            "finite",
        ),
        (
            {"opacity_threshold": 0.0, "world_scale_threshold": True},
            TypeError,
            "real number",
        ),
        (
            {"opacity_threshold": 0.0, "world_scale_threshold": -1.0},
            ValueError,
            "non-negative",
        ),
        (
            {"opacity_threshold": 0.0, "world_scale_threshold": float("inf")},
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_thresholds_are_rejected_before_mutation(
    thresholds: dict[str, object],
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model([0.2, 0.8])
    optimizer = _optimizer(model)
    statistics = _statistics(model)
    parameters = model.gaussian_parameter_dict()
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )

    with pytest.raises(expected_exception, match=message):
        prune_gaussians(model, optimizer, statistics, **thresholds)

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


@pytest.mark.parametrize("mismatch", ["count", "dtype", "device"])
def test_incompatible_statistics_fail_before_model_or_optimizer_mutation(
    mismatch: str,
) -> None:
    model = _model([0.2, 0.8])
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
        prune_gaussians(
            model,
            optimizer,
            statistics,
            opacity_threshold=0.5,
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]


def test_invalid_adam_state_leaves_every_component_unchanged() -> None:
    model = _model([0.1, 0.8, 0.9])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model)
    parameters = model.gaussian_parameter_dict()
    statistic_references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )
    optimizer.state[model.raw_scales]["exp_avg"] = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="shape"):
        prune_gaussians(
            model,
            optimizer,
            statistics,
            opacity_threshold=0.5,
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


def test_statistics_commit_failure_rolls_back_model_and_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([0.1, 0.8, 0.9])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model)
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
        prune_gaussians(
            model,
            optimizer,
            statistics,
            opacity_threshold=0.5,
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        assert optimizer.state[parameters[name]] is optimizer_states[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


def test_all_gaussians_can_be_pruned_to_zero_with_adam_state() -> None:
    model = _model([0.1, 0.2, 0.3])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model)

    result = prune_gaussians(
        model,
        optimizer,
        statistics,
        opacity_threshold=1.0,
    )

    assert result.num_gaussians_before == 3
    assert result.num_gaussians_after == 0
    assert result.num_pruned_total == 3
    assert model.num_gaussians == 0
    assert statistics.num_gaussians == 0
    assert statistics.position_gradient_accumulator.shape == (0,)
    assert statistics.position_gradient_denominator.shape == (0,)
    assert statistics.max_screen_radius.shape == (0,)
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        assert parameter.shape[0] == 0
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape

    optimizer.zero_grad(set_to_none=True)
    zero_loss = sum(parameter.sum() for parameter in model.parameters())
    zero_loss.backward()
    optimizer.step()
    transformed = model.transformed_parameters()
    assert transformed.means_world.shape == (0, 3)
