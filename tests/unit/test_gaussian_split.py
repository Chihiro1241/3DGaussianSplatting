from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from gaussian_splatting.config import load_config
from gaussian_splatting.math.covariance import quaternion_rotation_matrix
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GaussianModel,
)
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
    clone_gaussians,
    split_gaussians,
)
from gaussian_splatting.training.optimizer import create_optimizer


ROOT = Path(__file__).resolve().parents[2]


def _model(
    scales: list[list[float]],
    *,
    quaternions: list[list[float]] | None = None,
    dtype: torch.dtype = torch.float32,
) -> GaussianModel:
    count = len(scales)
    row = torch.arange(count, dtype=dtype).unsqueeze(1)
    means_world = torch.cat((row, row + 0.25, row + 0.5), dim=1)
    if quaternions is None:
        raw_quaternions = torch.zeros((count, 4), dtype=dtype)
        raw_quaternions[:, 0] = 1.0
    else:
        raw_quaternions = torch.tensor(quaternions, dtype=dtype).reshape(count, 4)
    scale_tensor = (
        torch.tensor(scales, dtype=dtype)
        if count
        else torch.empty((0, 3), dtype=dtype)
    )
    raw_scales = torch.log(scale_tensor)
    raw_opacities = torch.linspace(-1.0, 1.0, count, dtype=dtype).unsqueeze(1)
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
def test_split_selection_sampling_scale_attributes_statistics_and_ordering(
    dtype: torch.dtype,
) -> None:
    half_sqrt_two = 2.0**-0.5
    model = _model(
        [
            [1.0, 0.8, 0.7],
            [2.0, 1.0, 0.5],
            [3.0, 1.0, 1.0],
            [1.5, 2.5, 1.0],
        ],
        quaternions=[
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0 * half_sqrt_two, 0.0, 0.0, 2.0 * half_sqrt_two],
        ],
        dtype=dtype,
    )
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 1.0, 0.5, 3.0])
    old_values = _values(model)
    old_statistics = (
        statistics.position_gradient_accumulator.clone(),
        statistics.position_gradient_denominator.clone(),
        statistics.max_screen_radius.clone(),
    )
    parent_indices = torch.tensor([1, 3])
    keep_indices = torch.tensor([0, 2])
    transformed = model.transformed_parameters()
    parent_scales = transformed.scales.detach()[parent_indices]
    rotations = quaternion_rotation_matrix(
        transformed.quaternions.detach()[parent_indices]
    )
    repeated_scales = parent_scales.repeat(2, 1)
    repeated_rotations = rotations.repeat(2, 1, 1)
    repeated_means = old_values["means_world"][parent_indices].repeat(2, 1)

    torch.manual_seed(12345)
    state_before = torch.get_rng_state()
    epsilon = torch.randn((4, 3), dtype=dtype)
    state_after_expected_sampling = torch.get_rng_state()
    expected_local_offsets = repeated_scales * epsilon
    expected_world_offsets = torch.bmm(
        repeated_rotations, expected_local_offsets.unsqueeze(-1)
    ).squeeze(-1)
    expected_child_means = repeated_means + expected_world_offsets
    torch.set_rng_state(state_before)

    result = split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert result.num_gaussians_before == 4
    assert result.num_split_parents == 2
    assert result.num_children_created == 4
    assert result.num_gaussians_after == 6
    assert torch.equal(torch.get_rng_state(), state_after_expected_sampling)
    assert not torch.equal(state_before, state_after_expected_sampling)
    assert len(optimizer.state) == 0
    for name in GAUSSIAN_PARAMETER_NAMES:
        torch.testing.assert_close(
            getattr(model, name)[:2],
            old_values[name][keep_indices],
            rtol=0.0,
            atol=0.0,
        )
        assert _groups(optimizer)[name]["params"][0] is getattr(model, name)
        assert getattr(model, name).is_leaf
    torch.testing.assert_close(
        model.means_world[2:], expected_child_means, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        torch.exp(model.raw_scales[2:]), repeated_scales / 1.6
    )
    for name in ("raw_quaternions", "raw_opacities", "sh_dc", "sh_rest"):
        parent_values = old_values[name][parent_indices]
        repeats = (2,) + (1,) * (parent_values.ndim - 1)
        torch.testing.assert_close(
            getattr(model, name)[2:],
            parent_values.repeat(repeats),
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        statistics.position_gradient_accumulator[:2], old_statistics[0][keep_indices]
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator[:2], old_statistics[1][keep_indices]
    )
    torch.testing.assert_close(
        statistics.max_screen_radius[:2], old_statistics[2][keep_indices]
    )
    assert torch.count_nonzero(statistics.position_gradient_accumulator[2:]).item() == 0
    assert torch.count_nonzero(statistics.position_gradient_denominator[2:]).item() == 0
    assert torch.count_nonzero(statistics.max_screen_radius[2:]).item() == 0


def test_split_threshold_zero_requires_observation_not_positive_gradient() -> None:
    model = _model([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    optimizer = _optimizer(model)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.position_gradient_denominator[1] = 1
    unseen_values = {
        name: getattr(model, name)[0].detach().clone()
        for name in GAUSSIAN_PARAMETER_NAMES
    }
    torch.manual_seed(54321)

    result = split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=0.0,
        world_scale_threshold=1.0,
    )

    assert result.num_split_parents == 1
    assert result.num_children_created == 2
    assert result.num_gaussians_after == 3
    for name in GAUSSIAN_PARAMETER_NAMES:
        torch.testing.assert_close(
            getattr(model, name)[0], unseen_values[name], rtol=0.0, atol=0.0
        )


def test_split_adam_state_keeps_unsplit_moments_and_zeros_children() -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 1.0, 1.0], [0.7, 0.8, 0.9]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [0.0, 2.0, 0.0])
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
    keep_mask = torch.tensor([True, False, True])
    torch.manual_seed(1)

    result = split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert result.num_split_parents == 1
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        state = optimizer.state[parameter]
        assert old_parameters[name] not in optimizer.state
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state[moment][:2],
                old_states[name][moment][keep_mask],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                state[moment][2:],
                torch.zeros_like(parameter[2:]),
                rtol=0.0,
                atol=0.0,
            )
            assert state[moment].shape == parameter.shape
        torch.testing.assert_close(
            state["step"], old_states[name]["step"], rtol=0.0, atol=0.0
        )


def _split_child_means(seed: int) -> Tensor:
    model = _model([[2.0, 1.0, 0.5]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0])
    torch.manual_seed(seed)
    split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )
    return model.means_world.detach().clone()


def test_split_is_deterministic_for_same_seed_and_changes_for_different_seed() -> None:
    first = _split_child_means(77)
    second = _split_child_means(77)
    different = _split_child_means(78)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different)


def test_split_noop_preserves_every_identity_and_rng_state() -> None:
    model = _model([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 0.5])
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
    torch.manual_seed(91)
    rng_state = torch.get_rng_state().clone()

    result = split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert result.num_split_parents == 0
    assert result.num_children_created == 0
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        assert optimizer.state[parameters[name]] is optimizer_states[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


def test_statistics_commit_failure_rolls_back_model_optimizer_statistics_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[2.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
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
    torch.manual_seed(123)
    rng_state = torch.get_rng_state().clone()

    def fail_commit(state: object) -> None:
        raise RuntimeError("intentional statistics commit failure")

    monkeypatch.setattr(statistics, "_commit_state", fail_commit)

    with pytest.raises(RuntimeError, match="intentional statistics"):
        split_gaussians(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            world_scale_threshold=1.0,
        )

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        assert optimizer.state[parameters[name]] is optimizer_states[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]


def test_invalid_adam_state_is_rejected_before_sampling() -> None:
    model = _model([[2.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 0.0])
    parameters = model.gaussian_parameter_dict()
    optimizer.state[model.raw_scales]["exp_avg"] = torch.zeros((1, 3))
    torch.manual_seed(124)
    rng_state = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="shape"):
        split_gaussians(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            world_scale_threshold=1.0,
        )

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]


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
def test_invalid_thresholds_do_not_mutate_or_consume_rng(
    thresholds: dict[str, object],
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model([[2.0, 1.0, 1.0]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0])
    parameters = model.gaussian_parameter_dict()
    torch.manual_seed(4)
    rng_state = torch.get_rng_state().clone()

    with pytest.raises(expected_exception, match=message):
        split_gaussians(model, optimizer, statistics, **thresholds)

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]


@pytest.mark.parametrize("mismatch", ["count", "dtype", "device"])
def test_incompatible_statistics_fail_before_mutation_or_rng_consumption(
    mismatch: str,
) -> None:
    model = _model([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
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
    torch.manual_seed(5)
    rng_state = torch.get_rng_state().clone()

    with pytest.raises((TypeError, ValueError), match=mismatch):
        split_gaussians(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            world_scale_threshold=1.0,
        )

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]


def test_clone_then_split_keeps_every_indexed_component_aligned() -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 1.0, 1.0]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 0.0])

    cloned = clone_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )
    assert cloned.num_cloned == 1
    statistics.position_gradient_accumulator.zero_()
    statistics.position_gradient_denominator.fill_(1)
    statistics.position_gradient_accumulator[1] = 2.0
    torch.manual_seed(6)

    split = split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        world_scale_threshold=1.0,
    )

    assert split.num_split_parents == 1
    assert model.num_gaussians == 4
    assert statistics.num_gaussians == 4
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        assert parameter.shape[0] == 4
        assert _groups(optimizer)[name]["params"][0] is parameter
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape


def test_split_on_zero_gaussian_model_is_noop_without_rng_consumption() -> None:
    model = _model([])
    optimizer = _optimizer(model)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    parameters = model.gaussian_parameter_dict()
    torch.manual_seed(7)
    rng_state = torch.get_rng_state().clone()

    result = split_gaussians(
        model,
        optimizer,
        statistics,
        gradient_threshold=0.0,
        world_scale_threshold=1.0,
    )

    assert result.num_gaussians_before == 0
    assert result.num_gaussians_after == 0
    assert result.num_split_parents == 0
    assert result.num_children_created == 0
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
