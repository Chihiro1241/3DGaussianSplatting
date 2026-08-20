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
import gaussian_splatting.training.density_control as density_control
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
    run_density_control_event,
)
from gaussian_splatting.training.optimizer import create_optimizer


ROOT = Path(__file__).resolve().parents[2]


def _model(
    scales: list[list[float]],
    *,
    opacities: list[float] | None = None,
    dtype: torch.dtype = torch.float32,
) -> GaussianModel:
    count = len(scales)
    if opacities is None:
        opacities = [0.8] * count
    row = torch.arange(count, dtype=dtype).unsqueeze(1)
    means_world = torch.cat((row, row + 0.25, row + 0.5), dim=1)
    raw_quaternions = torch.zeros((count, 4), dtype=dtype)
    raw_quaternions[:, 0] = 1.0
    if count > 1:
        raw_quaternions[1, 0] = 2.0**-0.5
        raw_quaternions[1, 3] = 2.0**-0.5
    scale_tensor = (
        torch.tensor(scales, dtype=dtype)
        if count
        else torch.empty((0, 3), dtype=dtype)
    )
    raw_scales = torch.log(scale_tensor)
    opacity_tensor = torch.tensor(opacities, dtype=dtype).reshape(count, 1)
    raw_opacities = torch.logit(opacity_tensor)
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
    *,
    radii: list[int] | None = None,
) -> ScreenSpaceDensityStatistics:
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.position_gradient_accumulator.copy_(
        torch.tensor(mean_gradients, dtype=model.means_world.dtype)
    )
    statistics.position_gradient_denominator.fill_(1)
    if radii is None:
        radii = list(range(1, model.num_gaussians + 1))
    statistics.max_screen_radius.copy_(torch.tensor(radii, dtype=torch.int64))
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


def _assert_statistics_reset(
    statistics: ScreenSpaceDensityStatistics,
    expected_count: int,
) -> None:
    assert statistics.num_gaussians == expected_count
    assert statistics.position_gradient_accumulator.shape == (expected_count,)
    assert statistics.position_gradient_denominator.shape == (expected_count,)
    assert statistics.max_screen_radius.shape == (expected_count,)
    assert torch.count_nonzero(statistics.position_gradient_accumulator).item() == 0
    assert torch.count_nonzero(statistics.position_gradient_denominator).item() == 0
    assert torch.count_nonzero(statistics.max_screen_radius).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_event_clones_and_splits_pre_event_gaussians_exactly_once(
    dtype: torch.dtype,
) -> None:
    model = _model(
        [[0.5, 0.5, 0.5], [2.0, 1.0, 1.0], [0.7, 0.8, 0.9]],
        dtype=dtype,
    )
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 2.0, 0.0])
    torch.manual_seed(10)

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
    )

    assert result.num_gaussians_before == 3
    assert result.num_cloned == 1
    assert result.num_split_parents == 1
    assert result.num_children_created == 2
    assert result.num_pruned_total == 0
    assert result.num_gaussians_after == 5
    assert model.num_gaussians == 5
    assert result.num_gaussians_after == (
        result.num_gaussians_before
        + result.num_cloned
        + result.num_split_parents
        - result.num_pruned_total
    )
    assert result.clone_result.num_gaussians_before == 3
    assert result.split_result.num_gaussians_before == 4
    assert result.split_result.num_gaussians_after == 5
    assert result.statistics_reset
    _assert_statistics_reset(statistics, 5)
    assert len(optimizer.state) == 0


def test_clone_only_event_does_not_consume_rng() -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 1.0, 1.0]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 0.0])
    torch.manual_seed(11)
    rng_state = torch.get_rng_state().clone()

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
    )

    assert result.num_cloned == 1
    assert result.num_split_parents == 0
    assert result.num_pruned_total == 0
    assert model.num_gaussians == 3
    assert torch.equal(torch.get_rng_state(), rng_state)
    _assert_statistics_reset(statistics, 3)


def test_split_only_event_creates_children_removes_parent_and_advances_rng() -> None:
    model = _model([[2.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 0.0])
    original_second = model.means_world[1].detach().clone()
    torch.manual_seed(12)
    rng_state = torch.get_rng_state().clone()

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
    )

    assert result.num_cloned == 0
    assert result.num_split_parents == 1
    assert result.num_children_created == 2
    assert result.num_pruned_total == 0
    assert model.num_gaussians == 3
    torch.testing.assert_close(
        model.means_world[0], original_second, rtol=0.0, atol=0.0
    )
    assert not torch.equal(torch.get_rng_state(), rng_state)
    _assert_statistics_reset(statistics, 3)


def test_pruning_applies_to_clone_and_split_children_and_can_end_at_zero() -> None:
    model = _model(
        [[0.5, 0.5, 0.5], [2.0, 1.0, 1.0]],
        opacities=[0.1, 0.1],
    )
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 2.0])
    torch.manual_seed(13)

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.5,
    )

    assert result.num_cloned == 1
    assert result.num_split_parents == 1
    assert result.num_children_created == 2
    assert result.num_low_opacity == 4
    assert result.num_pruned_total == 4
    assert result.num_gaussians_after == 0
    assert model.num_gaussians == 0
    _assert_statistics_reset(statistics, 0)
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        assert parameter.shape[0] == 0
        assert _groups(optimizer)[name]["params"][0] is parameter


def test_pre_event_screen_radius_prunes_original_but_not_new_clone() -> None:
    model = _model([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [0.0, 2.0], radii=[12, 5])
    original_clone_parent = model.means_world[1].detach().clone()

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
        prune_screen_radius_threshold=10.0,
    )

    assert result.num_cloned == 1
    assert result.num_split_parents == 0
    assert result.num_large_screen == 1
    assert result.num_pruned_total == 1
    assert model.num_gaussians == 2
    torch.testing.assert_close(
        model.means_world[0], original_clone_parent, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model.means_world[1], original_clone_parent, rtol=0.0, atol=0.0
    )
    _assert_statistics_reset(statistics, 2)


def test_event_adam_state_preserves_survivors_and_zeros_all_new_rows() -> None:
    model = _model(
        [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [2.0, 1.0, 1.0]]
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
    torch.manual_seed(14)

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
    )

    assert result.num_gaussians_after == 5
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = getattr(model, name)
        state = optimizer.state[parameter]
        torch.testing.assert_close(
            state["exp_avg"][:2],
            old_states[name]["exp_avg"][:2],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["exp_avg_sq"][:2],
            old_states[name]["exp_avg_sq"][:2],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["exp_avg"][2:],
            torch.zeros_like(parameter[2:]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["exp_avg_sq"][2:],
            torch.zeros_like(parameter[2:]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["step"], old_states[name]["step"], rtol=0.0, atol=0.0
        )
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq"].shape == parameter.shape
        assert old_parameters[name] not in optimizer.state
    _assert_statistics_reset(statistics, 5)


def test_noop_event_only_resets_statistics() -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [0.0, 0.0], radii=[2, 3])
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
    torch.manual_seed(15)
    rng_state = torch.get_rng_state().clone()

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
    )

    assert result.num_cloned == 0
    assert result.num_split_parents == 0
    assert result.num_pruned_total == 0
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        assert optimizer.state[parameters[name]] is optimizer_states[name]
    assert statistics.position_gradient_accumulator is statistic_references[0]
    assert statistics.position_gradient_denominator is statistic_references[1]
    assert statistics.max_screen_radius is statistic_references[2]
    _assert_statistics_reset(statistics, 2)


def _event_state(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    statistics: ScreenSpaceDensityStatistics,
) -> tuple[
    dict[str, torch.nn.Parameter],
    dict[str, dict[str, object]],
    tuple[Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
    Tensor,
]:
    parameters = model.gaussian_parameter_dict()
    states = {
        name: optimizer.state[parameter]
        for name, parameter in parameters.items()
        if parameter in optimizer.state
    }
    references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )
    values = tuple(value.clone() for value in references)
    return parameters, states, references, values, torch.get_rng_state().clone()


def _assert_event_state_restored(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    statistics: ScreenSpaceDensityStatistics,
    snapshot: tuple[
        dict[str, torch.nn.Parameter],
        dict[str, dict[str, object]],
        tuple[Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor],
        Tensor,
    ],
) -> None:
    parameters, states, references, values, rng_state = snapshot
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
        if name in states:
            assert optimizer.state[parameters[name]] is states[name]
    assert statistics.position_gradient_accumulator is references[0]
    assert statistics.position_gradient_denominator is references[1]
    assert statistics.max_screen_radius is references[2]
    for actual, expected in zip(references, values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_event_rolls_back_when_processing_fails_after_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 1.0, 1.0]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 0.0])
    torch.manual_seed(16)
    snapshot = _event_state(model, optimizer, statistics)

    def fail_split(*args: object, **kwargs: object) -> object:
        raise RuntimeError("intentional post-clone failure")

    monkeypatch.setattr(density_control, "split_gaussians", fail_split)

    with pytest.raises(RuntimeError, match="post-clone"):
        run_density_control_event(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            densify_world_scale_threshold=1.0,
            prune_opacity_threshold=0.0,
        )

    _assert_event_state_restored(model, optimizer, statistics, snapshot)


def test_event_rolls_back_rng_and_state_when_processing_fails_after_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[2.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 0.0])
    torch.manual_seed(17)
    snapshot = _event_state(model, optimizer, statistics)

    def fail_prune(*args: object, **kwargs: object) -> object:
        raise RuntimeError("intentional post-split failure")

    monkeypatch.setattr(density_control, "prune_gaussians", fail_prune)

    with pytest.raises(RuntimeError, match="post-split"):
        run_density_control_event(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            densify_world_scale_threshold=1.0,
            prune_opacity_threshold=0.0,
        )

    _assert_event_state_restored(model, optimizer, statistics, snapshot)


def test_event_rolls_back_completed_prune_when_statistics_reset_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 1.0, 1.0]], opacities=[0.1, 0.8])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [0.0, 0.0])
    torch.manual_seed(18)
    snapshot = _event_state(model, optimizer, statistics)

    def fail_reset() -> None:
        statistics.position_gradient_accumulator.zero_()
        raise RuntimeError("intentional reset failure")

    monkeypatch.setattr(statistics, "reset", fail_reset)

    with pytest.raises(RuntimeError, match="reset failure"):
        run_density_control_event(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            densify_world_scale_threshold=1.0,
            prune_opacity_threshold=0.5,
        )

    _assert_event_state_restored(model, optimizer, statistics, snapshot)


def _run_seeded_split_event(seed: int) -> dict[str, Tensor]:
    model = _model([[2.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0, 0.0])
    torch.manual_seed(seed)
    run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=1.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=0.0,
    )
    return {
        name: parameter.detach().clone()
        for name, parameter in model.gaussian_parameter_dict().items()
    }


def test_event_is_deterministic_for_same_seed_and_varies_for_different_seed() -> None:
    first = _run_seeded_split_event(19)
    second = _run_seeded_split_event(19)
    different = _run_seeded_split_event(20)

    for name in GAUSSIAN_PARAMETER_NAMES:
        torch.testing.assert_close(first[name], second[name], rtol=0.0, atol=0.0)
    assert not torch.equal(first["means_world"], different["means_world"])


@pytest.mark.parametrize(
    "thresholds,expected_exception,message",
    [
        ({"gradient_threshold": True}, TypeError, "real number"),
        ({"gradient_threshold": float("nan")}, ValueError, "finite"),
        ({"gradient_threshold": -1.0}, ValueError, "non-negative"),
        ({"densify_world_scale_threshold": False}, TypeError, "real number"),
        ({"densify_world_scale_threshold": 0.0}, ValueError, "positive"),
        ({"densify_world_scale_threshold": float("inf")}, ValueError, "finite"),
        ({"prune_opacity_threshold": True}, TypeError, "real number"),
        ({"prune_opacity_threshold": -0.1}, ValueError, r"\[0, 1\]"),
        (
            {"prune_screen_radius_threshold": float("nan")},
            ValueError,
            "finite",
        ),
        ({"prune_world_scale_threshold": -1.0}, ValueError, "non-negative"),
    ],
)
def test_invalid_event_thresholds_fail_before_mutation_or_rng_consumption(
    thresholds: dict[str, object],
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model([[2.0, 1.0, 1.0]])
    optimizer = _optimizer(model)
    statistics = _statistics(model, [2.0])
    parameters = model.gaussian_parameter_dict()
    torch.manual_seed(21)
    rng_state = torch.get_rng_state().clone()
    arguments: dict[str, object] = {
        "gradient_threshold": 1.0,
        "densify_world_scale_threshold": 1.0,
        "prune_opacity_threshold": 0.0,
    }
    arguments.update(thresholds)

    with pytest.raises(expected_exception, match=message):
        run_density_control_event(model, optimizer, statistics, **arguments)

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]


def test_invalid_adam_state_fails_before_event_mutation_or_rng_consumption() -> None:
    model = _model([[0.5, 0.5, 0.5], [2.0, 1.0, 1.0]])
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    statistics = _statistics(model, [2.0, 2.0])
    parameters = model.gaussian_parameter_dict()
    optimizer.state[model.raw_scales]["exp_avg"] = torch.zeros((1, 3))
    torch.manual_seed(24)
    rng_state = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="shape"):
        run_density_control_event(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            densify_world_scale_threshold=1.0,
            prune_opacity_threshold=0.0,
        )

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]


@pytest.mark.parametrize("mismatch", ["count", "dtype", "device"])
def test_incompatible_event_statistics_fail_before_mutation_or_rng_consumption(
    mismatch: str,
) -> None:
    model = _model([[2.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
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
    torch.manual_seed(22)
    rng_state = torch.get_rng_state().clone()

    with pytest.raises((TypeError, ValueError), match=mismatch):
        run_density_control_event(
            model,
            optimizer,
            statistics,
            gradient_threshold=1.0,
            densify_world_scale_threshold=1.0,
            prune_opacity_threshold=0.0,
        )

    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]


def test_zero_gaussian_event_is_safe_and_resets_empty_statistics() -> None:
    model = _model([])
    optimizer = _optimizer(model)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    parameters = model.gaussian_parameter_dict()
    torch.manual_seed(23)
    rng_state = torch.get_rng_state().clone()

    result = run_density_control_event(
        model,
        optimizer,
        statistics,
        gradient_threshold=0.0,
        densify_world_scale_threshold=1.0,
        prune_opacity_threshold=1.0,
    )

    assert result.num_gaussians_before == 0
    assert result.num_gaussians_after == 0
    assert result.num_cloned == 0
    assert result.num_split_parents == 0
    assert result.num_pruned_total == 0
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is parameters[name]
        assert _groups(optimizer)[name]["params"][0] is parameters[name]
    _assert_statistics_reset(statistics, 0)
