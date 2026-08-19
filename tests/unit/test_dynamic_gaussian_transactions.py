from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from gaussian_splatting.config import load_config
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GAUSSIAN_PARAMETER_SPECS,
    GaussianModel,
)
from gaussian_splatting.training.optimizer import (
    append_gaussian_parameters,
    create_optimizer,
    keep_gaussian_parameters,
)


ROOT = Path(__file__).resolve().parents[2]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _parameter_tensors(
    count: int,
    *,
    dtype: torch.dtype,
    device: str,
    offset: float = 0.0,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, (name, shape_tail) in enumerate(GAUSSIAN_PARAMETER_SPECS):
        elements_per_row = math.prod(shape_tail)
        values = torch.arange(
            count * elements_per_row,
            dtype=dtype,
            device=device,
        ).reshape(count, *shape_tail)
        tensors[name] = values + offset + 1000.0 * parameter_index
    return tensors


def _model(
    count: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> GaussianModel:
    return GaussianModel(**_parameter_tensors(count, dtype=dtype, device=device))


def _optimizer(model: GaussianModel) -> torch.optim.Adam:
    return create_optimizer(model, load_config(ROOT / "configs" / "default.yaml"))


def _snapshot(model: GaussianModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.gaussian_parameter_dict().items()
    }


def _groups(optimizer: torch.optim.Adam) -> dict[str, dict[str, object]]:
    return {str(group["name"]): group for group in optimizer.param_groups}


def _initialize_adam_state(
    model: GaussianModel, optimizer: torch.optim.Adam
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def _assert_scalar_state_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    else:
        assert actual == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_append_preserves_existing_rows_and_appends_all_parameters(
    device: str, dtype: torch.dtype
) -> None:
    model = _model(3, dtype=dtype, device=device)
    optimizer = _optimizer(model)
    before = _snapshot(model)
    additions = _parameter_tensors(
        2, dtype=dtype, device=device, offset=10_000.0
    )

    installed = append_gaussian_parameters(model, optimizer, additions)

    assert model.num_gaussians == 5
    assert set(installed) == set(GAUSSIAN_PARAMETER_NAMES)
    for name in GAUSSIAN_PARAMETER_NAMES:
        expected = torch.cat((before[name], additions[name]), dim=0)
        torch.testing.assert_close(getattr(model, name), expected, rtol=0.0, atol=0.0)
        assert installed[name] is getattr(model, name)
        assert installed[name].dtype == dtype
        assert installed[name].device.type == device
        assert _groups(optimizer)[name]["params"][0] is installed[name]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_keep_applies_one_mask_to_every_parameter(
    device: str, dtype: torch.dtype
) -> None:
    model = _model(4, dtype=dtype, device=device)
    optimizer = _optimizer(model)
    before = _snapshot(model)
    keep_mask = torch.tensor([True, False, True, False], device=device)

    installed = keep_gaussian_parameters(model, optimizer, keep_mask)

    assert model.num_gaussians == 2
    for name in GAUSSIAN_PARAMETER_NAMES:
        torch.testing.assert_close(
            getattr(model, name), before[name][keep_mask], rtol=0.0, atol=0.0
        )
        assert installed[name] is getattr(model, name)
        assert _groups(optimizer)[name]["params"][0] is installed[name]


def test_model_replaces_only_a_complete_validated_parameter_set() -> None:
    model = _model(2)
    replacement_tensors = _parameter_tensors(3, dtype=torch.float32, device="cpu")
    replacements = {
        name: nn.Parameter(value.clone())
        for name, value in replacement_tensors.items()
    }

    model.replace_gaussian_parameters(replacements)

    assert model.num_gaussians == 3
    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is replacements[name]


def test_adam_append_preserves_old_moments_and_zero_initializes_new_rows() -> None:
    model = _model(4)
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    old_parameters = model.gaussian_parameter_dict()
    old_states = {
        name: {
            state_name: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else value
            for state_name, value in optimizer.state[parameter].items()
        }
        for name, parameter in old_parameters.items()
    }
    additions = _parameter_tensors(
        2, dtype=torch.float32, device="cpu", offset=20_000.0
    )

    installed = append_gaussian_parameters(model, optimizer, additions)

    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = installed[name]
        state = optimizer.state[parameter]
        assert old_parameters[name] not in optimizer.state
        assert _groups(optimizer)[name]["params"][0] is parameter
        for moment_name in ("exp_avg", "exp_avg_sq"):
            old_moment = old_states[name][moment_name]
            assert isinstance(old_moment, torch.Tensor)
            torch.testing.assert_close(
                state[moment_name][:4], old_moment, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                state[moment_name][4:],
                torch.zeros_like(additions[name]),
                rtol=0.0,
                atol=0.0,
            )
            assert state[moment_name].shape == parameter.shape
        _assert_scalar_state_equal(state["step"], old_states[name]["step"])
    _initialize_adam_state(model, optimizer)


def test_adam_keep_applies_the_parameter_mask_to_every_moment() -> None:
    model = _model(5)
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    old_parameters = model.gaussian_parameter_dict()
    old_moments = {
        name: {
            moment_name: optimizer.state[parameter][moment_name].detach().clone()
            for moment_name in ("exp_avg", "exp_avg_sq")
        }
        for name, parameter in old_parameters.items()
    }
    old_steps = {
        name: optimizer.state[parameter]["step"].detach().clone()
        for name, parameter in old_parameters.items()
    }
    keep_mask = torch.tensor([False, True, True, False, True])

    installed = keep_gaussian_parameters(model, optimizer, keep_mask)

    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = installed[name]
        state = optimizer.state[parameter]
        assert old_parameters[name] not in optimizer.state
        assert _groups(optimizer)[name]["params"][0] is parameter
        for moment_name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state[moment_name],
                old_moments[name][moment_name][keep_mask],
                rtol=0.0,
                atol=0.0,
            )
            assert state[moment_name].shape == parameter.shape
        _assert_scalar_state_equal(state["step"], old_steps[name])
    _initialize_adam_state(model, optimizer)


def test_append_and_keep_before_first_step_do_not_create_adam_state() -> None:
    model = _model(3)
    optimizer = _optimizer(model)
    assert len(optimizer.state) == 0

    append_gaussian_parameters(
        model,
        optimizer,
        _parameter_tensors(2, dtype=torch.float32, device="cpu", offset=30_000.0),
    )
    keep_gaussian_parameters(
        model, optimizer, torch.tensor([True, False, True, True, False])
    )

    assert model.num_gaussians == 3
    assert len(optimizer.state) == 0
    for name, parameter in model.gaussian_parameter_dict().items():
        assert _groups(optimizer)[name]["params"][0] is parameter
    _initialize_adam_state(model, optimizer)
    for parameter in model.gaussian_parameter_dict().values():
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape


def test_multiple_append_keep_append_operations_preserve_index_alignment() -> None:
    model = _model(3)
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    expected = _snapshot(model)
    first_additions = _parameter_tensors(
        2, dtype=torch.float32, device="cpu", offset=40_000.0
    )
    append_gaussian_parameters(model, optimizer, first_additions)
    expected = {
        name: torch.cat((expected[name], first_additions[name]), dim=0)
        for name in GAUSSIAN_PARAMETER_NAMES
    }

    keep_mask = torch.tensor([False, True, True, False, True])
    keep_gaussian_parameters(model, optimizer, keep_mask)
    expected = {name: value[keep_mask] for name, value in expected.items()}

    second_additions = _parameter_tensors(
        1, dtype=torch.float32, device="cpu", offset=50_000.0
    )
    append_gaussian_parameters(model, optimizer, second_additions)
    expected = {
        name: torch.cat((expected[name], second_additions[name]), dim=0)
        for name in GAUSSIAN_PARAMETER_NAMES
    }

    assert model.num_gaussians == 4
    for name in GAUSSIAN_PARAMETER_NAMES:
        torch.testing.assert_close(
            getattr(model, name), expected[name], rtol=0.0, atol=0.0
        )
        assert _groups(optimizer)[name]["params"][0] is getattr(model, name)
        assert optimizer.state[getattr(model, name)]["exp_avg"].shape == expected[name].shape
        assert optimizer.state[getattr(model, name)]["exp_avg_sq"].shape == expected[name].shape
    _initialize_adam_state(model, optimizer)


@pytest.mark.parametrize(
    "mutation, expected_exception, message",
    [
        (
            lambda additions: additions.__setitem__(
                "means_world", torch.zeros((2, 3))
            ),
            ValueError,
            "same N",
        ),
        (
            lambda additions: additions.__setitem__(
                "raw_quaternions", torch.zeros((1, 3))
            ),
            ValueError,
            "shape",
        ),
        (
            lambda additions: additions.__setitem__(
                "raw_opacities", torch.zeros((1, 1), dtype=torch.float64)
            ),
            TypeError,
            "dtype",
        ),
    ],
)
def test_invalid_append_is_rejected_before_any_parameter_changes(
    mutation: object,
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model(3)
    optimizer = _optimizer(model)
    old_parameters = model.gaussian_parameter_dict()
    additions = _parameter_tensors(1, dtype=torch.float32, device="cpu")
    assert callable(mutation)
    mutation(additions)

    with pytest.raises(expected_exception, match=message):
        append_gaussian_parameters(model, optimizer, additions)

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is old_parameters[name]
        assert _groups(optimizer)[name]["params"][0] is old_parameters[name]


def test_append_rejects_missing_and_unknown_parameter_names() -> None:
    model = _model(2)
    optimizer = _optimizer(model)
    additions = _parameter_tensors(1, dtype=torch.float32, device="cpu")
    additions.pop("sh_rest")
    additions["unexpected"] = torch.zeros((1, 1))

    with pytest.raises(ValueError, match="missing parameters.*unknown parameters"):
        append_gaussian_parameters(model, optimizer, additions)


@pytest.mark.parametrize(
    "mask, expected_exception, message",
    [
        (torch.tensor([[True, False, True]]), ValueError, "shape"),
        (torch.tensor([True, False]), ValueError, "shape"),
        (torch.tensor([1, 0, 1]), TypeError, "torch.bool"),
    ],
)
def test_keep_rejects_invalid_mask_without_changing_references(
    mask: torch.Tensor,
    expected_exception: type[Exception],
    message: str,
) -> None:
    model = _model(3)
    optimizer = _optimizer(model)
    old_parameters = model.gaussian_parameter_dict()

    with pytest.raises(expected_exception, match=message):
        keep_gaussian_parameters(model, optimizer, mask)

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is old_parameters[name]
        assert _groups(optimizer)[name]["params"][0] is old_parameters[name]


def test_keep_rejects_a_mask_on_another_device() -> None:
    model = _model(3)
    optimizer = _optimizer(model)
    meta_mask = torch.ones(3, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="model device"):
        keep_gaussian_parameters(model, optimizer, meta_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_append_rejects_additions_on_another_device() -> None:
    model = _model(2, device="cuda")
    optimizer = _optimizer(model)
    additions = _parameter_tensors(1, dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="same device|model device"):
        append_gaussian_parameters(model, optimizer, additions)


def test_invalid_adam_state_is_rejected_without_partial_changes() -> None:
    model = _model(3)
    optimizer = _optimizer(model)
    _initialize_adam_state(model, optimizer)
    old_parameters = model.gaussian_parameter_dict()
    optimizer.state[model.raw_scales]["exp_avg"] = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="shape"):
        append_gaussian_parameters(
            model,
            optimizer,
            _parameter_tensors(1, dtype=torch.float32, device="cpu"),
        )

    for name in GAUSSIAN_PARAMETER_NAMES:
        assert getattr(model, name) is old_parameters[name]
        assert _groups(optimizer)[name]["params"][0] is old_parameters[name]
