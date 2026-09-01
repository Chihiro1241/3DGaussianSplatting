"""Adam parameter groups for a Gaussian model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
from torch import Tensor, nn

from gaussian_splatting.config import Config, TrainingConfig
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GaussianModel,
)


_OPTIMIZER_PARAMETER_ORDER = (
    "means_world",
    "sh_dc",
    "sh_rest",
    "raw_opacities",
    "raw_scales",
    "raw_quaternions",
)
_MOMENT_STATE_NAMES = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
_NO_STATE = object()


def create_optimizer(
    model: GaussianModel,
    config: Config | TrainingConfig,
    *,
    position_lr_scale: float = 1.0,
) -> torch.optim.Adam:
    """Create the six named Adam parameter groups required by the design.

    TeX: eq:gradient_descent_update (implemented by ``Adam.step``).
    """

    training = config.training if isinstance(config, Config) else config
    if training.optimizer != "adam":
        raise ValueError("the initial implementation only supports optimizer='adam'")
    learning_rates = {
        "means_world": training.position_lr_initial * position_lr_scale,
        "sh_dc": training.sh_dc_lr,
        "sh_rest": training.sh_rest_lr,
        "raw_opacities": training.opacity_lr,
        "raw_scales": training.scale_lr,
        "raw_quaternions": training.quaternion_lr,
    }
    if set(learning_rates) != set(GAUSSIAN_PARAMETER_NAMES):  # pragma: no cover
        raise RuntimeError("optimizer learning rates do not cover every Gaussian parameter")
    parameter_groups = [
        {
            "params": [getattr(model, name)],
            "lr": learning_rates[name],
            "name": name,
        }
        for name in _OPTIMIZER_PARAMETER_ORDER
    ]
    return torch.optim.Adam(
        parameter_groups,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_epsilon,
        weight_decay=training.weight_decay,
    )


def _validated_parameter_groups(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
) -> dict[str, dict[str, Any]]:
    if not isinstance(model, GaussianModel):
        raise TypeError("model must be a GaussianModel")
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("optimizer must be torch.optim.Adam")
    current_parameters = model.gaussian_parameter_dict()
    model.validate_gaussian_parameter_tensors(current_parameters)

    groups: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if not isinstance(name, str) or name not in GAUSSIAN_PARAMETER_NAMES:
            raise ValueError("optimizer contains an unknown or unnamed parameter group")
        if name in groups:
            raise ValueError(f"optimizer contains duplicate parameter group {name!r}")
        parameters = group.get("params")
        if not isinstance(parameters, list) or len(parameters) != 1:
            raise ValueError(f"optimizer group {name!r} must contain exactly one parameter")
        if parameters[0] is not current_parameters[name]:
            raise ValueError(
                f"optimizer group {name!r} does not reference the model's current parameter"
            )
        groups[name] = group
    missing = sorted(set(GAUSSIAN_PARAMETER_NAMES) - set(groups))
    if missing:
        raise ValueError(f"optimizer is missing Gaussian parameter groups: {missing}")
    return groups


def _migrated_state(
    state: Mapping[str, Any],
    parameter: nn.Parameter,
    *,
    operation: Literal["append", "keep", "keep_append", "reset"],
    addition: Tensor | None = None,
    keep_mask: Tensor | None = None,
) -> dict[str, Any]:
    migrated = dict(state)
    present_moments = set(state).intersection(_MOMENT_STATE_NAMES)
    required_moments = {"exp_avg", "exp_avg_sq"}
    if present_moments and not required_moments.issubset(present_moments):
        raise ValueError("Adam state must contain both exp_avg and exp_avg_sq")
    for state_name, value in state.items():
        if state_name not in _MOMENT_STATE_NAMES:
            if isinstance(value, Tensor) and value.ndim > 0:
                raise ValueError(
                    f"unsupported non-scalar Adam state {state_name!r} "
                    "for a Gaussian parameter"
                )
            continue
        if not isinstance(value, Tensor):
            raise TypeError(f"Adam state {state_name!r} must be a torch.Tensor")
        if value.ndim == 0:
            raise ValueError(
                f"Adam state {state_name!r} must have the parameter's shape"
            )
        if value.shape != parameter.shape:
            raise ValueError(
                f"Adam state {state_name!r} shape {tuple(value.shape)} does not match "
                f"parameter shape {tuple(parameter.shape)}"
            )
        if value.dtype != parameter.dtype:
            raise TypeError(f"Adam state {state_name!r} dtype does not match its parameter")
        if value.device != parameter.device:
            raise ValueError(f"Adam state {state_name!r} device does not match its parameter")
        if operation == "append":
            if addition is None:  # pragma: no cover - internal contract
                raise RuntimeError("append state migration requires an addition tensor")
            migrated[state_name] = torch.cat(
                (value, torch.zeros_like(addition)), dim=0
            )
        elif operation == "keep":
            if keep_mask is None:  # pragma: no cover - internal contract
                raise RuntimeError("keep state migration requires a keep mask")
            migrated[state_name] = value[keep_mask]
        elif operation == "keep_append":
            if addition is None or keep_mask is None:  # pragma: no cover
                raise RuntimeError(
                    "keep_append state migration requires additions and a keep mask"
                )
            migrated[state_name] = torch.cat(
                (value[keep_mask], torch.zeros_like(addition)), dim=0
            )
        else:
            migrated[state_name] = torch.zeros_like(value)
    return migrated


def _prepare_optimizer_states(
    optimizer: torch.optim.Adam,
    old_parameters: Mapping[str, nn.Parameter],
    *,
    operation: Literal["append", "keep", "keep_append", "reset"],
    additions: Mapping[str, Tensor] | None = None,
    keep_mask: Tensor | None = None,
) -> dict[str, object]:
    states: dict[str, object] = {}
    for name in GAUSSIAN_PARAMETER_NAMES:
        parameter = old_parameters[name]
        if parameter not in optimizer.state:
            states[name] = _NO_STATE
            continue
        states[name] = _migrated_state(
            optimizer.state[parameter],
            parameter,
            operation=operation,
            addition=None if additions is None else additions[name],
            keep_mask=keep_mask,
        )
    return states


def _commit_parameter_transaction(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    groups: Mapping[str, dict[str, Any]],
    new_parameters: Mapping[str, nn.Parameter],
    new_states: Mapping[str, object],
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, nn.Parameter]:
    old_parameters = model.gaussian_parameter_dict()
    old_states = {
        name: optimizer.state.get(old_parameters[name], _NO_STATE)
        for name in GAUSSIAN_PARAMETER_NAMES
    }
    model_replaced = False
    try:
        model.replace_gaussian_parameters(new_parameters)
        model_replaced = True
        for name in GAUSSIAN_PARAMETER_NAMES:
            old_parameter = old_parameters[name]
            new_parameter = new_parameters[name]
            groups[name]["params"][0] = new_parameter
            optimizer.state.pop(old_parameter, None)
            state = new_states[name]
            if state is not _NO_STATE:
                optimizer.state[new_parameter] = state
        if commit_callback is not None:
            commit_callback()
    except BaseException:
        for name in GAUSSIAN_PARAMETER_NAMES:
            optimizer.state.pop(new_parameters[name], None)
            groups[name]["params"][0] = old_parameters[name]
            old_state = old_states[name]
            if old_state is not _NO_STATE:
                optimizer.state[old_parameters[name]] = old_state
        if model_replaced:
            model.replace_gaussian_parameters(old_parameters)
        raise
    return dict(new_parameters)


def append_gaussian_parameters(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    additions: Mapping[str, Tensor],
    *,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, nn.Parameter]:
    """Append rows and atomically commit optional index-aligned side state.

    If ``commit_callback`` raises, the model and optimizer are restored to their
    pre-transaction references and state.
    """

    groups = _validated_parameter_groups(model, optimizer)
    added_count, added_dtype, added_device = model.validate_gaussian_parameter_tensors(
        additions
    )
    old_parameters = model.gaussian_parameter_dict()
    _, model_dtype, model_device = model.validate_gaussian_parameter_tensors(
        old_parameters
    )
    if added_dtype != model_dtype:
        raise TypeError("appended Gaussian tensors must match the model dtype")
    if added_device != model_device:
        raise ValueError("appended Gaussian tensors must be on the model device")

    new_states = _prepare_optimizer_states(
        optimizer,
        old_parameters,
        operation="append",
        additions=additions,
    )
    if added_count == 0:
        return old_parameters
    new_parameters = {
        name: nn.Parameter(
            torch.cat(
                (old_parameters[name].detach(), additions[name].detach()), dim=0
            ),
            requires_grad=True,
        )
        for name in GAUSSIAN_PARAMETER_NAMES
    }
    return _commit_parameter_transaction(
        model,
        optimizer,
        groups,
        new_parameters,
        new_states,
        commit_callback=commit_callback,
    )


def replace_named_gaussian_parameter(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    name: str,
    replacement: Tensor,
    *,
    commit_callback: Callable[[], None] | None = None,
) -> nn.Parameter:
    """Replace one same-shaped Parameter and zero all of its Adam moments.

    Scalar state is preserved. No state is created when the old Parameter has
    none. A callback failure rolls the model, parameter group, and state back
    to their original references.
    """
    groups = _validated_parameter_groups(model, optimizer)
    if not isinstance(name, str) or name not in GAUSSIAN_PARAMETER_NAMES:
        raise ValueError(f"unknown Gaussian parameter name: {name!r}")
    if not isinstance(replacement, Tensor):
        raise TypeError("replacement must be a torch.Tensor")

    old_parameters = model.gaussian_parameter_dict()
    candidate_tensors: dict[str, Tensor] = dict(old_parameters)
    candidate_tensors[name] = replacement
    model.validate_gaussian_parameter_tensors(candidate_tensors)
    old_parameter = old_parameters[name]
    if replacement.shape != old_parameter.shape:
        raise ValueError(
            f"replacement shape {tuple(replacement.shape)} does not match "
            f"parameter shape {tuple(old_parameter.shape)}"
        )

    old_state = optimizer.state.get(old_parameter, _NO_STATE)
    new_state: object = _NO_STATE
    if old_state is not _NO_STATE:
        new_state = _migrated_state(
            old_state,
            old_parameter,
            operation="reset",
        )
    new_parameter = nn.Parameter(replacement.detach().clone(), requires_grad=True)
    new_parameters = dict(old_parameters)
    new_parameters[name] = new_parameter

    model_replaced = False
    try:
        model.replace_gaussian_parameters(new_parameters)
        model_replaced = True
        groups[name]["params"][0] = new_parameter
        optimizer.state.pop(old_parameter, None)
        if new_state is not _NO_STATE:
            optimizer.state[new_parameter] = new_state
        if commit_callback is not None:
            commit_callback()
    except BaseException:
        optimizer.state.pop(new_parameter, None)
        groups[name]["params"][0] = old_parameter
        if old_state is not _NO_STATE:
            optimizer.state[old_parameter] = old_state
        if model_replaced:
            model.replace_gaussian_parameters(old_parameters)
        raise
    return new_parameter


def keep_gaussian_parameters(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    keep_mask: Tensor,
    *,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, nn.Parameter]:
    """Keep the same indices in every Parameter, Adam moment, and callback state.

    If ``commit_callback`` raises, the model and optimizer are restored to their
    pre-transaction references and state. This lets an index-aligned side state
    join the same rollback boundary.
    """

    groups = _validated_parameter_groups(model, optimizer)
    if not isinstance(keep_mask, Tensor):
        raise TypeError("keep_mask must be a torch.Tensor")
    if keep_mask.dtype != torch.bool:
        raise TypeError("keep_mask must have dtype torch.bool")
    if keep_mask.ndim != 1 or keep_mask.shape[0] != model.num_gaussians:
        raise ValueError(
            f"keep_mask must have shape ({model.num_gaussians},), "
            f"got {tuple(keep_mask.shape)}"
        )
    if keep_mask.device != model.means_world.device:
        raise ValueError("keep_mask must be on the model device")

    old_parameters = model.gaussian_parameter_dict()
    new_states = _prepare_optimizer_states(
        optimizer,
        old_parameters,
        operation="keep",
        keep_mask=keep_mask,
    )
    if bool(keep_mask.all().item()):
        return old_parameters
    new_parameters = {
        name: nn.Parameter(
            old_parameters[name].detach()[keep_mask],
            requires_grad=True,
        )
        for name in GAUSSIAN_PARAMETER_NAMES
    }
    return _commit_parameter_transaction(
        model,
        optimizer,
        groups,
        new_parameters,
        new_states,
        commit_callback=commit_callback,
    )


def keep_and_append_gaussian_parameters(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    keep_mask: Tensor,
    additions: Mapping[str, Tensor],
    *,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, nn.Parameter]:
    """Atomically keep old rows, append new rows, and migrate Adam state.

    Existing moments selected by ``keep_mask`` are preserved. Appended rows
    receive zero moments. If ``commit_callback`` raises, every model and
    optimizer reference is rolled back.
    """

    groups = _validated_parameter_groups(model, optimizer)
    if not isinstance(keep_mask, Tensor):
        raise TypeError("keep_mask must be a torch.Tensor")
    if keep_mask.dtype != torch.bool:
        raise TypeError("keep_mask must have dtype torch.bool")
    if keep_mask.shape != (model.num_gaussians,):
        raise ValueError(
            f"keep_mask must have shape ({model.num_gaussians},), "
            f"got {tuple(keep_mask.shape)}"
        )
    if keep_mask.device != model.means_world.device:
        raise ValueError("keep_mask must be on the model device")

    added_count, added_dtype, added_device = model.validate_gaussian_parameter_tensors(
        additions
    )
    old_parameters = model.gaussian_parameter_dict()
    _, model_dtype, model_device = model.validate_gaussian_parameter_tensors(
        old_parameters
    )
    if added_dtype != model_dtype:
        raise TypeError("appended Gaussian tensors must match the model dtype")
    if added_device != model_device:
        raise ValueError("appended Gaussian tensors must be on the model device")

    new_states = _prepare_optimizer_states(
        optimizer,
        old_parameters,
        operation="keep_append",
        additions=additions,
        keep_mask=keep_mask,
    )
    if bool(keep_mask.all().item()) and added_count == 0:
        return old_parameters
    new_parameters = {
        name: nn.Parameter(
            torch.cat(
                (
                    old_parameters[name].detach()[keep_mask],
                    additions[name].detach(),
                ),
                dim=0,
            ),
            requires_grad=True,
        )
        for name in GAUSSIAN_PARAMETER_NAMES
    }
    return _commit_parameter_transaction(
        model,
        optimizer,
        groups,
        new_parameters,
        new_states,
        commit_callback=commit_callback,
    )


__all__ = [
    "append_gaussian_parameters",
    "create_optimizer",
    "keep_and_append_gaussian_parameters",
    "keep_gaussian_parameters",
    "replace_named_gaussian_parameter",
]
