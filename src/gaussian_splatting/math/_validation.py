"""Shared validation helpers for mathematical tensor functions."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor


SUPPORTED_DTYPES = (torch.float32, torch.float64)


def validate_float_tensor(
    tensor: Tensor,
    name: str,
    *,
    shape: Sequence[int | None] | None = None,
    ndim: int | None = None,
    finite: bool = True,
) -> Tensor:
    """Validate a floating tensor's type, dtype, shape, and finite values."""

    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} must have dtype torch.float32 or torch.float64; "
            f"got {tensor.dtype}"
        )
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got shape {tuple(tensor.shape)}")
    if shape is not None:
        if tensor.ndim != len(shape):
            raise ValueError(f"{name} must have shape {format_shape(shape)}; got {tuple(tensor.shape)}")
        for actual, expected in zip(tensor.shape, shape, strict=True):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} must have shape {format_shape(shape)}; got {tuple(tensor.shape)}"
                )
    if finite and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def format_shape(shape: Sequence[int | None]) -> str:
    """Format a validation shape, using ``*`` for a free dimension."""

    return "(" + ", ".join("*" if value is None else str(value) for value in shape) + ")"


def validate_same_dtype_device(*named_tensors: tuple[str, Tensor]) -> None:
    """Require all supplied tensors to share dtype and device."""

    if not named_tensors:
        return
    reference_name, reference = named_tensors[0]
    for name, tensor in named_tensors[1:]:
        if tensor.dtype != reference.dtype:
            raise TypeError(
                f"{name} and {reference_name} must have the same dtype; "
                f"got {tensor.dtype} and {reference.dtype}"
            )
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} and {reference_name} must be on the same device; "
                f"got {tensor.device} and {reference.device}"
            )


def validate_positive_real(value: Real, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite positive Python real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not torch.isfinite(torch.tensor(converted)):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0 if allow_zero else converted <= 0.0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}; got {converted}")
    return converted


def validate_real(value: Real, name: str) -> float:
    """Validate a finite Python real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not torch.isfinite(torch.tensor(converted)):
        raise ValueError(f"{name} must be finite")
    return converted


def tensor_scalar(
    value: Tensor | Real,
    name: str,
    reference: Tensor,
    *,
    positive: bool = False,
) -> Tensor:
    """Coerce a Python real or validate a scalar tensor against ``reference``."""

    if isinstance(value, Tensor):
        validate_float_tensor(value, name, shape=())
        validate_same_dtype_device((name, value), ("reference", reference))
        result = value
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a scalar torch.Tensor or real number")
        result = reference.new_tensor(float(value))
        if not bool(torch.isfinite(result)):
            raise ValueError(f"{name} must be finite")
    if positive and not bool(result > 0):
        raise ValueError(f"{name} must be positive; got {result.item()}")
    return result


def validate_symmetric(matrix: Tensor, name: str) -> None:
    """Reject a matrix that is asymmetric beyond the design tolerance."""

    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"{name} must be square")
    if not torch.allclose(matrix, matrix.transpose(-1, -2), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} must be symmetric within rtol=1e-5 and atol=1e-6")


def validate_jacobian_tensor(
    tensor: Tensor,
    name: str,
    *,
    shape: Sequence[int | None],
) -> Tensor:
    """Validate a CPU float64 tensor used by an analytic Jacobian."""

    validate_float_tensor(tensor, name, shape=shape)
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype torch.float64")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    return tensor


def jacobian_scalar(value: Tensor | Real, name: str) -> Tensor:
    """Return a scalar as a CPU float64 tensor for Jacobian calculations."""

    if isinstance(value, Tensor):
        validate_jacobian_tensor(value, name, shape=())
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a scalar torch.Tensor or real number")
    result = torch.tensor(float(value), dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def validate_component(component: int, size: int, name: str = "component") -> int:
    """Validate an integer component selector."""

    if isinstance(component, bool) or not isinstance(component, int):
        raise TypeError(f"{name} must be an integer")
    if component < 0 or component >= size:
        raise ValueError(f"{name} must be in [0, {size - 1}]; got {component}")
    return component


def validate_active_degree(active_degree: int) -> int:
    """Validate a supported degree for the fixed degree-three SH representation."""

    if isinstance(active_degree, bool) or not isinstance(active_degree, int):
        raise TypeError("active_degree must be an integer")
    if active_degree < 0 or active_degree > 3:
        raise ValueError(f"active_degree must be in [0, 3]; got {active_degree}")
    return active_degree
