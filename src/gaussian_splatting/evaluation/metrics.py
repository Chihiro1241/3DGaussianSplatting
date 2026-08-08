"""Image quality metrics for rendered views."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_pair(rendered: Tensor, target: Tensor) -> None:
    if rendered.shape != target.shape:
        raise ValueError("rendered and target must have the same shape")
    if rendered.ndim != 3 or rendered.shape[0] != 3:
        raise ValueError("images must have shape (3,H,W)")
    if rendered.dtype not in (torch.float32, torch.float64):
        raise TypeError("images must use torch.float32 or torch.float64")
    if rendered.dtype != target.dtype:
        raise TypeError("rendered and target must have the same dtype")
    if rendered.device != target.device:
        raise ValueError("rendered and target must be on the same device")


def mse(rendered: Tensor, target: Tensor) -> Tensor:
    """Return mean squared error over every pixel and RGB component.

    TeX: eq:mse
    """

    _validate_pair(rendered, target)
    return torch.mean((rendered - target).square())


def psnr(rendered: Tensor, target: Tensor, image_max: float = 1.0) -> Tensor:
    """Return image PSNR, yielding ``+inf`` for an exact match.

    TeX: eq:psnr
    """

    if image_max <= 0.0:
        raise ValueError("image_max must be positive")
    error = mse(rendered, target)
    maximum = error.new_tensor(image_max)
    return torch.where(
        error == 0,
        error.new_tensor(torch.inf),
        10.0 * torch.log10(maximum.square() / error),
    )


def mean_psnr(values: Tensor | list[Tensor]) -> Tensor:
    """Return the arithmetic mean of per-view PSNR values.

    TeX: eq:mean_psnr
    """

    if isinstance(values, list):
        if not values:
            raise ValueError("at least one PSNR value is required")
        reference = values[0]
        if not isinstance(reference, Tensor) or reference.numel() != 1:
            raise ValueError("each PSNR list item must be a scalar tensor")
        if reference.dtype not in (torch.float32, torch.float64):
            raise TypeError("PSNR values must use torch.float32 or torch.float64")
        for value in values[1:]:
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError("each PSNR list item must be a scalar tensor")
            if value.dtype != reference.dtype:
                raise TypeError("all PSNR values must have the same dtype")
            if value.device != reference.device:
                raise ValueError("all PSNR values must be on the same device")
        values = torch.stack(values)
    if values.numel() == 0:
        raise ValueError("at least one PSNR value is required")
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError("PSNR values must use torch.float32 or torch.float64")
    return values.mean()
