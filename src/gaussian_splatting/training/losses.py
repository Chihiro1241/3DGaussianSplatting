"""Image reconstruction losses used by the reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_images(rendered: Tensor, target: Tensor) -> None:
    if rendered.shape != target.shape:
        raise ValueError(
            f"rendered and target must have the same shape; got "
            f"{tuple(rendered.shape)} and {tuple(target.shape)}"
        )
    if rendered.ndim != 3 or rendered.shape[0] != 3:
        raise ValueError(f"images must have shape (3,H,W); got {tuple(rendered.shape)}")
    if rendered.dtype not in (torch.float32, torch.float64):
        raise TypeError("images must use torch.float32 or torch.float64")
    if target.dtype != rendered.dtype:
        raise TypeError("rendered and target must have the same dtype")
    if target.device != rendered.device:
        raise ValueError("rendered and target must be on the same device")
    if not torch.isfinite(rendered).all() or not torch.isfinite(target).all():
        raise ValueError("images must not contain NaN or Inf")


def l1_norm(rgb: Tensor) -> Tensor:
    """Compute the L1 norm of RGB vectors along their last axis.

    TeX: eq:l1_norm
    """

    if rgb.shape[-1] != 3:
        raise ValueError(f"the last dimension must be RGB (3); got {tuple(rgb.shape)}")
    if rgb.dtype not in (torch.float32, torch.float64):
        raise TypeError("rgb must use torch.float32 or torch.float64")
    return rgb.abs().sum(dim=-1)


def l1_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """Return mean absolute RGB error for two ``(3,H,W)`` images.

    TeX: eq:l1_loss
    """

    _validate_images(rendered, target)
    return torch.mean(torch.abs(rendered - target))


def _gaussian_window(
    window_size: int,
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("SSIM sigma must be positive")
    coordinates = torch.arange(window_size, dtype=dtype, device=device)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(3, 1, window_size, window_size).contiguous()


def ssim_statistics(
    rendered: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute local means, variances, and covariance for SSIM.

    Each returned tensor has shape ``(3,H,W)``.  A normalized Gaussian
    window is applied independently to each channel with zero padding.

    TeX: eq:ssim_statistics
    """

    _validate_images(rendered, target)
    window = _gaussian_window(
        window_size, sigma, dtype=rendered.dtype, device=rendered.device
    )
    padding = window_size // 2
    x = rendered.unsqueeze(0)
    y = target.unsqueeze(0)

    mu_x = F.conv2d(x, window, padding=padding, groups=3)
    mu_y = F.conv2d(y, window, padding=padding, groups=3)
    second_x = F.conv2d(x * x, window, padding=padding, groups=3)
    second_y = F.conv2d(y * y, window, padding=padding, groups=3)
    cross_xy = F.conv2d(x * y, window, padding=padding, groups=3)

    variance_x = second_x - mu_x.square()
    variance_y = second_y - mu_y.square()
    covariance_xy = cross_xy - mu_x * mu_y
    return (
        mu_x.squeeze(0),
        mu_y.squeeze(0),
        variance_x.squeeze(0),
        variance_y.squeeze(0),
        covariance_xy.squeeze(0),
    )


def ssim_constants(
    k1: float = 0.01,
    k2: float = 0.03,
    dynamic_range: float = 1.0,
) -> tuple[float, float]:
    """Return the stabilizing constants ``C1`` and ``C2``.

    TeX: eq:ssim_constants
    """

    if k1 <= 0.0 or k2 <= 0.0 or dynamic_range <= 0.0:
        raise ValueError("k1, k2, and dynamic_range must be positive")
    return (k1 * dynamic_range) ** 2, (k2 * dynamic_range) ** 2


def ssim(
    rendered: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Compute mean SSIM over all pixels and RGB channels.

    TeX: eq:ssim
    """

    mu_x, mu_y, variance_x, variance_y, covariance_xy = ssim_statistics(
        rendered, target, window_size=window_size, sigma=sigma
    )
    c1_value, c2_value = ssim_constants(k1, k2, dynamic_range=1.0)
    c1 = rendered.new_tensor(c1_value)
    c2 = rendered.new_tensor(c2_value)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        variance_x + variance_y + c2
    )
    return torch.mean(numerator / denominator)


def dssim_loss(
    rendered: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Compute D-SSIM as ``1 - SSIM`` (without division by two).

    TeX: eq:dssim_loss
    """

    return 1.0 - ssim(
        rendered,
        target,
        window_size=window_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
    )


@dataclass(frozen=True)
class LossResult:
    """Individual terms and the combined reconstruction loss."""

    total: Tensor
    l1: Tensor
    dssim: Tensor
    ssim: Tensor


def total_loss(
    rendered: Tensor,
    target: Tensor,
    lambda_dssim: float = 0.2,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> LossResult:
    """Combine L1 and D-SSIM reconstruction losses.

    TeX: eq:total_loss
    """

    if not 0.0 <= lambda_dssim <= 1.0:
        raise ValueError("lambda_dssim must be in [0,1]")
    l1_value = l1_loss(rendered, target)
    ssim_value = ssim(
        rendered,
        target,
        window_size=window_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
    )
    dssim_value = 1.0 - ssim_value
    total_value = (1.0 - lambda_dssim) * l1_value + lambda_dssim * dssim_value
    if not torch.isfinite(total_value):
        raise FloatingPointError("total loss contains NaN or Inf")
    return LossResult(total=total_value, l1=l1_value, dssim=dssim_value, ssim=ssim_value)

