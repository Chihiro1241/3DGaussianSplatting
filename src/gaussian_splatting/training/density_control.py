"""Screen-space statistics used by adaptive density control."""

from __future__ import annotations

import torch
from torch import Tensor

from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import RenderResult


class ScreenSpaceDensityStatistics:
    """Accumulate screen-space observations for each original Gaussian index."""

    def __init__(
        self,
        num_gaussians: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if num_gaussians < 0:
            raise ValueError("num_gaussians must be non-negative")
        if not dtype.is_floating_point:
            raise TypeError("density statistics dtype must be floating point")

        device = torch.device(device)
        self.position_gradient_accumulator = torch.zeros(
            num_gaussians, dtype=dtype, device=device
        )
        self.position_gradient_denominator = torch.zeros(
            num_gaussians, dtype=torch.int64, device=device
        )
        self.max_screen_radius = torch.zeros(
            num_gaussians, dtype=torch.int64, device=device
        )

    @classmethod
    def for_model(cls, model: GaussianModel) -> ScreenSpaceDensityStatistics:
        """Create zero-initialized statistics matching a Gaussian model."""
        return cls(
            model.num_gaussians,
            dtype=model.means_world.dtype,
            device=model.means_world.device,
        )

    @property
    def num_gaussians(self) -> int:
        """Return the number of Gaussian-indexed rows."""
        return self.position_gradient_accumulator.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype used for gradient statistics."""
        return self.position_gradient_accumulator.dtype

    @property
    def device(self) -> torch.device:
        """Return the device holding all statistics."""
        return self.position_gradient_accumulator.device

    def validate_compatible(self, model: GaussianModel) -> None:
        """Raise if the statistics do not match the model's indexed parameters."""
        if self.num_gaussians != model.num_gaussians:
            raise ValueError(
                "density statistics Gaussian count does not match model: "
                f"{self.num_gaussians} != {model.num_gaussians}"
            )
        if self.dtype != model.means_world.dtype:
            raise TypeError(
                "density statistics dtype does not match model: "
                f"{self.dtype} != {model.means_world.dtype}"
            )
        if self.device != model.means_world.device:
            raise ValueError(
                "density statistics device does not match model: "
                f"{self.device} != {model.means_world.device}"
            )

    def accumulate(self, render: RenderResult) -> None:
        """Accumulate gradients and radii from a completed backward pass."""
        projected = render.projected
        indices = projected.original_indices
        means_screen = projected.means_screen
        radii = projected.radii
        visible_mask = render.visible_mask

        self._validate_render_shapes(
            visible_mask=visible_mask,
            indices=indices,
            means_screen=means_screen,
            radii=radii,
        )
        if indices.numel() == 0:
            return

        gradient = means_screen.grad
        if gradient is None:
            raise RuntimeError(
                "screen-space gradient is unavailable; render with "
                "retain_screen_grad=True and call loss.backward() before "
                "accumulating density statistics"
            )
        if gradient.shape != means_screen.shape:
            raise ValueError(
                "screen-space gradient shape does not match means_screen: "
                f"{tuple(gradient.shape)} != {tuple(means_screen.shape)}"
            )
        if gradient.dtype != self.dtype:
            raise TypeError(
                "screen-space gradient dtype does not match statistics: "
                f"{gradient.dtype} != {self.dtype}"
            )
        if gradient.device != self.device:
            raise ValueError(
                "screen-space gradient device does not match statistics: "
                f"{gradient.device} != {self.device}"
            )

        with torch.no_grad():
            gradient_norm = torch.linalg.vector_norm(gradient.detach(), dim=-1)
            if not torch.isfinite(gradient_norm).all():
                raise ValueError("screen-space gradient contains non-finite values")

            self.position_gradient_accumulator.index_add_(
                0, indices, gradient_norm
            )
            self.position_gradient_denominator.index_add_(
                0, indices, torch.ones_like(indices)
            )
            self.max_screen_radius[indices] = torch.maximum(
                self.max_screen_radius[indices], radii.detach()
            )

    def reset(self) -> None:
        """Clear the current densification window without reallocating state."""
        with torch.no_grad():
            self.position_gradient_accumulator.zero_()
            self.position_gradient_denominator.zero_()
            self.max_screen_radius.zero_()

    def mean_position_gradient(self) -> Tensor:
        """Return per-Gaussian means, using zero for unobserved rows."""
        mean = torch.zeros_like(self.position_gradient_accumulator)
        observed = self.position_gradient_denominator > 0
        mean[observed] = (
            self.position_gradient_accumulator[observed]
            / self.position_gradient_denominator[observed].to(self.dtype)
        )
        return mean

    def append_zeros(self, count: int) -> None:
        """Append zero-initialized rows for newly created Gaussians."""
        if count < 0:
            raise ValueError("append count must be non-negative")
        if count == 0:
            return

        accumulator = torch.cat(
            (
                self.position_gradient_accumulator,
                self.position_gradient_accumulator.new_zeros(count),
            )
        )
        denominator = torch.cat(
            (
                self.position_gradient_denominator,
                self.position_gradient_denominator.new_zeros(count),
            )
        )
        radius = torch.cat(
            (self.max_screen_radius, self.max_screen_radius.new_zeros(count))
        )
        self.position_gradient_accumulator = accumulator
        self.position_gradient_denominator = denominator
        self.max_screen_radius = radius

    def keep(self, keep_mask: Tensor) -> None:
        """Keep the same selected Gaussian indices in every statistic."""
        if keep_mask.dtype != torch.bool:
            raise TypeError("keep_mask must have dtype torch.bool")
        if keep_mask.device != self.device:
            raise ValueError(
                "keep_mask device does not match statistics: "
                f"{keep_mask.device} != {self.device}"
            )
        if keep_mask.shape != (self.num_gaussians,):
            raise ValueError(
                "keep_mask must have shape "
                f"({self.num_gaussians},), got {tuple(keep_mask.shape)}"
            )

        accumulator = self.position_gradient_accumulator[keep_mask]
        denominator = self.position_gradient_denominator[keep_mask]
        radius = self.max_screen_radius[keep_mask]
        self.position_gradient_accumulator = accumulator
        self.position_gradient_denominator = denominator
        self.max_screen_radius = radius

    def _validate_render_shapes(
        self,
        *,
        visible_mask: Tensor,
        indices: Tensor,
        means_screen: Tensor,
        radii: Tensor,
    ) -> None:
        if visible_mask.dtype != torch.bool:
            raise TypeError("visible_mask must have dtype torch.bool")
        if visible_mask.shape != (self.num_gaussians,):
            raise ValueError(
                "visible_mask must have shape "
                f"({self.num_gaussians},), got {tuple(visible_mask.shape)}"
            )
        if indices.dtype != torch.int64:
            raise TypeError("projected.original_indices must have dtype torch.int64")
        if indices.ndim != 1:
            raise ValueError("projected.original_indices must be one-dimensional")
        if means_screen.shape != (indices.shape[0], 2):
            raise ValueError(
                "projected.means_screen must have shape (visible_count, 2)"
            )
        if radii.shape != indices.shape:
            raise ValueError("projected.radii must match original_indices shape")
        if radii.dtype != self.max_screen_radius.dtype:
            raise TypeError(
                "projected.radii dtype does not match max_screen_radius: "
                f"{radii.dtype} != {self.max_screen_radius.dtype}"
            )
        tensors = (visible_mask, indices, means_screen, radii)
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("render tensors and density statistics must share a device")

        visible_count = indices.shape[0]
        if visible_count == 0:
            if torch.count_nonzero(visible_mask).item() != 0:
                raise ValueError("visible_mask and projected rows are inconsistent")
            return
        if torch.any(indices < 0) or torch.any(indices >= self.num_gaussians):
            raise IndexError("projected.original_indices contains an invalid index")
        if torch.unique(indices).numel() != visible_count:
            raise ValueError("projected.original_indices contains duplicate indices")
        if (
            torch.count_nonzero(visible_mask).item() != visible_count
            or not torch.all(visible_mask[indices])
        ):
            raise ValueError("visible_mask and projected.original_indices are inconsistent")
