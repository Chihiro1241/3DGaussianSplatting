"""Screen-space statistics used by adaptive density control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from gaussian_splatting.math.covariance import quaternion_rotation_matrix
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GaussianModel,
)
from gaussian_splatting.renderer.renderer import RenderResult
from gaussian_splatting.training.optimizer import (
    append_gaussian_parameters,
    keep_and_append_gaussian_parameters,
    keep_gaussian_parameters,
)


@dataclass(frozen=True)
class GaussianCloneResult:
    """Counts from one deterministic Gaussian clone transaction."""

    num_gaussians_before: int
    num_gaussians_after: int
    num_cloned: int


@dataclass(frozen=True)
class GaussianPruneResult:
    """Counts from one pruning transaction.

    Reason-specific counts are intentionally non-exclusive. A Gaussian that
    matches multiple criteria contributes once to ``num_pruned_total`` and
    once to every matching reason count.
    """

    num_gaussians_before: int
    num_gaussians_after: int
    num_pruned_total: int
    num_low_opacity: int
    num_large_screen: int
    num_large_world: int


@dataclass(frozen=True)
class GaussianSplitResult:
    """Counts from one two-child Gaussian split transaction."""

    num_gaussians_before: int
    num_gaussians_after: int
    num_split_parents: int
    num_children_created: int


@dataclass(frozen=True)
class _StatisticsState:
    position_gradient_accumulator: Tensor
    position_gradient_denominator: Tensor
    max_screen_radius: Tensor


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
        if not isinstance(model, GaussianModel):
            raise TypeError("model must be a GaussianModel")
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
        expected_shape = (model.num_gaussians,)
        states = (
            (
                "position_gradient_accumulator",
                self.position_gradient_accumulator,
                model.means_world.dtype,
            ),
            (
                "position_gradient_denominator",
                self.position_gradient_denominator,
                torch.int64,
            ),
            ("max_screen_radius", self.max_screen_radius, torch.int64),
        )
        for name, value, expected_dtype in states:
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
                )
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must have dtype {expected_dtype}, got {value.dtype}"
                )
            if value.device != model.means_world.device:
                raise ValueError(f"{name} must be on the model device")
        if not torch.isfinite(self.position_gradient_accumulator).all().item():
            raise ValueError("position_gradient_accumulator must be finite")
        if torch.any(self.position_gradient_accumulator < 0).item():
            raise ValueError("position_gradient_accumulator must be non-negative")
        if torch.any(self.position_gradient_denominator < 0).item():
            raise ValueError("position_gradient_denominator must be non-negative")
        if torch.any(self.max_screen_radius < 0).item():
            raise ValueError("max_screen_radius must be non-negative")

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
        state = self._prepare_append_zeros(count)
        self._commit_state(state)

    def _prepare_append_zeros(self, count: int) -> _StatisticsState:
        """Build an appended zero state without changing current statistics."""
        if count < 0:
            raise ValueError("append count must be non-negative")
        if count == 0:
            return _StatisticsState(
                position_gradient_accumulator=(
                    self.position_gradient_accumulator
                ),
                position_gradient_denominator=(
                    self.position_gradient_denominator
                ),
                max_screen_radius=self.max_screen_radius,
            )

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
        return _StatisticsState(
            position_gradient_accumulator=accumulator,
            position_gradient_denominator=denominator,
            max_screen_radius=radius,
        )

    def _prepare_keep_and_append_zeros(
        self,
        keep_mask: Tensor,
        append_count: int,
    ) -> _StatisticsState:
        """Build one final state containing kept rows followed by zero rows."""
        kept = self._prepare_keep(keep_mask)
        if append_count < 0:
            raise ValueError("append count must be non-negative")
        return _StatisticsState(
            position_gradient_accumulator=torch.cat(
                (
                    kept.position_gradient_accumulator,
                    self.position_gradient_accumulator.new_zeros(append_count),
                )
            ),
            position_gradient_denominator=torch.cat(
                (
                    kept.position_gradient_denominator,
                    self.position_gradient_denominator.new_zeros(append_count),
                )
            ),
            max_screen_radius=torch.cat(
                (
                    kept.max_screen_radius,
                    self.max_screen_radius.new_zeros(append_count),
                )
            ),
        )

    def keep(self, keep_mask: Tensor) -> None:
        """Keep the same selected Gaussian indices in every statistic."""
        state = self._prepare_keep(keep_mask)
        self._commit_state(state)

    def _prepare_keep(self, keep_mask: Tensor) -> _StatisticsState:
        """Validate a mask and build all replacement tensors without mutation."""
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

        return _StatisticsState(
            position_gradient_accumulator=(
                self.position_gradient_accumulator[keep_mask]
            ),
            position_gradient_denominator=(
                self.position_gradient_denominator[keep_mask]
            ),
            max_screen_radius=self.max_screen_radius[keep_mask],
        )

    def _commit_state(self, state: _StatisticsState) -> None:
        """Install a fully prepared state, restoring old references on failure."""
        previous = _StatisticsState(
            position_gradient_accumulator=self.position_gradient_accumulator,
            position_gradient_denominator=self.position_gradient_denominator,
            max_screen_radius=self.max_screen_radius,
        )
        try:
            self.position_gradient_accumulator = (
                state.position_gradient_accumulator
            )
            self.position_gradient_denominator = (
                state.position_gradient_denominator
            )
            self.max_screen_radius = state.max_screen_radius
        except BaseException:
            self.position_gradient_accumulator = (
                previous.position_gradient_accumulator
            )
            self.position_gradient_denominator = (
                previous.position_gradient_denominator
            )
            self.max_screen_radius = previous.max_screen_radius
            raise

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


def _validated_threshold(
    value: Real | None,
    name: str,
    *,
    allow_none: bool,
    unit_interval: bool = False,
    positive: bool = False,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError(f"{name} must be finite")
    if unit_interval:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must lie in the closed interval [0, 1]")
    elif positive:
        if threshold <= 0.0:
            raise ValueError(f"{name} must be positive")
    elif threshold < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return threshold


def clone_gaussians(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    statistics: ScreenSpaceDensityStatistics,
    *,
    gradient_threshold: Real,
    world_scale_threshold: Real,
) -> GaussianCloneResult:
    """Clone small Gaussians whose mean screen-position gradient is high.

    Selection uses inclusive boundaries: ``mean_gradient >= threshold`` and
    ``max(actual_scale) <= threshold``. Selected raw rows are copied exactly
    and appended in ascending original-index order.
    """
    if not isinstance(model, GaussianModel):
        raise TypeError("model must be a GaussianModel")
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("optimizer must be torch.optim.Adam")
    if not isinstance(statistics, ScreenSpaceDensityStatistics):
        raise TypeError("statistics must be ScreenSpaceDensityStatistics")
    gradient_threshold_value = _validated_threshold(
        gradient_threshold,
        "gradient_threshold",
        allow_none=False,
    )
    world_threshold_value = _validated_threshold(
        world_scale_threshold,
        "world_scale_threshold",
        allow_none=False,
        positive=True,
    )
    if gradient_threshold_value is None:  # pragma: no cover - validated above
        raise RuntimeError("gradient threshold validation returned None")
    if world_threshold_value is None:  # pragma: no cover - validated above
        raise RuntimeError("world scale threshold validation returned None")
    statistics.validate_compatible(model)

    with torch.no_grad():
        mean_gradient = statistics.mean_position_gradient()
        maximum_world_scale = model.transformed_parameters().scales.amax(dim=-1)
        high_gradient = mean_gradient >= gradient_threshold_value
        small_scale = maximum_world_scale <= world_threshold_value
        clone_mask = high_gradient & small_scale
        num_gaussians_before = model.num_gaussians
        num_cloned = int(clone_mask.sum().item())
        additions = {
            name: getattr(model, name).detach()[clone_mask]
            for name in GAUSSIAN_PARAMETER_NAMES
        }

    result = GaussianCloneResult(
        num_gaussians_before=num_gaussians_before,
        num_gaussians_after=num_gaussians_before + num_cloned,
        num_cloned=num_cloned,
    )
    if num_cloned == 0:
        append_gaussian_parameters(model, optimizer, additions)
        return result

    prepared_statistics = statistics._prepare_append_zeros(num_cloned)
    append_gaussian_parameters(
        model,
        optimizer,
        additions,
        commit_callback=lambda: statistics._commit_state(prepared_statistics),
    )
    return result


def _rng_state(device: torch.device) -> Tensor:
    if device.type == "cpu":
        return torch.get_rng_state()
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    raise ValueError("Gaussian split RNG supports CPU and CUDA devices")


def _restore_rng_state(device: torch.device, state: Tensor) -> None:
    if device.type == "cpu":
        torch.set_rng_state(state)
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
        return
    raise ValueError("Gaussian split RNG supports CPU and CUDA devices")


def split_gaussians(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    statistics: ScreenSpaceDensityStatistics,
    *,
    gradient_threshold: Real,
    world_scale_threshold: Real,
) -> GaussianSplitResult:
    """Replace each selected large, high-gradient Gaussian with two children.

    Children are ordered like ``selected.repeat(2, ...)``: child A for every
    selected parent in original-index order, followed by child B in the same
    parent order. Sampling uses the model device's global PyTorch RNG.
    """
    if not isinstance(model, GaussianModel):
        raise TypeError("model must be a GaussianModel")
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("optimizer must be torch.optim.Adam")
    if not isinstance(statistics, ScreenSpaceDensityStatistics):
        raise TypeError("statistics must be ScreenSpaceDensityStatistics")
    gradient_threshold_value = _validated_threshold(
        gradient_threshold,
        "gradient_threshold",
        allow_none=False,
    )
    world_threshold_value = _validated_threshold(
        world_scale_threshold,
        "world_scale_threshold",
        allow_none=False,
        positive=True,
    )
    if gradient_threshold_value is None:  # pragma: no cover - validated above
        raise RuntimeError("gradient threshold validation returned None")
    if world_threshold_value is None:  # pragma: no cover - validated above
        raise RuntimeError("world scale threshold validation returned None")
    statistics.validate_compatible(model)

    with torch.no_grad():
        transformed = model.transformed_parameters()
        mean_gradient = statistics.mean_position_gradient()
        maximum_world_scale = transformed.scales.amax(dim=-1)
        split_mask = (mean_gradient >= gradient_threshold_value) & (
            maximum_world_scale > world_threshold_value
        )
        keep_mask = ~split_mask
        num_gaussians_before = model.num_gaussians
        num_split_parents = int(split_mask.sum().item())
        num_children_created = 2 * num_split_parents

    result = GaussianSplitResult(
        num_gaussians_before=num_gaussians_before,
        num_gaussians_after=(
            num_gaussians_before - num_split_parents + num_children_created
        ),
        num_split_parents=num_split_parents,
        num_children_created=num_children_created,
    )

    # Validate model/optimizer references and every existing Adam state before
    # sampling. The all-true keep is guaranteed not to replace any object.
    keep_gaussian_parameters(
        model,
        optimizer,
        torch.ones_like(split_mask),
    )
    if num_split_parents == 0:
        return result

    device = model.means_world.device
    rng_state = _rng_state(device)
    try:
        with torch.no_grad():
            parent_scales = transformed.scales[split_mask]
            parent_rotations = quaternion_rotation_matrix(
                transformed.quaternions[split_mask]
            )
            repeated_scales = parent_scales.repeat(2, 1)
            repeated_rotations = parent_rotations.repeat(2, 1, 1)
            local_samples = repeated_scales * torch.randn(
                (num_children_created, 3),
                dtype=model.means_world.dtype,
                device=device,
            )
            world_offsets = torch.bmm(
                repeated_rotations, local_samples.unsqueeze(-1)
            ).squeeze(-1)
            additions: dict[str, Tensor] = {
                "means_world": (
                    model.means_world.detach()[split_mask].repeat(2, 1)
                    + world_offsets
                ),
                "raw_scales": torch.log(repeated_scales / 1.6),
            }
            for name in (
                "raw_quaternions",
                "raw_opacities",
                "sh_dc",
                "sh_rest",
            ):
                parent_values = getattr(model, name).detach()[split_mask]
                repeats = (2,) + (1,) * (parent_values.ndim - 1)
                additions[name] = parent_values.repeat(repeats)

            prepared_statistics = statistics._prepare_keep_and_append_zeros(
                keep_mask,
                num_children_created,
            )
        keep_and_append_gaussian_parameters(
            model,
            optimizer,
            keep_mask,
            additions,
            commit_callback=lambda: statistics._commit_state(
                prepared_statistics
            ),
        )
    except BaseException:
        _restore_rng_state(device, rng_state)
        raise
    return result


def prune_gaussians(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    statistics: ScreenSpaceDensityStatistics,
    *,
    opacity_threshold: Real,
    screen_radius_threshold: Real | None = None,
    world_scale_threshold: Real | None = None,
) -> GaussianPruneResult:
    """Prune the union of low-opacity, large-screen, and large-world rows.

    Opacity uses ``actual_opacity < opacity_threshold``; equality is kept.
    Enabled size criteria use strict ``>`` comparisons. Reason counts in the
    returned result are non-exclusive, while ``num_pruned_total`` is the size
    of their union.
    """
    if not isinstance(model, GaussianModel):
        raise TypeError("model must be a GaussianModel")
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("optimizer must be torch.optim.Adam")
    if not isinstance(statistics, ScreenSpaceDensityStatistics):
        raise TypeError("statistics must be ScreenSpaceDensityStatistics")
    opacity_threshold_value = _validated_threshold(
        opacity_threshold,
        "opacity_threshold",
        allow_none=False,
        unit_interval=True,
    )
    screen_threshold_value = _validated_threshold(
        screen_radius_threshold,
        "screen_radius_threshold",
        allow_none=True,
    )
    world_threshold_value = _validated_threshold(
        world_scale_threshold,
        "world_scale_threshold",
        allow_none=True,
    )
    if opacity_threshold_value is None:  # pragma: no cover - validated above
        raise RuntimeError("opacity threshold validation returned None")
    statistics.validate_compatible(model)

    with torch.no_grad():
        transformed = model.transformed_parameters()
        actual_opacities = transformed.opacities.squeeze(-1)
        maximum_world_scales = transformed.scales.amax(dim=-1)
        low_opacity = actual_opacities < opacity_threshold_value
        if screen_threshold_value is None:
            large_screen = torch.zeros_like(low_opacity)
        else:
            large_screen = (
                statistics.max_screen_radius > screen_threshold_value
            )
        if world_threshold_value is None:
            large_world = torch.zeros_like(low_opacity)
        else:
            large_world = maximum_world_scales > world_threshold_value
        prune_mask = low_opacity | large_screen | large_world
        keep_mask = ~prune_mask

        num_gaussians_before = model.num_gaussians
        num_low_opacity = int(low_opacity.sum().item())
        num_large_screen = int(large_screen.sum().item())
        num_large_world = int(large_world.sum().item())
        num_pruned_total = int(prune_mask.sum().item())

    result = GaussianPruneResult(
        num_gaussians_before=num_gaussians_before,
        num_gaussians_after=num_gaussians_before - num_pruned_total,
        num_pruned_total=num_pruned_total,
        num_low_opacity=num_low_opacity,
        num_large_screen=num_large_screen,
        num_large_world=num_large_world,
    )
    if num_pruned_total == 0:
        keep_gaussian_parameters(model, optimizer, keep_mask)
        return result

    prepared_statistics = statistics._prepare_keep(keep_mask)
    keep_gaussian_parameters(
        model,
        optimizer,
        keep_mask,
        commit_callback=lambda: statistics._commit_state(prepared_statistics),
    )
    return result


__all__ = [
    "GaussianCloneResult",
    "GaussianPruneResult",
    "GaussianSplitResult",
    "ScreenSpaceDensityStatistics",
    "clone_gaussians",
    "prune_gaussians",
    "split_gaussians",
]
