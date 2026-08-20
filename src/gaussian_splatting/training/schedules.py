"""Learning-rate schedules used by Gaussian Splatting training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch
from torch.optim import Optimizer

from gaussian_splatting.config import Config
from gaussian_splatting.data.camera import Camera


@dataclass(frozen=True)
class DensityControlScheduleDecision:
    """Pure iteration policy for optional density-control operations."""

    collect_statistics: bool
    run_density_control_event: bool
    run_opacity_reset: bool
    size_pruning_active: bool


@dataclass(frozen=True)
class DensityControlEventParameters:
    """Thresholds passed to one low-level density-control event."""

    gradient_threshold: float
    densify_world_scale_threshold: float
    prune_opacity_threshold: float
    prune_screen_radius_threshold: float | None
    prune_world_scale_threshold: float | None


def compute_scene_extent(train_cameras: Sequence[Camera]) -> float:
    """Return the official 3DGS-style radius of the training camera centers.

    The extent is ``1.1 * max(norm(center_i - mean(center_i)))``. Callers must
    pass only training cameras; evaluation cameras are intentionally not an
    input to this API.
    """
    if not isinstance(train_cameras, Sequence) or isinstance(
        train_cameras, (str, bytes)
    ):
        raise TypeError("train_cameras must be a sequence of Camera objects")
    if len(train_cameras) == 0:
        raise ValueError("train_cameras must not be empty")
    if any(not isinstance(camera, Camera) for camera in train_cameras):
        raise TypeError("train_cameras must contain only Camera objects")

    first_center = train_cameras[0].camera_center_world
    if not isinstance(first_center, torch.Tensor):
        raise TypeError("camera centers must be torch.Tensor objects")
    centers: list[torch.Tensor] = []
    for camera in train_cameras:
        center = camera.camera_center_world
        if not isinstance(center, torch.Tensor):
            raise TypeError("camera centers must be torch.Tensor objects")
        if center.shape != (3,):
            raise ValueError("every camera center must have shape (3,)")
        if center.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "camera centers must have dtype torch.float32 or torch.float64"
            )
        if center.dtype != first_center.dtype:
            raise TypeError("all training camera centers must have the same dtype")
        if center.device != first_center.device:
            raise ValueError("all training camera centers must be on the same device")
        if not torch.isfinite(center).all().item():
            raise ValueError("training camera centers must be finite")
        centers.append(center.detach())

    stacked_centers = torch.stack(centers, dim=0)
    center = stacked_centers.mean(dim=0)
    diagonal = torch.linalg.vector_norm(
        stacked_centers - center,
        dim=-1,
    ).amax()
    scene_extent = diagonal * 1.1
    if not torch.isfinite(scene_extent).item():
        raise ValueError("computed scene extent must be finite")
    if scene_extent.item() <= 0.0:
        raise ValueError("computed scene extent must be positive")
    return float(scene_extent.item())


def _validated_iteration(iteration: int) -> int:
    if type(iteration) is not int:
        raise TypeError("iteration must be an integer")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return iteration


def _validated_policy_config(config: Config) -> Config:
    if not isinstance(config, Config):
        raise TypeError("config must be a Config")
    return config


def _validated_scene_extent(scene_extent: Real) -> float:
    if isinstance(scene_extent, bool) or not isinstance(scene_extent, Real):
        raise TypeError("scene_extent must be a real number")
    value = float(scene_extent)
    if not math.isfinite(value):
        raise ValueError("scene_extent must be finite")
    if value <= 0.0:
        raise ValueError("scene_extent must be positive")
    return value


def density_control_schedule(
    config: Config,
    iteration: int,
) -> DensityControlScheduleDecision:
    """Return deterministic ADC decisions without mutating runtime state."""
    validated_config = _validated_policy_config(config)
    current_iteration = _validated_iteration(iteration)
    density = validated_config.density_control
    features = validated_config.features
    inside_statistics_window = (
        current_iteration < density.densify_until_iteration
    )
    collect_statistics = (
        features.adaptive_density_control and inside_statistics_window
    )
    run_density_control_event = (
        features.adaptive_density_control
        and current_iteration > density.densify_from_iteration
        and inside_statistics_window
        and current_iteration % density.densification_interval == 0
    )
    periodic_opacity_reset = (
        current_iteration % density.opacity_reset_interval == 0
    )
    white_background_reset = (
        validated_config.data.rgba_background == "white"
        and current_iteration == density.densify_from_iteration
    )
    run_opacity_reset = (
        features.opacity_reset
        and inside_statistics_window
        and (periodic_opacity_reset or white_background_reset)
    )
    size_pruning_active = (
        features.adaptive_density_control
        and inside_statistics_window
        and current_iteration > density.opacity_reset_interval
    )
    return DensityControlScheduleDecision(
        collect_statistics=collect_statistics,
        run_density_control_event=run_density_control_event,
        run_opacity_reset=run_opacity_reset,
        size_pruning_active=size_pruning_active,
    )


def density_control_event_parameters(
    config: Config,
    iteration: int,
    scene_extent: Real,
) -> DensityControlEventParameters:
    """Derive event thresholds for an iteration and positive scene extent."""
    validated_config = _validated_policy_config(config)
    extent = _validated_scene_extent(scene_extent)
    decision = density_control_schedule(validated_config, iteration)
    density = validated_config.density_control
    return DensityControlEventParameters(
        gradient_threshold=density.position_gradient_threshold,
        densify_world_scale_threshold=density.percent_dense * extent,
        prune_opacity_threshold=density.prune_opacity_threshold,
        prune_screen_radius_threshold=(
            density.prune_screen_radius_threshold
            if decision.size_pruning_active
            else None
        ),
        prune_world_scale_threshold=(
            density.prune_world_scale_fraction * extent
            if decision.size_pruning_active
            else None
        ),
    )


def position_learning_rate_schedule(
    iteration: int,
    total_iterations: int = 30_000,
    initial_learning_rate: float = 1.6e-4,
    final_learning_rate: float = 1.6e-6,
) -> float:
    """Return the exponentially interpolated position learning rate.

    TeX: eq:position_learning_rate_schedule
    """

    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if initial_learning_rate <= 0.0 or final_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    clamped_iteration = min(max(int(iteration), 0), int(total_iterations))
    fraction = clamped_iteration / total_iterations
    return math.exp(
        (1.0 - fraction) * math.log(initial_learning_rate)
        + fraction * math.log(final_learning_rate)
    )


class PositionLearningRateScheduler:
    """Update only the optimizer's ``means_world`` parameter group."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_iterations: int = 30_000,
        initial_learning_rate: float = 1.6e-4,
        final_learning_rate: float = 1.6e-6,
    ) -> None:
        self.optimizer = optimizer
        self.total_iterations = total_iterations
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.current_iteration = 0
        self._position_group()

    def _position_group(self) -> dict[str, Any]:
        groups = [g for g in self.optimizer.param_groups if g.get("name") == "means_world"]
        if len(groups) != 1:
            raise ValueError("optimizer must contain exactly one 'means_world' group")
        return groups[0]

    def step(self, iteration: int) -> float:
        """Set and return the absolute position learning rate for an iteration."""

        self.current_iteration = min(max(int(iteration), 0), self.total_iterations)
        learning_rate = position_learning_rate_schedule(
            self.current_iteration,
            self.total_iterations,
            self.initial_learning_rate,
            self.final_learning_rate,
        )
        self._position_group()["lr"] = learning_rate
        return learning_rate

    def state_dict(self) -> dict[str, int | float]:
        return {
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
        }

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        required = {
            "current_iteration",
            "total_iterations",
            "initial_learning_rate",
            "final_learning_rate",
        }
        missing = required.difference(state_dict)
        if missing:
            raise ValueError(f"scheduler state is missing keys: {sorted(missing)}")
        if int(state_dict["total_iterations"]) != self.total_iterations:
            raise ValueError("scheduler total_iterations does not match")
        if float(state_dict["initial_learning_rate"]) != self.initial_learning_rate:
            raise ValueError("scheduler initial learning rate does not match")
        if float(state_dict["final_learning_rate"]) != self.final_learning_rate:
            raise ValueError("scheduler final learning rate does not match")
        self.step(int(state_dict["current_iteration"]))


__all__ = [
    "DensityControlEventParameters",
    "DensityControlScheduleDecision",
    "PositionLearningRateScheduler",
    "compute_scene_extent",
    "density_control_event_parameters",
    "density_control_schedule",
    "position_learning_rate_schedule",
]
