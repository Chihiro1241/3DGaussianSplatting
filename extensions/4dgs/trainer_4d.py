"""Frame-to-frame Gaussian hand-off for the 4D Gaussian Splatting extension.

This module implements section D of the formulation document.  A time series is
optimized frame by frame with the *unmodified* 3DGS training stack: frame
``f = 1`` is initialized from the SfM point cloud exactly like a static scene,
and frame ``f >= 2`` starts from the parameters optimized for frame ``f - 1``.

TeX: eq:4dgs_frame_parameters (149), eq:4dgs_parameter_handoff (150),
eq:4dgs_parameter_handoff_components (151).

Only the learnable Gaussian parameters cross a frame boundary.  The per-frame
optimization state is rebuilt from scratch for every frame, which is how this
module realizes the documented resets:

* the Adam first and second moments (a fresh :func:`create_optimizer` holds no
  state for any parameter), and
* the adaptive-density-control statistics -- position-gradient accumulator,
  observation count, and maximum screen-space projection radius (a fresh
  :class:`ScreenSpaceDensityStatistics`).

:func:`assert_frame_state_is_reset` verifies both properties before a frame
starts training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from gaussian_splatting.config import Config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.io.checkpoint import (
    model_from_checkpoint_state,
    read_checkpoint,
)
from gaussian_splatting.model import GaussianModel, initialize_gaussian_model
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
)
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import (
    PositionLearningRateScheduler,
    compute_scene_extent,
)


def _validated_frame_number(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class FrameHandoff:
    """Optimized Gaussian parameters carried into the next frame.

    The stored tensors are detached copies of the raw parameters of frame
    ``source_frame`` -- centre position, raw quaternion, raw scale, raw
    opacity, and the DC and higher-order spherical-harmonic coefficients --
    which equations (150) and (151) prescribe as the initial values of the
    following frame.  ``active_sh_degree`` travels with them so that the
    inherited higher-order coefficients keep contributing to rendering.
    """

    parameters: Mapping[str, Tensor]
    active_sh_degree: int
    source_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        count, _, _ = GaussianModel.validate_gaussian_parameter_tensors(
            self.parameters
        )
        if count == 0:
            raise ValueError("a frame hand-off must contain at least one Gaussian")
        if type(self.active_sh_degree) is not int or self.active_sh_degree < 0:
            raise ValueError("active_sh_degree must be a non-negative integer")
        _validated_frame_number(self.source_frame, "source_frame")

    @classmethod
    def from_model(
        cls,
        model: GaussianModel,
        *,
        source_frame: int,
        device: str | torch.device | None = None,
    ) -> FrameHandoff:
        """Snapshot a trained model's raw parameters (TeX: eq:4dgs_parameter_handoff).

        ``device`` optionally relocates the snapshot, which lets a caller free
        the training device before the next frame's images are loaded.
        """

        if not isinstance(model, GaussianModel):
            raise TypeError("model must be a GaussianModel")
        _validated_frame_number(source_frame, "source_frame")
        target = None if device is None else torch.device(device)
        with torch.no_grad():
            parameters = {
                name: (
                    value.detach().clone()
                    if target is None
                    else value.detach().to(device=target, copy=True)
                )
                for name, value in model.gaussian_parameter_dict().items()
            }
        return cls(
            parameters=parameters,
            active_sh_degree=int(model.active_sh_degree),
            source_frame=int(source_frame),
        )

    @property
    def num_gaussians(self) -> int:
        """Return the number of Gaussians inherited by the next frame."""

        return int(self.parameters["means_world"].shape[0])

    def to_model(
        self,
        config: Config,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> GaussianModel:
        """Build the next frame's initial model from the inherited parameters.

        TeX: eq:4dgs_parameter_handoff_components.  No transformation is
        applied: the raw parameters become the new frame's initial raw
        parameters, so the rendered scale, opacity, and rotation are identical
        to the previous frame's optimized values.
        """

        if not isinstance(config, Config):
            raise TypeError("config must be a Config")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        tensors = {
            name: value.detach().to(device=device, dtype=dtype)
            for name, value in self.parameters.items()
        }
        model = GaussianModel(
            **tensors,
            epsilon_q=config.model.epsilon_q,
            sh_degree=config.model.sh_degree,
        )
        model.set_active_sh_degree(
            min(self.active_sh_degree, config.model.sh_degree)
        )
        return model


def handoff_from_checkpoint(
    path: str | Path,
    config: Config,
    *,
    source_frame: int,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    active_sh_degree: int | None = None,
) -> FrameHandoff:
    """Rebuild a hand-off from a finished frame's checkpoint.

    This is the restart path for an interrupted sequence: the checkpoint of
    frame ``source_frame`` supplies the initial values of frame
    ``source_frame + 1``.  Only the Gaussian parameters are read; the
    checkpoint's optimizer, scheduler, and density statistics are ignored
    because a new frame resets them.

    ``active_sh_degree`` overrides the degree derived from the checkpoint's
    within-frame iteration count, which matters when frames are trained for
    fewer iterations than the progressive-SH schedule needs.
    """

    state = read_checkpoint(path, map_location="cpu")
    model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
    if active_sh_degree is not None:
        model.set_active_sh_degree(int(active_sh_degree))
    return FrameHandoff.from_model(model, source_frame=source_frame)


@dataclass(frozen=True)
class FrameTrainingState:
    """Everything one frame needs, with all per-frame state freshly reset."""

    model: GaussianModel
    optimizer: torch.optim.Adam
    scheduler: PositionLearningRateScheduler
    density_statistics: ScreenSpaceDensityStatistics | None
    scene_extent: float
    carried_over: bool

    @property
    def num_gaussians(self) -> int:
        """Return the frame's initial Gaussian count."""

        return self.model.num_gaussians


def build_frame_training_state(
    *,
    config: Config,
    train_cameras: Sequence[Camera],
    device: str | torch.device,
    dtype: torch.dtype,
    handoff: FrameHandoff | None = None,
    initial_points: Tensor | None = None,
    initial_colors: Tensor | None = None,
) -> FrameTrainingState:
    """Create one frame's model, optimizer, scheduler, and ADC statistics.

    With ``handoff=None`` the model is initialized from ``initial_points`` and
    ``initial_colors`` exactly as static 3DGS initializes frame ``f = 1``.
    Otherwise the inherited parameters of frame ``f - 1`` become the initial
    values (TeX: eq:4dgs_parameter_handoff).

    In both cases the optimizer, the position learning-rate schedule, and the
    density-control statistics are constructed from scratch, so the Adam
    moments and the accumulated position gradients, observation counts, and
    maximum projection radii all start at zero for every frame.
    """

    if not isinstance(config, Config):
        raise TypeError("config must be a Config")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    if handoff is None:
        if initial_points is None or initial_colors is None:
            raise ValueError(
                "the first trained frame requires initial points and colors"
            )
        model = initialize_gaussian_model(
            initial_points.to(device=device, dtype=dtype),
            initial_colors.to(device=device, dtype=dtype),
            config,
        )
    else:
        if initial_points is not None or initial_colors is not None:
            raise ValueError(
                "a carried-over frame must not be re-initialized from points"
            )
        model = handoff.to_model(config, device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)
    if model.num_gaussians == 0:
        raise RuntimeError("a frame cannot start training with zero Gaussians")

    scene_extent = compute_scene_extent(train_cameras)
    optimizer = create_optimizer(model, config, position_lr_scale=scene_extent)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial * scene_extent,
        final_learning_rate=config.training.position_lr_final * scene_extent,
    )
    density_statistics = (
        ScreenSpaceDensityStatistics.for_model(model)
        if config.features.adaptive_density_control
        else None
    )
    state = FrameTrainingState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        density_statistics=density_statistics,
        scene_extent=scene_extent,
        carried_over=handoff is not None,
    )
    assert_frame_state_is_reset(state)
    return state


def assert_frame_state_is_reset(state: FrameTrainingState) -> None:
    """Verify that no per-frame optimization state crossed the boundary.

    Raises:
        RuntimeError: if any Adam moment, position-gradient accumulator,
            observation count, or maximum projection radius is carried over.
    """

    if not isinstance(state, FrameTrainingState):
        raise TypeError("state must be a FrameTrainingState")
    if len(state.optimizer.state) != 0:
        raise RuntimeError(
            "Adam moments must be empty when a frame starts training"
        )
    if state.scheduler.current_iteration != 0:
        raise RuntimeError(
            "the position learning-rate schedule must start a frame at zero"
        )
    if any(
        parameter.grad is not None for parameter in state.model.parameters()
    ):
        raise RuntimeError("parameter gradients must be empty when a frame starts")
    statistics = state.density_statistics
    if statistics is None:
        return
    statistics.validate_compatible(state.model)
    accumulated = (
        bool(statistics.position_gradient_accumulator.any().item())
        or bool(statistics.position_gradient_denominator.any().item())
        or bool(statistics.max_screen_radius.any().item())
    )
    if accumulated:
        raise RuntimeError(
            "density-control statistics must be zero when a frame starts training"
        )


def frame_output_directory(root: str | Path, frame_number: int) -> Path:
    """Return the per-frame run directory below a 4D output root."""

    _validated_frame_number(frame_number, "frame_number")
    return Path(root) / f"frame_{frame_number:04d}"


__all__ = [
    "FrameHandoff",
    "FrameTrainingState",
    "assert_frame_state_is_reset",
    "build_frame_training_state",
    "frame_output_directory",
    "handoff_from_checkpoint",
]
