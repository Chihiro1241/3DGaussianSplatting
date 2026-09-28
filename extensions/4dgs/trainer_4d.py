"""Frame-to-frame Gaussian hand-off for the 4D Gaussian Splatting extension.

This module implements section D of the formulation document.  A time series is
optimized frame by frame with the *unmodified* 3DGS training stack: frame
``f = 1`` is initialized from the SfM point cloud exactly like a static scene,
and frame ``f >= 2`` starts from the parameters optimized for frame ``f - 1``.

TeX: eq:4dgs_frame_parameters (149), eq:4dgs_parameter_handoff (150),
eq:4dgs_parameter_handoff_components (151).

Only the learnable Gaussian parameters cross a frame boundary by default.  The
per-frame optimization state is rebuilt from scratch for every frame, which is
how this module realizes the documented resets:

* the Adam first and second moments (a fresh :func:`create_optimizer` holds no
  state for any parameter), and
* the adaptive-density-control statistics -- position-gradient accumulator,
  observation count, and maximum screen-space projection radius (a fresh
  :class:`ScreenSpaceDensityStatistics`).

:func:`assert_frame_state_is_reset` verifies both properties before a frame
starts training.

``config.warm_start`` relaxes the first of the two for warm-start experiments.
With ``adam_state="carry"`` the previous frame's Adam moments are installed
into the new optimizer, which is only meaningful while the Gaussian count is
held fixed -- a densified or pruned frame has no correspondence between the old
and new moment tensors, and this module refuses the carry rather than guessing
one.  The density-control statistics always reset.  ``position_lr_mode="fixed"``
likewise replaces the ``total_iterations``-dependent exponential position
schedule with a constant rate, so that a frame's iteration budget and its
learning-rate schedule stop being coupled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
    FixedPositionLearningRateScheduler,
    PositionLearningRateScheduler,
    compute_scene_extent,
)

#: Adam moment names carried across a frame boundary.  ``step`` travels with
#: them because Adam's bias correction depends on it; dropping it would make an
#: inherited moment behave as if it came from a fresh run.
_ADAM_STATE_NAMES = ("step", "exp_avg", "exp_avg_sq")


def _validated_frame_number(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _as_tensor(value: object, name: str) -> Tensor:
    """Return Adam's ``step`` (a scalar in some releases) as a tensor."""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(float(value))
    raise TypeError(f"Adam state {name!r} must be a Tensor or a number")


def _captured_adam_state(
    optimizer: torch.optim.Optimizer,
    relocate: Callable[[Tensor], Tensor],
) -> dict[str, dict[str, Tensor]]:
    """Read the per-parameter Adam moments out of a live optimizer.

    Groups whose parameter has never been stepped hold no state and are simply
    absent from the result; installing a partial carry is still well defined
    because Adam creates the missing entries on its first step.
    """

    captured: dict[str, dict[str, Tensor]] = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if name is None:
            raise ValueError("every optimizer parameter group must be named")
        if len(group["params"]) != 1:
            raise ValueError(
                f"optimizer group {name!r} must hold exactly one parameter"
            )
        entry = optimizer.state.get(group["params"][0])
        if not entry:
            continue
        captured[name] = {
            moment: relocate(_as_tensor(entry[moment], moment))
            for moment in _ADAM_STATE_NAMES
            if moment in entry
        }
    return captured


def _adam_state_from_checkpoint(
    optimizer_state: Mapping[str, object],
) -> dict[str, dict[str, Tensor]]:
    """Recover named Adam moments from a saved ``optimizer.state_dict()``.

    ``Optimizer.state_dict`` replaces each group's parameters with integer
    indices but keeps the group's other keys, so the ``name`` this repository
    assigns in :func:`create_optimizer` survives and gives the mapping back.
    """

    indexed = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(indexed, Mapping) or not isinstance(groups, Sequence):
        raise ValueError(
            "checkpoint optimizer_state_dict must contain 'state' and "
            "'param_groups'"
        )
    captured: dict[str, dict[str, Tensor]] = {}
    for group in groups:
        name = group.get("name")
        if name is None:
            raise ValueError(
                "checkpoint optimizer parameter groups must be named; this "
                "checkpoint predates named groups and cannot supply a carry"
            )
        indices = group.get("params", [])
        if len(indices) != 1:
            raise ValueError(
                f"optimizer group {name!r} must hold exactly one parameter"
            )
        entry = indexed.get(indices[0])
        if not entry:
            continue
        captured[name] = {
            moment: _as_tensor(entry[moment], moment).detach().clone()
            for moment in _ADAM_STATE_NAMES
            if moment in entry
        }
    return captured


def install_adam_state(
    optimizer: torch.optim.Optimizer,
    moments: Mapping[str, Mapping[str, Tensor]],
) -> int:
    """Install inherited Adam moments into a freshly built optimizer.

    Returns the number of parameter groups that received state.  Every moment
    must already match its parameter's shape, which is the caller's guarantee
    that the Gaussian count did not change across the frame boundary.
    """

    installed = 0
    for group in optimizer.param_groups:
        name = group.get("name")
        entry = moments.get(name)
        if entry is None:
            continue
        parameter = group["params"][0]
        for moment in ("exp_avg", "exp_avg_sq"):
            if moment in entry and entry[moment].shape != parameter.shape:
                raise ValueError(
                    f"inherited Adam {moment} for {name!r} has shape "
                    f"{tuple(entry[moment].shape)}, but the frame's parameter "
                    f"has shape {tuple(parameter.shape)}; an Adam carry "
                    "requires a fixed Gaussian count"
                )
        state: dict[str, Tensor] = {}
        for moment, value in entry.items():
            # ``step`` is a bookkeeping scalar and stays on the CPU, which is
            # where non-capturable Adam keeps it.
            state[moment] = (
                value.detach().clone()
                if moment == "step"
                else value.detach().to(
                    device=parameter.device, dtype=parameter.dtype, copy=True
                )
            )
        optimizer.state[parameter] = state
        installed += 1
    return installed


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
    optimizer_state: Mapping[str, Mapping[str, Tensor]] | None = None

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
        if self.optimizer_state is None:
            return
        moments = {
            name: dict(entry) for name, entry in self.optimizer_state.items()
        }
        unknown = sorted(set(moments) - set(self.parameters))
        if unknown:
            raise ValueError(
                f"hand-off optimizer state names unknown parameters: {unknown}"
            )
        for name, entry in moments.items():
            missing = sorted(set(_ADAM_STATE_NAMES) - set(entry))
            if missing:
                raise ValueError(
                    f"hand-off Adam state for {name!r} is missing: {missing}"
                )
            for moment in ("exp_avg", "exp_avg_sq"):
                if entry[moment].shape != self.parameters[name].shape:
                    raise ValueError(
                        f"hand-off Adam {moment} for {name!r} has shape "
                        f"{tuple(entry[moment].shape)}, expected "
                        f"{tuple(self.parameters[name].shape)}"
                    )
        object.__setattr__(self, "optimizer_state", moments)

    @classmethod
    def from_model(
        cls,
        model: GaussianModel,
        *,
        source_frame: int,
        device: str | torch.device | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> FrameHandoff:
        """Snapshot a trained model's raw parameters (TeX: eq:4dgs_parameter_handoff).

        ``device`` optionally relocates the snapshot, which lets a caller free
        the training device before the next frame's images are loaded.

        ``optimizer`` additionally snapshots that frame's Adam moments, for the
        ``warm_start.adam_state="carry"`` path.  It is read through the named
        parameter groups rather than through ``state_dict()`` so that no
        parameter-index bookkeeping is involved.
        """

        if not isinstance(model, GaussianModel):
            raise TypeError("model must be a GaussianModel")
        _validated_frame_number(source_frame, "source_frame")
        target = None if device is None else torch.device(device)

        def relocate(value: Tensor) -> Tensor:
            return (
                value.detach().clone()
                if target is None
                else value.detach().to(device=target, copy=True)
            )

        with torch.no_grad():
            parameters = {
                name: relocate(value)
                for name, value in model.gaussian_parameter_dict().items()
            }
            optimizer_state = (
                None
                if optimizer is None
                else _captured_adam_state(optimizer, relocate)
            )
        return cls(
            parameters=parameters,
            active_sh_degree=int(model.active_sh_degree),
            source_frame=int(source_frame),
            optimizer_state=optimizer_state,
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
    carry_optimizer_state: bool = False,
) -> FrameHandoff:
    """Rebuild a hand-off from a finished frame's checkpoint.

    This is the restart path for an interrupted sequence: the checkpoint of
    frame ``source_frame`` supplies the initial values of frame
    ``source_frame + 1``.  By default only the Gaussian parameters are
    read; the checkpoint's optimizer, scheduler, and density statistics are
    ignored because a new frame resets them.

    ``active_sh_degree`` overrides the degree derived from the checkpoint's
    within-frame iteration count, which matters when frames are trained for
    fewer iterations than the progressive-SH schedule needs.

    ``carry_optimizer_state`` additionally recovers the checkpoint's Adam
    moments, which is what makes ``warm_start.adam_state="carry"`` survive a
    restart of an interrupted sequence instead of silently degrading to a
    reset on the resumed frame.
    """

    state = read_checkpoint(path, map_location="cpu")
    model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
    if active_sh_degree is not None:
        model.set_active_sh_degree(int(active_sh_degree))
    optimizer_state = None
    if carry_optimizer_state:
        saved = state.get("optimizer_state_dict")
        if not isinstance(saved, Mapping):
            raise ValueError(
                f"checkpoint has no optimizer_state_dict to carry: {path}"
            )
        optimizer_state = _adam_state_from_checkpoint(saved)
        if not optimizer_state:
            raise ValueError(
                f"checkpoint optimizer state is empty, so there is nothing to "
                f"carry: {path}"
            )
    handoff = FrameHandoff.from_model(model, source_frame=source_frame)
    if optimizer_state is None:
        return handoff
    return replace(handoff, optimizer_state=optimizer_state)


@dataclass(frozen=True)
class FrameTrainingState:
    """Everything one frame needs, with all per-frame state freshly reset."""

    model: GaussianModel
    optimizer: torch.optim.Adam
    scheduler: PositionLearningRateScheduler | FixedPositionLearningRateScheduler
    density_statistics: ScreenSpaceDensityStatistics | None
    scene_extent: float
    carried_over: bool
    #: What actually happened to the Adam moments -- ``"reset"`` or
    #: ``"carry"`` -- rather than merely what the configuration asked for.
    adam_state: str = "reset"
    #: The absolute position learning rate when the schedule is fixed, else
    #: ``None`` for the iteration-dependent exponential schedule.
    position_learning_rate: float | None = None

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

    ``config.warm_start`` then amends this for carried-over frames only, which
    is what keeps frame 1 byte-identical to a static run:

    * ``position_lr_mode="fixed"`` installs a constant position learning rate
      of ``position_lr_fixed * scene_extent`` in place of the exponential decay
      over ``training.iterations``.
    * ``adam_state="carry"`` installs the hand-off's Adam moments into the
      fresh optimizer.  This requires ``features.adaptive_density_control`` to
      be off, because a change in the Gaussian count would leave the inherited
      moments without a parameter to belong to.
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

    # Warm-start overrides apply only to carried-over frames.  Frame 1 is an
    # ordinary static 3DGS run and must keep the reproduction's behaviour.
    warm_start = config.warm_start
    carried_over = handoff is not None
    fixed_schedule = carried_over and warm_start.position_lr_mode == "fixed"
    position_learning_rate: float | None = None
    if fixed_schedule:
        position_learning_rate = warm_start.position_lr_fixed * scene_extent
        scheduler = FixedPositionLearningRateScheduler(
            optimizer,
            learning_rate=position_learning_rate,
            total_iterations=config.training.iterations,
        )
    else:
        scheduler = PositionLearningRateScheduler(
            optimizer,
            total_iterations=config.training.iterations,
            initial_learning_rate=config.training.position_lr_initial * scene_extent,
            final_learning_rate=config.training.position_lr_final * scene_extent,
        )

    adam_state = "reset"
    if carried_over and warm_start.adam_state == "carry":
        if handoff.optimizer_state is None:
            raise ValueError(
                "warm_start.adam_state='carry' requires a hand-off carrying "
                "Adam moments; pass optimizer= to FrameHandoff.from_model, or "
                "carry_optimizer_state=True to handoff_from_checkpoint"
            )
        if config.features.adaptive_density_control:
            raise ValueError(
                "warm_start.adam_state='carry' requires "
                "features.adaptive_density_control=false: densification and "
                "pruning change the Gaussian count, and the inherited Adam "
                "moments would no longer correspond to any parameter"
            )
        install_adam_state(optimizer, handoff.optimizer_state)
        adam_state = "carry"

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
        carried_over=carried_over,
        adam_state=adam_state,
        position_learning_rate=position_learning_rate,
    )
    assert_frame_state(state)
    return state


def assert_frame_state(state: FrameTrainingState) -> None:
    """Verify that a frame starts from the state its configuration prescribes.

    Everything that must reset on every frame is checked unconditionally: the
    position learning-rate schedule, leftover parameter gradients, and the
    adaptive-density-control statistics.  The Adam moments are checked against
    ``state.adam_state``, so that a ``"carry"`` frame is required to actually
    hold inherited moments and a ``"reset"`` frame is required to hold none.

    Raises:
        RuntimeError: if the frame's starting state contradicts its
            configuration.
    """

    if not isinstance(state, FrameTrainingState):
        raise TypeError("state must be a FrameTrainingState")
    if state.adam_state not in ("reset", "carry"):
        raise RuntimeError(f"unknown adam_state: {state.adam_state!r}")
    moments_present = len(state.optimizer.state) != 0
    if state.adam_state == "reset" and moments_present:
        raise RuntimeError(
            "Adam moments must be empty when a frame starts training with "
            "warm_start.adam_state='reset'"
        )
    if state.adam_state == "carry" and not moments_present:
        raise RuntimeError(
            "warm_start.adam_state='carry' was requested but no Adam moments "
            "were installed"
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


def assert_frame_state_is_reset(state: FrameTrainingState) -> None:
    """Verify that no per-frame optimization state crossed the boundary.

    This is the strict form assumed by the default configuration, and it
    rejects an Adam carry outright.

    Raises:
        RuntimeError: if any Adam moment, position-gradient accumulator,
            observation count, or maximum projection radius is carried over.
    """

    if not isinstance(state, FrameTrainingState):
        raise TypeError("state must be a FrameTrainingState")
    if state.adam_state != "reset" or len(state.optimizer.state) != 0:
        raise RuntimeError(
            "Adam moments must be empty when a frame starts training"
        )
    assert_frame_state(state)


def frame_output_directory(root: str | Path, frame_number: int) -> Path:
    """Return the per-frame run directory below a 4D output root."""

    _validated_frame_number(frame_number, "frame_number")
    return Path(root) / f"frame_{frame_number:04d}"


__all__ = [
    "FrameHandoff",
    "FrameTrainingState",
    "assert_frame_state",
    "assert_frame_state_is_reset",
    "build_frame_training_state",
    "frame_output_directory",
    "handoff_from_checkpoint",
    "install_adam_state",
]
