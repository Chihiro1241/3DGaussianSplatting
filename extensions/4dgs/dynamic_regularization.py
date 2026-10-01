"""Physically based motion priors of Dynamic 3D Gaussians.

This module adds the regularization of Luiten et al., "Dynamic 3D Gaussians:
Tracking by Persistent Dynamic View Synthesis" (3DV 2024) on top of the plain
frame-to-frame hand-off in :mod:`trainer_4d`.  The hand-off alone only gives a
frame a good starting point; nothing stops the optimizer from then moving each
Gaussian independently.  The priors below tie every Gaussian's motion to that
of its neighbours, following the reference implementation
(``JonathonLuiten/Dynamic3DGaussians``, ``train.py``):

* **Neighbour graph.**  Built once, from the reference frame (the first frame
  of the sequence), as the ``k`` nearest Gaussian centres of every Gaussian
  with weights ``w_ij = exp(-lambda_w * ||mu_j - mu_i||^2)``.  It is reused
  unchanged for every later frame, which is why the Gaussian count must stay
  fixed once the graph exists.

* **Local rigidity.**  The neighbourhood of Gaussian ``i`` should move like a
  rigid body carried by ``i``'s own rotation::

      L_rigid = mean_ij sqrt(w_ij * || (mu_j,t-1 - mu_i,t-1)
                                      - R_i,t-1 R_i,t^-1 (mu_j,t - mu_i,t) ||^2)

* **Rotation similarity.**  Neighbours should rotate by the same amount::

      L_rot = mean_ij sqrt(w_ij * || q_j,t q_j,t-1^-1 - q_i,t q_i,t-1^-1 ||^2)

* **Long-term isometry.**  Neighbour distances should stay what they were in
  the reference frame::

      L_iso = mean_ij sqrt(w_ij * (||mu_j,t - mu_i,t|| - ||mu_j,0 - mu_i,0||)^2)

* **Soft colour consistency.**  The DC colour should not drift from the
  previous frame: ``L_color = mean_i ||c_i,t - c_i,t-1||_1``.

The weight enters inside the square root, exactly as the reference code's
``weighted_l2_loss_v1/v2`` (the paper writes it outside); ``1e-20`` keeps the
gradient of the square root finite at zero.

Two further measures of the reference implementation are available:

* **Velocity initialization.**  Frame ``t`` starts from ``mu_t-1 + (mu_t-1 -
  mu_t-2)`` and the normalized ``2 q_t-1 - q_t-2``, rather than from the
  previous frame's values.
* **Frozen opacity and scale.**  After the reference frame, opacity and scale
  are held fixed (zero Adam learning rate), so that only position, rotation,
  and colour describe the motion.

What is deliberately not reproduced: the foreground/background split of the
reference (and with it the background-anchoring and floor losses, and the
segmentation rendering) needs per-pixel segmentation masks that the datasets
used here do not have, so the priors apply to every Gaussian; and the
per-camera colour correction parameters are not part of this model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from gaussian_splatting.config import Config, DynamicRegularizationConfig
from gaussian_splatting.math.covariance import quaternion_rotation_matrix
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.training.losses import RegularizationLoss

from trainer_4d import FrameHandoff


#: Keeps ``sqrt`` differentiable where a residual is exactly zero.
_SQRT_EPSILON = 1e-20

#: Parameter groups frozen by ``freeze_opacity_and_scale``.
FROZEN_PARAMETER_GROUPS = ("raw_opacities", "raw_scales")

#: Directory, under the 4D output root, holding the persisted state.
STATE_DIRECTORY_NAME = "dynamic_regularization"
_GRAPH_FILE_NAME = "neighbor_graph.pt"
_MOTION_FILE_NAME = "motion_origin.pt"
_STATE_FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# quaternion helpers
# --------------------------------------------------------------------------


def quaternion_multiply(first: Tensor, second: Tensor) -> Tensor:
    """Return the Hamilton product of two ``(..., 4)`` ``(w, x, y, z)`` tensors."""

    w1, x1, y1, z1 = first.unbind(dim=-1)
    w2, x2, y2, z2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quaternion_conjugate(quaternions: Tensor) -> Tensor:
    """Return the conjugate, which is the inverse of a unit quaternion."""

    return torch.cat((quaternions[..., :1], -quaternions[..., 1:]), dim=-1)


def unit_quaternions(raw_quaternions: Tensor, epsilon_q: float) -> Tensor:
    """Normalize raw quaternions exactly as the renderer does."""

    norms = torch.linalg.vector_norm(raw_quaternions, dim=-1, keepdim=True)
    return raw_quaternions / norms.clamp_min(epsilon_q)


# --------------------------------------------------------------------------
# neighbour graph
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NeighborGraph:
    """The fixed ``k``-nearest-neighbour graph of the reference frame.

    ``indices[i]`` lists the ``k`` neighbours of Gaussian ``i`` (never ``i``
    itself), ``distances`` their reference-frame distances, and ``weights``
    the corresponding ``exp(-lambda_w d^2)``.
    """

    indices: Tensor
    weights: Tensor
    distances: Tensor
    reference_frame: int

    def __post_init__(self) -> None:
        if self.indices.ndim != 2 or self.indices.dtype != torch.long:
            raise ValueError("neighbour indices must be a (N, k) int64 tensor")
        if self.weights.shape != self.indices.shape:
            raise ValueError("neighbour weights must match the index shape")
        if self.distances.shape != self.indices.shape:
            raise ValueError("neighbour distances must match the index shape")
        if type(self.reference_frame) is not int or self.reference_frame < 1:
            raise ValueError("reference_frame must be a positive integer")

    @property
    def num_gaussians(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_neighbors(self) -> int:
        return int(self.indices.shape[1])

    def to(self, *, device: str | torch.device, dtype: torch.dtype) -> NeighborGraph:
        return replace(
            self,
            indices=self.indices.to(device=device),
            weights=self.weights.to(device=device, dtype=dtype),
            distances=self.distances.to(device=device, dtype=dtype),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _STATE_FORMAT_VERSION,
            "indices": self.indices.detach().cpu(),
            "weights": self.weights.detach().cpu(),
            "distances": self.distances.detach().cpu(),
            "reference_frame": self.reference_frame,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> NeighborGraph:
        _check_format(state, "neighbour graph")
        return cls(
            indices=state["indices"],
            weights=state["weights"],
            distances=state["distances"],
            reference_frame=int(state["reference_frame"]),
        )


def build_neighbor_graph(
    means: Tensor,
    *,
    num_neighbors: int,
    weight_lambda: float,
    reference_frame: int,
) -> NeighborGraph:
    """Find every Gaussian's ``num_neighbors`` nearest centres.

    The search runs on the CPU in float64 through a k-d tree, which is what the
    reference implementation does with Open3D.  Each point is excluded from
    its own neighbourhood by index rather than by position in the result, so
    that coincident centres cannot make a Gaussian its own neighbour.
    """

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    count = int(means.shape[0])
    if type(num_neighbors) is not int or num_neighbors < 1:
        raise ValueError("num_neighbors must be a positive integer")
    if count <= num_neighbors:
        raise ValueError(
            f"a neighbour graph with k={num_neighbors} needs more than "
            f"{num_neighbors} Gaussians, got {count}"
        )
    points = means.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(points).all():
        raise ValueError("means must be finite to build a neighbour graph")
    distances, indices = cKDTree(points).query(points, k=num_neighbors + 1, workers=-1)
    own = np.arange(count)[:, None]
    keep = indices != own
    # A row whose own index is absent (more coincident points than k + 1)
    # simply drops its farthest candidate instead.
    overfull = keep.sum(axis=1) > num_neighbors
    keep[overfull, -1] = False
    indices = indices[keep].reshape(count, num_neighbors)
    distances = distances[keep].reshape(count, num_neighbors)
    weights = np.exp(-float(weight_lambda) * np.square(distances))
    return NeighborGraph(
        indices=torch.from_numpy(np.ascontiguousarray(indices)).to(torch.long),
        weights=torch.from_numpy(np.ascontiguousarray(weights)),
        distances=torch.from_numpy(np.ascontiguousarray(distances)),
        reference_frame=int(reference_frame),
    )


# --------------------------------------------------------------------------
# previous-frame motion state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameMotionState:
    """The optimized centres, unit rotations, and DC colours of one frame."""

    means: Tensor
    quaternions: Tensor
    sh_dc: Tensor
    frame: int

    def __post_init__(self) -> None:
        count = self.means.shape[0]
        if self.means.shape != (count, 3):
            raise ValueError("means must have shape (N, 3)")
        if self.quaternions.shape != (count, 4):
            raise ValueError("quaternions must have shape (N, 4)")
        if self.sh_dc.shape[0] != count:
            raise ValueError("sh_dc must hold one entry per Gaussian")
        if type(self.frame) is not int or self.frame < 1:
            raise ValueError("frame must be a positive integer")

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def from_handoff(cls, handoff: FrameHandoff, *, epsilon_q: float) -> FrameMotionState:
        parameters = handoff.parameters
        return cls(
            means=parameters["means_world"].detach().clone(),
            quaternions=unit_quaternions(
                parameters["raw_quaternions"].detach(), epsilon_q
            ),
            sh_dc=parameters["sh_dc"].detach().clone(),
            frame=int(handoff.source_frame),
        )

    def to(self, *, device: str | torch.device, dtype: torch.dtype) -> FrameMotionState:
        return replace(
            self,
            means=self.means.to(device=device, dtype=dtype),
            quaternions=self.quaternions.to(device=device, dtype=dtype),
            sh_dc=self.sh_dc.to(device=device, dtype=dtype),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _STATE_FORMAT_VERSION,
            "means": self.means.detach().cpu(),
            "quaternions": self.quaternions.detach().cpu(),
            "sh_dc": self.sh_dc.detach().cpu(),
            "frame": self.frame,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> FrameMotionState:
        _check_format(state, "motion origin")
        return cls(
            means=state["means"],
            quaternions=state["quaternions"],
            sh_dc=state["sh_dc"],
            frame=int(state["frame"]),
        )


def extrapolate_handoff(
    handoff: FrameHandoff,
    origin: FrameMotionState | None,
    *,
    epsilon_q: float,
) -> FrameHandoff:
    """Advance a hand-off by the motion it made over its own frame.

    With ``origin`` the state of frame ``t - 2`` and ``handoff`` that of frame
    ``t - 1``, the returned hand-off starts frame ``t`` at
    ``mu + (mu - mu_origin)`` and ``normalize(q + (q - q_origin))``, the
    reference implementation's constant-velocity guess.  Without an origin
    (the first frame after the reference) there is no velocity yet and the
    hand-off is returned unchanged.
    """

    if origin is None:
        return handoff
    if origin.num_gaussians != handoff.num_gaussians:
        raise ValueError(
            f"motion origin has {origin.num_gaussians} Gaussians but the "
            f"hand-off has {handoff.num_gaussians}; velocity initialization "
            "requires a fixed Gaussian count"
        )
    if origin.frame != handoff.source_frame - 1:
        raise ValueError(
            f"motion origin is frame {origin.frame}, but a hand-off from frame "
            f"{handoff.source_frame} needs frame {handoff.source_frame - 1}"
        )
    parameters = dict(handoff.parameters)
    means = parameters["means_world"]
    quaternions = unit_quaternions(parameters["raw_quaternions"], epsilon_q)
    origin_means = origin.means.to(device=means.device, dtype=means.dtype)
    origin_quaternions = origin.quaternions.to(
        device=quaternions.device, dtype=quaternions.dtype
    )
    parameters["means_world"] = means + (means - origin_means)
    parameters["raw_quaternions"] = unit_quaternions(
        quaternions + (quaternions - origin_quaternions), epsilon_q
    )
    # The Adam moments, if any, stay valid: the count and shapes are unchanged.
    return replace(handoff, parameters=parameters)


# --------------------------------------------------------------------------
# the regularizer
# --------------------------------------------------------------------------


def _weighted_l2(residual: Tensor, weights: Tensor) -> Tensor:
    """``weighted_l2_loss_v2``: the residual's last axis is a vector."""

    return torch.sqrt(residual.square().sum(dim=-1) * weights + _SQRT_EPSILON).mean()


def _weighted_scalar_l2(residual: Tensor, weights: Tensor) -> Tensor:
    """``weighted_l2_loss_v1``: the residual is a scalar per neighbour pair."""

    return torch.sqrt(residual.square() * weights + _SQRT_EPSILON).mean()


class DynamicRegularizer:
    """Evaluate the Dynamic 3D Gaussians priors for one carried-over frame.

    ``previous`` is the optimized state of frame ``t - 1`` and stays fixed for
    the whole of frame ``t``.  Instances are callables suitable for
    :class:`gaussian_splatting.training.trainer.Trainer`'s ``regularizer``.
    """

    def __init__(
        self,
        graph: NeighborGraph,
        previous: FrameMotionState,
        config: DynamicRegularizationConfig,
    ) -> None:
        if not isinstance(config, DynamicRegularizationConfig):
            raise TypeError("config must be a DynamicRegularizationConfig")
        if graph.num_gaussians != previous.num_gaussians:
            raise ValueError(
                f"neighbour graph has {graph.num_gaussians} Gaussians but the "
                f"previous frame has {previous.num_gaussians}; the priors "
                "require a fixed Gaussian count"
            )
        self.graph = graph
        self.previous = previous
        self.config = config
        self._weights = {
            "rigid": config.lambda_rigid,
            "rotation": config.lambda_rotation,
            "isometry": config.lambda_isometry,
            "color": config.lambda_color,
        }
        with torch.no_grad():
            # (mu_j,t-1 - mu_i,t-1) and q_i,t-1^-1 do not change within a frame.
            self._previous_offsets = (
                previous.means[graph.indices] - previous.means[:, None, :]
            )
            self._previous_inverse_rotations = quaternion_conjugate(
                previous.quaternions
            )

    @property
    def num_gaussians(self) -> int:
        return self.graph.num_gaussians

    def __call__(self, model: GaussianModel) -> RegularizationLoss:
        if model.num_gaussians != self.num_gaussians:
            raise RuntimeError(
                f"the model has {model.num_gaussians} Gaussians but the "
                f"neighbour graph was built for {self.num_gaussians}; "
                "densification or pruning must stay off while regularizing"
            )
        graph = self.graph
        indices = graph.indices
        means = model.means_world
        terms: dict[str, Tensor] = {}

        needs_offsets = self._weights["rigid"] > 0.0 or self._weights["isometry"] > 0.0
        needs_rotation = self._weights["rigid"] > 0.0 or self._weights["rotation"] > 0.0
        offsets = means[indices] - means[:, None, :] if needs_offsets else None
        relative_rotations = None
        if needs_rotation:
            quaternions = unit_quaternions(model.raw_quaternions, model.epsilon_q)
            relative_rotations = quaternion_multiply(
                quaternions, self._previous_inverse_rotations
            )

        if self._weights["rigid"] > 0.0:
            # R_rel = R_t R_t-1^-1, so R_t-1 R_t^-1 = R_rel^T.
            rotation = quaternion_rotation_matrix(relative_rotations)
            offsets_in_previous_frame = torch.einsum(
                "nba,nkb->nka", rotation, offsets
            )
            terms["rigid"] = _weighted_l2(
                offsets_in_previous_frame - self._previous_offsets, graph.weights
            )
        if self._weights["rotation"] > 0.0:
            terms["rotation"] = _weighted_l2(
                relative_rotations[indices] - relative_rotations[:, None, :],
                graph.weights,
            )
        if self._weights["isometry"] > 0.0:
            lengths = torch.sqrt(offsets.square().sum(dim=-1) + _SQRT_EPSILON)
            terms["isometry"] = _weighted_scalar_l2(
                lengths - graph.distances, graph.weights
            )
        if self._weights["color"] > 0.0:
            difference = model.sh_dc - self.previous.sh_dc
            terms["color"] = difference.abs().reshape(difference.shape[0], -1).sum(-1).mean()

        total = means.new_zeros(())
        for name, value in terms.items():
            total = total + self._weights[name] * value
        return RegularizationLoss(total=total, terms=terms)


# --------------------------------------------------------------------------
# optimizer and configuration
# --------------------------------------------------------------------------


def freeze_parameter_groups(
    optimizer: torch.optim.Optimizer, names: Iterable[str] = FROZEN_PARAMETER_GROUPS
) -> list[str]:
    """Give the named Adam groups a zero learning rate; return what was frozen.

    A zero rate is what the reference implementation uses.  It leaves the
    group in the optimizer, so checkpoints keep their usual layout.
    """

    wanted = set(names)
    frozen: list[str] = []
    for group in optimizer.param_groups:
        if group.get("name") in wanted:
            group["lr"] = 0.0
            frozen.append(group["name"])
    missing = sorted(wanted - set(frozen))
    if missing:
        raise ValueError(f"optimizer has no parameter group named {missing}")
    return frozen


def validate_regularization_config(config: Config) -> None:
    """Reject configurations under which the priors cannot hold.

    A 4D run shares one configuration across its frames, so the constraints on
    carried-over frames apply to the configuration as a whole.
    """

    if not config.dynamic_regularization.enabled:
        return
    if config.features.adaptive_density_control:
        raise ValueError(
            "dynamic_regularization requires "
            "features.adaptive_density_control=false: the neighbour graph "
            "fixes the Gaussian set.  Train frame 1 with density control "
            "separately and start this run from it with --start-frame 2 "
            "--carry-over-checkpoint"
        )
    if config.features.opacity_reset:
        raise ValueError(
            "dynamic_regularization requires features.opacity_reset=false: a "
            "reset would overwrite the opacities of the reference frame"
        )


# --------------------------------------------------------------------------
# persistence across restarts
# --------------------------------------------------------------------------


def _check_format(state: Mapping[str, object], label: str) -> None:
    version = state.get("format_version")
    if version != _STATE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported {label} format version {version!r}; expected "
            f"{_STATE_FORMAT_VERSION}"
        )


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def save_neighbor_graph(directory: Path, graph: NeighborGraph) -> Path:
    """Write the graph once per run; it never changes after the reference frame."""

    path = directory / _GRAPH_FILE_NAME
    _atomic_save(graph.state_dict(), path)
    return path


def load_neighbor_graph(directory: Path) -> NeighborGraph | None:
    path = directory / _GRAPH_FILE_NAME
    if not path.is_file():
        return None
    return NeighborGraph.from_state_dict(torch.load(path, map_location="cpu"))


def save_motion_origin(
    directory: Path, origin: FrameMotionState | None, *, next_frame: int
) -> Path:
    """Record the velocity origin that frame ``next_frame`` will need.

    Only one origin is ever kept, overwritten after every frame: restarting at
    ``next_frame`` needs nothing older, and a per-frame copy would cost as much
    disk as another set of Gaussians per frame.
    """

    path = directory / _MOTION_FILE_NAME
    _atomic_save(
        {
            "format_version": _STATE_FORMAT_VERSION,
            "next_frame": int(next_frame),
            "origin": None if origin is None else origin.state_dict(),
        },
        path,
    )
    return path


def load_motion_origin(
    directory: Path, *, next_frame: int
) -> tuple[bool, FrameMotionState | None]:
    """Return ``(found, origin)`` for a restart at ``next_frame``.

    A file recorded for a different frame is refused: using it would give the
    restarted frame a velocity measured between the wrong pair of frames.
    """

    path = directory / _MOTION_FILE_NAME
    if not path.is_file():
        return False, None
    state = torch.load(path, map_location="cpu")
    _check_format(state, "motion origin")
    recorded = int(state["next_frame"])
    if recorded != next_frame:
        raise ValueError(
            f"{path} holds the velocity origin for frame {recorded}, not for "
            f"frame {next_frame}.  Restart at frame {recorded}, or point "
            "--regularization-state at a state directory for this frame"
        )
    origin = state["origin"]
    return True, None if origin is None else FrameMotionState.from_state_dict(origin)


__all__ = [
    "DynamicRegularizer",
    "FROZEN_PARAMETER_GROUPS",
    "FrameMotionState",
    "NeighborGraph",
    "STATE_DIRECTORY_NAME",
    "build_neighbor_graph",
    "extrapolate_handoff",
    "freeze_parameter_groups",
    "load_motion_origin",
    "load_neighbor_graph",
    "quaternion_conjugate",
    "quaternion_multiply",
    "save_motion_origin",
    "save_neighbor_graph",
    "unit_quaternions",
    "validate_regularization_config",
]
