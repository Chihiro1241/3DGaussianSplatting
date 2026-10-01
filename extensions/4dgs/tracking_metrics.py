"""Long-term point-tracking metrics of Dynamic 3D Gaussians, Table 1.

Luiten et al. (arXiv:2308.09713, Sec. 4) score tracks with the PointOdyssey
metrics: median trajectory error (MTE), position accuracy (delta) and
survival, in normalized pixels for 2D and in centimetres for 3D.  Neither
paper spells the metrics out completely and the Dynamic 3D Gaussians
evaluation code was never released, so the definitions below follow the
PointOdyssey reference evaluation, ``test_on_pod.py`` of PIPs++
(``aharley/pips2``), as it stood when the paper was written (before commit
36441fe of 2023-09-14, which lowered the survival threshold from 50 to 16):

* **Errors.**  2D: ``|| pred / sc - gt / sc ||`` with ``sc = (W / 256,
  H / 256)``, i.e. both axes rescaled to a 256 x 256 image (anisotropically).
  3D: the Euclidean distance, converted to centimetres.

* **delta** (``d_avg``).  The percentage of valid ``(track, frame)`` pairs,
  pooled over all tracks, whose error is strictly below each threshold
  ``{1, 2, 4, 8, 16}``, averaged over the thresholds.  Frame 0 is excluded
  (the query frame, where every tracker is exact).

* **Survival.**  Per track, the fraction of frames before the first valid
  frame whose error exceeds the threshold (50); invalid frames never count
  as failures.  The mean over tracks and frames, in percent.  Frame 0 is
  included.

* **MTE** (``median_l2``).  The median error over each track's valid frames,
  then the mean over tracks.  Frame 0 is included.

A frame is valid when the ground truth is defined there; PIPs++ counts
occluded frames and drops only frames whose point leaves the image (see
:func:`in_image_mask`).  Everything here is plain NumPy so that trajectories
from any method can be scored; producing them is :mod:`point_tracking`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


#: delta thresholds (normalized pixels for 2D, centimetres for 3D).
DELTA_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)

#: Survival threshold of PIPs++ at the time of the paper (normalized pixels).
SURVIVAL_THRESHOLD_2D = 50.0

#: Survival threshold of the paper for 3D ("within 50cm of the ground-truth").
SURVIVAL_THRESHOLD_3D_CM = 50.0

#: Side of the square image PointOdyssey normalizes pixel coordinates to.
NORMALIZED_SIZE = 256.0


@dataclass(frozen=True)
class TrackMetrics:
    """The three metrics of one set of tracks (one scene)."""

    mte: float
    delta_avg: float
    survival: float
    num_tracks: int
    delta: dict[float, float] = field(default_factory=dict)


def errors_3d(predicted: np.ndarray, truth: np.ndarray, unit_to_cm: float = 1.0) -> np.ndarray:
    """Return ``(Q, T)`` 3D errors in centimetres for ``(Q, T, 3)`` tracks."""

    _check_tracks(predicted, truth, 3)
    return np.linalg.norm(predicted - truth, axis=-1) * float(unit_to_cm)


def errors_2d(predicted: np.ndarray, truth: np.ndarray, image_sizes: np.ndarray) -> np.ndarray:
    """Return ``(Q, T)`` errors in normalized pixels for ``(Q, T, 2)`` tracks.

    ``image_sizes`` is ``(width, height)``, either one ``(2,)`` for all tracks
    or ``(Q, 2)`` when the tracks come from cameras of different sizes.
    """

    _check_tracks(predicted, truth, 2)
    scale = np.broadcast_to(np.asarray(image_sizes, dtype=np.float64), (predicted.shape[0], 2))
    scale = (scale / NORMALIZED_SIZE)[:, None, :]
    return np.linalg.norm(predicted / scale - truth / scale, axis=-1)


def in_image_mask(truth: np.ndarray, image_sizes: np.ndarray) -> np.ndarray:
    """PIPs++'s validity: finite and inside ``[1, W-2] x [1, H-2]`` (``(Q, T)``)."""

    sizes = np.broadcast_to(np.asarray(image_sizes, dtype=np.float64), (truth.shape[0], 2))
    width = sizes[:, None, 0]
    height = sizes[:, None, 1]
    x = truth[..., 0]
    y = truth[..., 1]
    finite = np.isfinite(truth).all(axis=-1)
    with np.errstate(invalid="ignore"):
        inside = (x >= 1) & (x <= width - 2) & (y >= 1) & (y <= height - 2)
    return finite & inside


def track_metrics(
    errors: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    thresholds: Sequence[float] = DELTA_THRESHOLDS,
    survival_threshold: float = SURVIVAL_THRESHOLD_2D,
) -> TrackMetrics:
    """Score ``(Q, T)`` errors (frame 0 first) under an optional validity mask.

    Tracks with no valid frame have no median and are left out of the MTE
    (PIPs++ never meets them: its tracks must be valid at frame 0).
    """

    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2:
        raise ValueError(f"errors must have shape (Q, T), got {errors.shape}")
    valid = np.ones_like(errors, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != errors.shape:
        raise ValueError(f"valid must have shape {errors.shape}, got {valid.shape}")
    if errors.shape[0] == 0:
        raise ValueError("there are no tracks to score")
    # An undefined prediction (NaN) is simply wrong at a valid frame.
    errors = np.where(np.isfinite(errors), errors, np.inf)

    later = valid[:, 1:]
    delta = {}
    for threshold in thresholds:
        hits = (errors[:, 1:] < threshold) & later
        delta[float(threshold)] = 100.0 * hits.sum() / max(int(later.sum()), 1)
    delta_avg = float(np.mean(list(delta.values())))

    failed = (errors > survival_threshold) & valid
    survival = 100.0 * float(np.cumprod(~failed, axis=1).mean())

    medians = [np.median(row[mask]) for row, mask in zip(errors, valid) if mask.any()]
    mte = float(np.mean(medians)) if medians else float("nan")
    return TrackMetrics(
        mte=mte, delta_avg=delta_avg, survival=survival,
        num_tracks=int(errors.shape[0]), delta=delta,
    )


def scene_mean(results: Sequence[TrackMetrics]) -> TrackMetrics:
    """Average per-scene results, as the "Mean" column of Table 1."""

    if not results:
        raise ValueError("there are no scenes to average")
    thresholds = results[0].delta.keys()
    return TrackMetrics(
        mte=float(np.mean([r.mte for r in results])),
        delta_avg=float(np.mean([r.delta_avg for r in results])),
        survival=float(np.mean([r.survival for r in results])),
        num_tracks=int(sum(r.num_tracks for r in results)),
        delta={t: float(np.mean([r.delta[t] for r in results])) for t in thresholds},
    )


def _check_tracks(predicted: np.ndarray, truth: np.ndarray, dimension: int) -> None:
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predicted and ground-truth tracks differ in shape: "
            f"{predicted.shape} vs {truth.shape}"
        )
    if predicted.ndim != 3 or predicted.shape[-1] != dimension:
        raise ValueError(f"tracks must have shape (Q, T, {dimension}), got {predicted.shape}")


__all__ = [
    "DELTA_THRESHOLDS",
    "NORMALIZED_SIZE",
    "SURVIVAL_THRESHOLD_2D",
    "SURVIVAL_THRESHOLD_3D_CM",
    "TrackMetrics",
    "errors_2d",
    "errors_3d",
    "in_image_mask",
    "scene_mean",
    "track_metrics",
]
