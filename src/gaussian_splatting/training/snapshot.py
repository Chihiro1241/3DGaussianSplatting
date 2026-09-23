"""Lightweight per-iteration Gaussian snapshots for training visualization.

A training checkpoint stores the optimizer moments, the scheduler, and every
random-number state, which makes it far too large to write at a visualization
cadence.  This module writes the three quantities a learning-progress viewer
actually needs -- centre position, opacity, and Gaussian count -- into one
compressed ``.npz`` per recorded iteration.

The Gaussian count changes at every density-control event, so the recorded
arrays are ragged across iterations and cannot be stacked into a single file.
Each snapshot is therefore its own file below ``<run>/snapshots/``, and
``index.json`` in the same directory lists them for a viewer:

.. code-block:: text

    <run>/snapshots/
      index.json
      iteration_00000000.npz
      iteration_00000500.npz
      ...

A 4D run (``scripts/train_4d.py``) writes one such directory per frame, which gives the
frame-by-iteration grid the viewer animates over.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from gaussian_splatting.model.gaussian_model import GaussianModel


SNAPSHOT_DIRECTORY_NAME = "snapshots"
SNAPSHOT_INDEX_NAME = "index.json"
SNAPSHOT_FORMAT_VERSION = 1

_SNAPSHOT_STEM = "iteration_{iteration:08d}"


def snapshot_filename(iteration: int) -> str:
    """Return the file name a snapshot of ``iteration`` is stored under."""

    if type(iteration) is not int or iteration < 0:
        raise ValueError(f"iteration must be a non-negative integer, got {iteration!r}")
    return _SNAPSHOT_STEM.format(iteration=iteration) + ".npz"


def subsample_indices(count: int, max_points: int | None) -> np.ndarray | None:
    """Return evenly spaced indices thinning ``count`` points to ``max_points``.

    ``None`` means no thinning is required.  The stride is deterministic rather
    than random so that consecutive snapshots keep drawing the same region of
    the parameter array, which stops the animation from flickering between
    unrelated point subsets.  Density control still appends and removes
    Gaussians, so the correspondence is approximate, not exact.
    """

    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    if max_points is None:
        return None
    if type(max_points) is not int or max_points <= 0:
        raise ValueError("max_points must be a positive integer or None")
    if count <= max_points:
        return None
    return np.unique(
        np.linspace(0, count - 1, num=max_points).round().astype(np.int64)
    )


def write_snapshot_npz(
    path: str | Path,
    *,
    means: np.ndarray,
    opacities: np.ndarray,
    iteration: int,
    frame: int,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Write one snapshot and return the index entry describing it.

    ``means`` has shape ``(N, 3)`` and ``opacities`` shape ``(N,)`` or
    ``(N, 1)``; ``opacities`` must already be the transformed sigmoid values,
    not the raw parameters.  ``num_gaussians`` in the returned entry is always
    the true count ``N`` even when the stored arrays were thinned, so a count
    plot stays exact regardless of ``max_points``.  The recorded bounds, in
    contrast, describe only the stored points, because those are what a viewer
    draws and scales its axes to.
    """

    means_array = np.asarray(means, dtype=np.float32)
    opacity_array = np.asarray(opacities, dtype=np.float32).reshape(-1)
    if means_array.ndim != 2 or means_array.shape[1] != 3:
        raise ValueError(f"means must have shape (N, 3), got {means_array.shape}")
    if opacity_array.shape[0] != means_array.shape[0]:
        raise ValueError("means and opacities must describe the same Gaussian count")
    if type(frame) is not int or frame < 1:
        raise ValueError(f"frame must be a positive integer, got {frame!r}")

    count = int(means_array.shape[0])
    selection = subsample_indices(count, max_points)
    if selection is not None:
        means_array = means_array[selection]
        opacity_array = opacity_array[selection]
    stored = int(means_array.shape[0])

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        means=means_array,
        opacities=opacity_array,
        iteration=np.int64(iteration),
        frame=np.int64(frame),
        num_gaussians=np.int64(count),
        stored_points=np.int64(stored),
    )
    finite = np.isfinite(means_array).all(axis=1)
    visible = means_array[finite]
    return {
        "iteration": int(iteration),
        "frame": int(frame),
        "num_gaussians": count,
        "stored_points": stored,
        "file": destination.name,
        "mean_opacity": float(opacity_array.mean()) if stored else 0.0,
        "bounds": (
            [visible.min(axis=0).tolist(), visible.max(axis=0).tolist()]
            if visible.size
            else None
        ),
        "robust_bounds": _percentile_bounds(visible),
    }


ROBUST_BOUNDS_PERCENTILE = 0.5


def _percentile_bounds(points: np.ndarray) -> list[list[float]] | None:
    """Return per-axis bounds that ignore the far-flung tail of the cloud.

    Optimization regularly leaves a handful of Gaussians hundreds of scene
    units away from the reconstruction.  Fixing a viewer's axes to the true
    min/max would then shrink the scene itself to a dot, so the index also
    records the central 99% of each axis.
    """

    if points.size == 0:
        return None
    low = np.percentile(points, ROBUST_BOUNDS_PERCENTILE, axis=0)
    high = np.percentile(points, 100.0 - ROBUST_BOUNDS_PERCENTILE, axis=0)
    return [low.astype(float).tolist(), high.astype(float).tolist()]


def _model_arrays(model: GaussianModel) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        means = model.means_world.detach().to("cpu", torch.float32).numpy()
        opacities = (
            torch.sigmoid(model.raw_opacities.detach())
            .to("cpu", torch.float32)
            .reshape(-1)
            .numpy()
        )
    return means, opacities


def atomic_write_json(path: Path, payload: Any) -> None:
    """Publish ``payload`` as JSON via a temporary file and one rename."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class GaussianSnapshotWriter:
    """Record Gaussian centres, opacities, and counts at a fixed cadence.

    The writer is deliberately passive: it decides nothing about the training
    loop and only answers :meth:`should_record` and stores what it is handed.
    Constructing one has no effect on training, so a run without
    ``--snapshot-interval`` behaves exactly as before.

    Args:
        directory: Snapshot directory, normally ``<run>/snapshots``.
        interval: Record every ``interval`` iterations.  ``0`` disables the
            cadence, leaving only ``extra_iterations``.
        frame: One-based frame number; ``1`` for a static single-frame run.
        max_points: Thin each snapshot to at most this many Gaussians before
            writing.  ``None`` stores every Gaussian.
        extra_iterations: Additional iterations to record regardless of the
            cadence, such as a dense sampling of the early iterations.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        interval: int,
        frame: int = 1,
        max_points: int | None = 20000,
        extra_iterations: Iterable[int] = (),
    ) -> None:
        if type(interval) is not int or interval < 0:
            raise ValueError(f"interval must be a non-negative integer, got {interval!r}")
        if type(frame) is not int or frame < 1:
            raise ValueError(f"frame must be a positive integer, got {frame!r}")
        if max_points is not None and (type(max_points) is not int or max_points <= 0):
            raise ValueError("max_points must be a positive integer or None")
        extra = {int(value) for value in extra_iterations}
        if any(value < 0 for value in extra):
            raise ValueError("extra_iterations must be non-negative")

        self.directory = Path(directory)
        self.interval = interval
        self.frame = frame
        self.max_points = max_points
        self.extra_iterations = frozenset(extra)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, dict[str, Any]] = self._scan_existing()

    def _scan_existing(self) -> dict[int, dict[str, Any]]:
        """Adopt snapshots left by an earlier, resumed run of the same frame."""

        index_path = self.directory / SNAPSHOT_INDEX_NAME
        if not index_path.is_file():
            return {}
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            entries = payload["snapshots"]
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return {}
        adopted: dict[int, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or "iteration" not in entry:
                continue
            if not (self.directory / str(entry.get("file"))).is_file():
                continue
            adopted[int(entry["iteration"])] = entry
        return adopted

    def should_record(self, iteration: int) -> bool:
        """Return whether ``iteration`` falls on the configured cadence."""

        if type(iteration) is not int or iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        if iteration in self.extra_iterations:
            return True
        if self.interval == 0:
            return False
        return iteration % self.interval == 0

    def record(self, model: GaussianModel, iteration: int) -> Path:
        """Write one snapshot of ``model`` and refresh ``index.json``.

        The index is rewritten after every snapshot so that an interrupted or
        crashed run still leaves a directory the viewer can open.
        """

        if type(iteration) is not int or iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        means, opacities = _model_arrays(model)
        destination = self.directory / snapshot_filename(iteration)
        entry = write_snapshot_npz(
            destination,
            means=means,
            opacities=opacities,
            iteration=iteration,
            frame=self.frame,
            max_points=self.max_points,
        )
        self._entries[iteration] = entry
        self.write_index()
        return destination

    def write_index(self) -> Path:
        """Publish the snapshot index for this frame."""

        index_path = self.directory / SNAPSHOT_INDEX_NAME
        atomic_write_json(
            index_path,
            {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "frame": self.frame,
                "interval": self.interval,
                "max_points": self.max_points,
                "snapshots": [
                    self._entries[iteration] for iteration in sorted(self._entries)
                ],
            },
        )
        return index_path

    @property
    def recorded_iterations(self) -> tuple[int, ...]:
        """Return every iteration this writer knows a snapshot for."""

        return tuple(sorted(self._entries))


def read_snapshot(path: str | Path) -> dict[str, Any]:
    """Load one snapshot ``.npz`` into plain Python and NumPy values."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"snapshot does not exist: {source}")
    with np.load(source) as payload:
        return {
            "means": np.asarray(payload["means"], dtype=np.float32),
            "opacities": np.asarray(payload["opacities"], dtype=np.float32),
            "iteration": int(payload["iteration"]),
            "frame": int(payload["frame"]),
            "num_gaussians": int(payload["num_gaussians"]),
            "stored_points": int(payload["stored_points"]),
        }


def discover_snapshot_frames(root: str | Path) -> dict[int, Path]:
    """Map frame number to snapshot directory below a run or 4D run root.

    Accepts either a single-scene run directory (``<run>/snapshots``), a 4D run
    root holding ``frame_XXXX`` subdirectories, or a snapshot directory itself.
    """

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {base}")

    candidates: list[Path] = []
    if (base / SNAPSHOT_INDEX_NAME).is_file():
        candidates = [base]
    elif (base / SNAPSHOT_DIRECTORY_NAME).is_dir():
        candidates = [base / SNAPSHOT_DIRECTORY_NAME]
    else:
        candidates = sorted(
            directory / SNAPSHOT_DIRECTORY_NAME
            for directory in base.glob("frame_*")
            if (directory / SNAPSHOT_DIRECTORY_NAME).is_dir()
        )

    frames: dict[int, Path] = {}
    for directory in candidates:
        index_path = directory / SNAPSHOT_INDEX_NAME
        if not index_path.is_file():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        frames[int(payload.get("frame", 1))] = directory
    if not frames:
        raise FileNotFoundError(
            f"no snapshot index below {base}; train with --snapshot-interval "
            "or run eval/extract_snapshots.py on an existing run"
        )
    return dict(sorted(frames.items()))


def read_snapshot_index(directory: str | Path) -> list[dict[str, Any]]:
    """Return the snapshot entries of one frame, ordered by iteration."""

    index_path = Path(directory) / SNAPSHOT_INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"snapshot index does not exist: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [dict(entry) for entry in payload["snapshots"]]
    entries.sort(key=lambda entry: int(entry["iteration"]))
    return entries


def parse_iteration_list(text: str | None) -> tuple[int, ...]:
    """Parse a ``"0,100,500"`` command-line iteration list."""

    if not text:
        return ()
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("iteration list entries must be non-negative")
        values.append(value)
    return tuple(sorted(set(values)))


def combined_bounds(
    entries: Sequence[dict[str, Any]],
    *,
    margin: float = 0.05,
    key: str = "bounds",
) -> tuple[list[float], list[float]] | None:
    """Return the axis-aligned box covering every entry's recorded bounds.

    A viewer must fix the axis ranges across an animation: letting Plotly
    autoscale each animation frame makes the scene jump around instead of
    showing the Gaussians move.  ``key`` selects ``"bounds"`` (the full
    extent of the stored points) or ``"robust_bounds"`` (their central 99%,
    which keeps stray Gaussians from shrinking the scene to a dot).
    """

    lower = [math.inf] * 3
    upper = [-math.inf] * 3
    for entry in entries:
        bounds = entry.get(key) or entry.get("bounds")
        if not bounds:
            continue
        for axis in range(3):
            lower[axis] = min(lower[axis], float(bounds[0][axis]))
            upper[axis] = max(upper[axis], float(bounds[1][axis]))
    if any(math.isinf(value) for value in lower + upper):
        return None
    for axis in range(3):
        span = upper[axis] - lower[axis]
        pad = (span if span > 0 else 1.0) * margin
        lower[axis] -= pad
        upper[axis] += pad
    return lower, upper


__all__ = [
    "GaussianSnapshotWriter",
    "ROBUST_BOUNDS_PERCENTILE",
    "atomic_write_json",
    "SNAPSHOT_DIRECTORY_NAME",
    "SNAPSHOT_FORMAT_VERSION",
    "SNAPSHOT_INDEX_NAME",
    "combined_bounds",
    "discover_snapshot_frames",
    "parse_iteration_list",
    "read_snapshot",
    "read_snapshot_index",
    "snapshot_filename",
    "subsample_indices",
    "write_snapshot_npz",
]
