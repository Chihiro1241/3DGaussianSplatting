"""Evaluate rendered images against ground-truth images already on disk.

Unlike :func:`gaussian_splatting.evaluation.runner.evaluate_camera_set`, this
path needs no checkpoint: it pairs two directories of images and scores each
pair with the same metric implementations, so both paths report identical
numbers for identical images.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.evaluation.metrics import LPIPSMetric, ms_ssim, psnr
from gaussian_splatting.training.losses import dssim_loss, ssim

# Metric sets follow Wu et al. (2024) for the 4D benchmarks and the 3DGS
# paper for the static datasets present in this repository.
DATASET_METRICS: dict[str, tuple[str, ...]] = {
    "dnerf": ("psnr", "ssim", "lpips"),
    "hypernerf": ("psnr", "ms_ssim"),
    "neu3d": ("psnr", "d_ssim", "lpips"),
    "nerf_synthetic": ("psnr", "ssim", "lpips"),
    "colmap": ("psnr", "ssim", "lpips"),
}

SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

# NeRF Synthetic keeps depth and normal maps next to the RGB ground truth.
GT_EXCLUDE_SUBSTRINGS = ("_depth_", "_normal_")

# More groups than this means file stems are frame names, not camera names.
MAX_CAMERA_GROUPS = 32

AVERAGE_ROW_LABEL = "*** AVERAGE ***"


@dataclass(frozen=True)
class ImagePairs:
    pairs: list[tuple[Path, Path]]
    missing: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)


def _is_gt_rgb(path: Path) -> bool:
    return not any(token in path.name for token in GT_EXCLUDE_SUBSTRINGS)


def collect_image_pairs(render_dir: Path, gt_dir: Path) -> ImagePairs:
    """Match each rendered image to one ground-truth image.

    A ground truth at the same relative path wins.  Otherwise the ground truth
    is looked up by file stem, because ``render_camera_set`` writes flat base
    names while datasets such as NeRF Synthetic nest them (``test/r_0.png``).
    Stems shared by several ground truths are skipped as ambiguous.
    """

    if not render_dir.is_dir():
        raise FileNotFoundError(f"render directory does not exist: {render_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"ground-truth directory does not exist: {gt_dir}")

    render_files = sorted(
        path
        for path in render_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    gt_by_stem: dict[str, list[Path]] = {}
    for path in gt_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and _is_gt_rgb(path):
            gt_by_stem.setdefault(path.stem, []).append(path)

    result = ImagePairs(pairs=[])
    for rendered in render_files:
        relative = rendered.relative_to(render_dir)
        direct = gt_dir / relative
        if direct.is_file():
            result.pairs.append((rendered, direct))
            continue
        candidates = gt_by_stem.get(rendered.stem, [])
        if len(candidates) == 1:
            result.pairs.append((rendered, candidates[0]))
        elif candidates:
            result.ambiguous.append(str(relative))
        else:
            result.missing.append(str(relative))
    return result


def load_rgb(path: Path, *, background: str = "white") -> np.ndarray:
    """Return an ``(H,W,3)`` uint8 image, compositing RGBA onto ``background``.

    ``convert("RGB")`` alone would drop alpha and turn the transparent
    background of D-NeRF / NeRF Synthetic ground truth black.  This follows
    the same rule as ``gaussian_splatting.data.dataset._image_to_tensor``.
    """

    image = Image.open(path)
    if image.mode == "RGBA":
        rgba = np.asarray(image, dtype=np.float32)
        alpha = rgba[..., 3:4] / 255.0
        fill = 0.0 if background == "black" else 255.0
        rgb = rgba[..., :3] * alpha + fill * (1.0 - alpha)
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _to_tensor(pixels: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(pixels).to(device).permute(2, 0, 1).float().div(255.0)


def frame_index(render_dir: Path, path: Path) -> int:
    """Return 42 for ``renders/frame_0042/cam00.png``, or -1 without a frame."""

    try:
        relative = path.relative_to(render_dir)
    except ValueError:
        return -1
    for part in relative.parts:
        match = re.fullmatch(r"frame_(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


class ImageMetrics:
    """Score one image pair with the metric set of a dataset."""

    def __init__(
        self,
        dataset: str,
        *,
        device: torch.device | str,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        ssim_k1: float = 0.01,
        ssim_k2: float = 0.03,
        lpips_metric: LPIPSMetric | None = None,
    ) -> None:
        if dataset not in DATASET_METRICS:
            raise ValueError(f"dataset must be one of {sorted(DATASET_METRICS)}")
        self.names = DATASET_METRICS[dataset]
        self.device = device
        self._ssim_options = {
            "window_size": ssim_window_size,
            "sigma": ssim_sigma,
            "k1": ssim_k1,
            "k2": ssim_k2,
        }
        if "lpips" in self.names and lpips_metric is None:
            lpips_metric = LPIPSMetric(device=device)
        self._lpips = lpips_metric

    def __call__(self, rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        values: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            if "psnr" in self.names:
                values["psnr"] = psnr(rendered, target)
            if "ssim" in self.names:
                values["ssim"] = ssim(rendered, target, **self._ssim_options)
            if "d_ssim" in self.names:
                values["d_ssim"] = dssim_loss(rendered, target, **self._ssim_options)
            if "ms_ssim" in self.names:
                values["ms_ssim"] = ms_ssim(rendered, target)
            if "lpips" in self.names:
                assert self._lpips is not None
                values["lpips"] = self._lpips(rendered, target)
        return {name: float(value.item()) for name, value in values.items()}

    def score_files(
        self,
        rendered_path: Path,
        target_path: Path,
        *,
        background: str = "white",
    ) -> dict[str, float]:
        """Load, resize the render to the ground truth if needed, and score."""

        rendered = load_rgb(rendered_path, background=background)
        target = load_rgb(target_path, background=background)
        if rendered.shape != target.shape:
            height, width = target.shape[:2]
            rendered = np.array(
                Image.fromarray(rendered).resize((width, height), Image.LANCZOS)
            )
        return self(_to_tensor(rendered, self.device), _to_tensor(target, self.device))


def evaluate_image_pairs(
    pairs: list[tuple[Path, Path]],
    metrics: ImageMetrics,
    *,
    render_dir: Path,
    background: str = "white",
    progress_every: int = 0,
) -> list[dict[str, object]]:
    """Return one row per pair with ``filename``, ``frame``, ``camera`` and metrics."""

    rows: list[dict[str, object]] = []
    for index, (rendered_path, target_path) in enumerate(pairs, start=1):
        values = metrics.score_files(rendered_path, target_path, background=background)
        rows.append(
            {
                "filename": rendered_path.name,
                "frame": frame_index(render_dir, rendered_path),
                "camera": rendered_path.stem,
                **values,
            }
        )
        if progress_every and (index % progress_every == 0 or index == len(pairs)):
            print(f"  {index}/{len(pairs)}")
    return rows


def finite_mean(values: list[float]) -> float:
    """Average finite values only; PSNR is ``+inf`` for an exact match."""

    kept = [value for value in values if math.isfinite(value)]
    return float(np.mean(kept)) if kept else float("nan")


def metric_averages(
    rows: list[dict[str, object]], names: tuple[str, ...]
) -> dict[str, float]:
    return {name: finite_mean([float(row[name]) for row in rows]) for name in names}


def camera_summary(
    rows: list[dict[str, object]], names: tuple[str, ...]
) -> dict[str, object] | None:
    """Group rows by camera and return mean/std/min/max per metric.

    Neu3D renders are named after cameras (``cam00.png``), so every frame
    shares the same few stems.  D-NeRF renders are named after frames
    (``r_000.png``), which yields one image per group; ``None`` is returned
    then because a per-camera table would be meaningless.
    """

    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["camera"]), []).append(row)
    if len(groups) > MAX_CAMERA_GROUPS or all(len(group) < 2 for group in groups.values()):
        return None

    def stats(subset: list[dict[str, object]]) -> dict[str, object]:
        entry: dict[str, object] = {"count": len(subset)}
        for name in names:
            values = [float(row[name]) for row in subset if math.isfinite(float(row[name]))]
            if values:
                entry[name] = {
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
        return entry

    return {
        "cameras": {name: stats(group) for name, group in sorted(groups.items())},
        "overall": stats(rows),
    }


def frame_blocks(
    rows: list[dict[str, object]], names: tuple[str, ...], block: int
) -> list[tuple[str, int, dict[str, float]]]:
    """Average rows over consecutive 1-based frame blocks of width ``block``."""

    numbered = [row for row in rows if int(row["frame"]) > 0]
    if not numbered or block <= 0:
        return []
    lowest = min(int(row["frame"]) for row in numbered)
    highest = max(int(row["frame"]) for row in numbered)
    blocks: list[tuple[str, int, dict[str, float]]] = []
    start = lowest - ((lowest - 1) % block)
    while start <= highest:
        stop = start + block - 1
        subset = [row for row in numbered if start <= int(row["frame"]) <= stop]
        if subset:
            blocks.append((f"{start}-{stop}", len(subset), metric_averages(subset, names)))
        start += block
    return blocks


def write_metrics_csv(
    rows: list[dict[str, object]], names: tuple[str, ...], path: Path
) -> None:
    """Write one row per image followed by an average row (``metrics.csv``)."""

    averages = metric_averages(rows, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", *names], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {"filename": AVERAGE_ROW_LABEL, **{name: f"{averages[name]:.4f}" for name in names}}
        )


def write_per_frame_csv(
    rows: list[dict[str, object]], names: tuple[str, ...], path: Path
) -> None:
    """Write ``frame,camera,<metrics>`` rows (``per_frame.csv``)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "camera", *names], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_camera_summary_json(
    summary: dict[str, object], path: Path, *, metrics_csv: Path | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"csv": str(metrics_csv) if metrics_csv else None, **summary}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "AVERAGE_ROW_LABEL",
    "DATASET_METRICS",
    "ImageMetrics",
    "ImagePairs",
    "camera_summary",
    "collect_image_pairs",
    "evaluate_image_pairs",
    "finite_mean",
    "frame_blocks",
    "frame_index",
    "load_rgb",
    "metric_averages",
    "write_camera_summary_json",
    "write_metrics_csv",
    "write_per_frame_csv",
]
