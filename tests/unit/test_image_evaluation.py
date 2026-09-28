from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gaussian_splatting.evaluation import images
from gaussian_splatting.evaluation.metrics import LPIPSMetric, psnr
from gaussian_splatting.training.losses import dssim_loss


class _MeanSquaredPerceptualDistance(torch.nn.Module):
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (rendered - target).square().mean(dim=(1, 2, 3), keepdim=True)


def _lpips_stub() -> LPIPSMetric:
    return LPIPSMetric(device="cpu", network=_MeanSquaredPerceptualDistance())


def _save(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _neu3d_tree(root: Path) -> tuple[Path, Path]:
    """Two frames x two cameras, with renders offset from the ground truth."""

    generator = np.random.default_rng(0)
    for frame in (1, 2):
        for camera in ("cam00", "cam09"):
            truth = generator.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
            rendered = np.clip(truth.astype(np.int16) + 8 * frame, 0, 255).astype(np.uint8)
            _save(root / "gt" / f"frame_{frame:04d}" / f"{camera}.png", truth)
            _save(root / "renders" / f"frame_{frame:04d}" / f"{camera}.png", rendered)
    return root / "renders", root / "gt"


def test_collect_image_pairs_prefers_relative_path_then_unique_stem(tmp_path: Path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    _save(tmp_path / "renders" / "r_0.png", pixels)
    _save(tmp_path / "renders" / "r_1.png", pixels)
    _save(tmp_path / "renders" / "r_2.png", pixels)
    _save(tmp_path / "gt" / "test" / "r_0.png", pixels)
    _save(tmp_path / "gt" / "test" / "r_0_depth_0000.png", pixels)
    _save(tmp_path / "gt" / "a" / "r_1.png", pixels)
    _save(tmp_path / "gt" / "b" / "r_1.png", pixels)

    found = images.collect_image_pairs(tmp_path / "renders", tmp_path / "gt")

    assert found.pairs == [(tmp_path / "renders" / "r_0.png", tmp_path / "gt" / "test" / "r_0.png")]
    assert found.ambiguous == ["r_1.png"]
    assert found.missing == ["r_2.png"]


def test_load_rgb_composites_alpha_onto_background(tmp_path: Path) -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    path = tmp_path / "transparent.png"
    Image.fromarray(rgba, mode="RGBA").save(path)

    assert images.load_rgb(path, background="white").min() == 255
    assert images.load_rgb(path, background="black").max() == 0


def test_frame_index_reads_frame_directory(tmp_path: Path) -> None:
    render_dir = tmp_path / "renders"
    assert images.frame_index(render_dir, render_dir / "frame_0042" / "cam00.png") == 42
    assert images.frame_index(render_dir, render_dir / "cam00.png") == -1


def test_image_metrics_match_reference_implementations() -> None:
    rendered = torch.rand((3, 16, 16), dtype=torch.float32)
    target = torch.rand((3, 16, 16), dtype=torch.float32)
    metrics = images.ImageMetrics("neu3d", device="cpu", lpips_metric=_lpips_stub())

    values = metrics(rendered, target)

    assert set(values) == {"psnr", "d_ssim", "lpips"}
    assert values["psnr"] == pytest.approx(float(psnr(rendered, target)))
    assert values["d_ssim"] == pytest.approx(float(dssim_loss(rendered, target)))


def test_image_metrics_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        images.ImageMetrics("unknown", device="cpu")


def test_evaluate_and_write_neu3d_outputs(tmp_path: Path) -> None:
    render_dir, gt_dir = _neu3d_tree(tmp_path)
    metrics = images.ImageMetrics("neu3d", device="cpu", lpips_metric=_lpips_stub())
    found = images.collect_image_pairs(render_dir, gt_dir)

    rows = images.evaluate_image_pairs(found.pairs, metrics, render_dir=render_dir)
    assert [(row["frame"], row["camera"]) for row in rows] == [
        (1, "cam00"), (1, "cam09"), (2, "cam00"), (2, "cam09"),
    ]

    metrics_csv = tmp_path / "results" / "metrics.csv"
    per_frame_csv = tmp_path / "results" / "per_frame.csv"
    summary_json = tmp_path / "results" / "summary.json"
    images.write_metrics_csv(rows, metrics.names, metrics_csv)
    images.write_per_frame_csv(rows, metrics.names, per_frame_csv)
    summary = images.camera_summary(rows, metrics.names)
    assert summary is not None
    images.write_camera_summary_json(summary, summary_json, metrics_csv=metrics_csv)

    with metrics_csv.open(encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert list(metric_rows[0]) == ["filename", "psnr", "d_ssim", "lpips"]
    assert metric_rows[-1]["filename"] == images.AVERAGE_ROW_LABEL
    with per_frame_csv.open(encoding="utf-8") as handle:
        assert list(next(csv.DictReader(handle))) == ["frame", "camera", "psnr", "d_ssim", "lpips"]
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert sorted(payload["cameras"]) == ["cam00", "cam09"]
    assert payload["overall"]["count"] == 4

    blocks = images.frame_blocks(rows, metrics.names, block=1)
    assert [label for label, _, _ in blocks] == ["1-1", "2-2"]
    # Frame 2 renders are offset twice as far from the ground truth.
    assert blocks[0][2]["psnr"] > blocks[1][2]["psnr"]


def test_camera_summary_is_none_for_frame_named_files() -> None:
    rows = [{"camera": f"r_{index:03d}", "frame": -1, "psnr": 30.0} for index in range(3)]
    assert images.camera_summary(rows, ("psnr",)) is None


def test_finite_mean_ignores_exact_match_psnr() -> None:
    assert images.finite_mean([float("inf"), 20.0, 30.0]) == pytest.approx(25.0)
    assert np.isnan(images.finite_mean([float("inf")]))
