"""Collect per-scene Gaussian count and training memory stats into a CSV."""
import csv
import json
from pathlib import Path

# データセット別レイアウト: <ROOT_3DGS>/<dataset>/<scene>/  (1 シーン 1 ラン)
ROOT_3DGS = Path("output/3DGS")
OUT = Path("output/3DGS/benchmark_report/report/scene_memory.csv")

# (directory, dataset, scene) in report order.
SCENES = [
    ("mipnerf360/bicycle", "Mip-NeRF360", "bicycle"),
    ("mipnerf360/bonsai", "Mip-NeRF360", "bonsai"),
    ("mipnerf360/counter", "Mip-NeRF360", "counter"),
    ("mipnerf360/flowers", "Mip-NeRF360", "flowers"),
    ("mipnerf360/garden", "Mip-NeRF360", "garden"),
    ("mipnerf360/kitchen", "Mip-NeRF360", "kitchen"),
    ("mipnerf360/room", "Mip-NeRF360", "room"),
    ("mipnerf360/stump", "Mip-NeRF360", "stump"),
    ("mipnerf360/treehill", "Mip-NeRF360", "treehill"),
    ("tandt/train", "Tanks&Temples", "train"),
    ("tandt/truck", "Tanks&Temples", "truck"),
    ("deepblending/drjohnson", "Deep Blending", "drjohnson"),
    ("deepblending/playroom", "Deep Blending", "playroom"),
    ("nerf_synthetic/chair", "Synthetic NeRF", "chair"),
    ("nerf_synthetic/drums", "Synthetic NeRF", "drums"),
    ("nerf_synthetic/ficus", "Synthetic NeRF", "ficus"),
    ("nerf_synthetic/hotdog", "Synthetic NeRF", "hotdog"),
    ("nerf_synthetic/lego", "Synthetic NeRF", "lego"),
    ("nerf_synthetic/materials", "Synthetic NeRF", "materials"),
    ("nerf_synthetic/mic", "Synthetic NeRF", "mic"),
    ("nerf_synthetic/ship", "Synthetic NeRF", "ship"),
]


def peak_process_vram(run_dir: Path) -> float:
    """Peak process VRAM in MiB, taking the max over every VRAM record.

    drjohnson was trained in two passes (0-7k, then 7k-30k), so its samples are
    split across vram_samples_7000_to_30000.json and validation_summary.json.
    """
    peaks = []
    for path in sorted(run_dir.glob("vram_samples_*.json")):
        peaks.append(json.loads(path.read_text())["peak_process_vram_MiB"])
    validation = run_dir / "validation_summary.json"
    if validation.exists():
        peaks.append(json.loads(validation.read_text())["peak_process_vram_MiB"])
    if not peaks:
        raise FileNotFoundError(f"no VRAM samples under {run_dir}")
    return max(peaks)


def main() -> None:
    rows = []
    for directory, dataset, scene in SCENES:
        run_dir = ROOT_3DGS / directory
        telemetry = json.loads((run_dir / "training_telemetry.json").read_text())
        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "gaussian_count": telemetry["milestones"]["30000"]["gaussian_count"],
                "peak_allocated_MiB": telemetry["peak_cuda_memory_allocated_MiB"],
                "peak_reserved_MiB": telemetry["peak_cuda_memory_reserved_MiB"],
                "peak_process_vram_MiB": peak_process_vram(run_dir),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} scenes)")


if __name__ == "__main__":
    main()
