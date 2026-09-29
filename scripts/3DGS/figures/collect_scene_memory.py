"""Collect per-scene Gaussian count and training memory stats into a CSV."""
import csv
import json
from pathlib import Path

from scenes import REPORT_DIR, ROOT_3DGS, SCENES

OUT = REPORT_DIR / "scene_memory.csv"


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
    for directory, dataset in SCENES:
        run_dir = ROOT_3DGS / directory
        scene = Path(directory).name
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
