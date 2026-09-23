"""
eval/gaussian_count_trend.py
各フレームの training_telemetry.json から Gaussian 数と所要時間を集め、
最大 / 最小 / 平均とブロックごとの推移を出す。

使い方:
    python eval/gaussian_count_trend.py --run_dir output/4DGS/neu3d/coffee_martini/baseline_30k --block 10

baseline は毎フレームを SfM 点群から独立に学習するので、Gaussian 数の
フレーム間の振れがそのまま「密度制御の再現性」を表す。warm-start と違って
前フレームを引き継がないぶん、ここが大きく振れるほどフレーム間で
別々の表現に収束していることになる。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def collect(run_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("frame_*/training_telemetry.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") != "COMPLETED":
            continue
        count = data.get("gaussian_count")
        if not isinstance(count, int):
            continue
        records.append(
            {
                "frame": int(path.parent.name.split("_")[1]),
                "gaussian_count": count,
                "iteration": data.get("iteration"),
                "wall_seconds": data.get("training_wall_time_seconds"),
                "peak_cuda_MiB": data.get("peak_cuda_memory_allocated_MiB"),
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = collect(args.run_dir)
    if not records:
        print(f"[エラー] {args.run_dir} に完了フレームがありません")
        return 1

    counts = [int(r["gaussian_count"]) for r in records]
    times = [float(r["wall_seconds"] or 0.0) for r in records]

    print("=" * 70)
    print(f"  Gaussian 数の推移  |  {args.run_dir}")
    print("=" * 70)
    print(f"  完了フレーム : {len(records)}")
    print(f"  最小         : {min(counts):,}  (frame "
          f"{records[counts.index(min(counts))]['frame']:04d})")
    print(f"  最大         : {max(counts):,}  (frame "
          f"{records[counts.index(max(counts))]['frame']:04d})")
    print(f"  平均         : {int(mean(counts)):,}")
    print(f"  標準偏差     : {pstdev(counts):,.0f}"
          f"  (平均比 {100 * pstdev(counts) / mean(counts):.2f}%)")
    print(f"  振れ幅       : {max(counts) - min(counts):,} "
          f"({100 * (max(counts) - min(counts)) / min(counts):.2f}%)")
    if any(times):
        print(f"  学習時間     : 平均 {mean(times):.0f}s / 合計 "
              f"{sum(times) / 3600:.1f} 時間")

    if args.block > 0 and len(records) > args.block:
        print()
        print(f"  --- {args.block} フレームブロックごと ---")
        print(f"  {'frames':<14}{'n':>5}{'平均':>12}{'最小':>12}{'最大':>12}")
        print("  " + "-" * 55)
        lowest = min(int(r["frame"]) for r in records)
        highest = max(int(r["frame"]) for r in records)
        start = lowest - ((lowest - 1) % args.block)
        while start <= highest:
            stop = start + args.block - 1
            subset = [
                int(r["gaussian_count"]) for r in records
                if start <= int(r["frame"]) <= stop
            ]
            if subset:
                label = f"{start}-{stop}"
                print(f"  {label:<14}{len(subset):>5}{int(mean(subset)):>12,}"
                      f"{min(subset):>12,}{max(subset):>12,}")
            start += args.block

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["frame", "gaussian_count", "iteration",
                            "wall_seconds", "peak_cuda_MiB"],
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"\n  CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
