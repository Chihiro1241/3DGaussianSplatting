"""
eval/compare_runs.py
eval/evaluate_per_frame.py が書いた 2 本の CSV を突き合わせて、
カメラ別・ブロック別に並べる。

使い方:
    python eval/compare_runs.py \
        --a output/4DGS/neu3d/coffee_martini/baseline_30k/results/per_frame.csv  --a_label "30,000 iter" \
        --b output/4DGS/neu3d/coffee_martini/baseline_7k/results/per_frame.csv   --b_label "7,000 iter" \
        --block 10

同じフレーム集合だけを比べる。片方にしか無いフレームを混ぜると
「どちらが良いか」ではなく「どのフレームを含んだか」の比較になってしまう。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

METRICS = ("psnr", "d_ssim", "lpips")
ARROWS = {"psnr": "↑", "d_ssim": "↓", "lpips": "↓"}
# 大きいほど良い指標かどうか。差分の符号の解釈に使う。
HIGHER_IS_BETTER = {"psnr": True, "d_ssim": False, "lpips": False}


def read(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            try:
                row = {
                    "frame": int(record["frame"]),
                    "camera": record["camera"],
                    **{m: float(record[m]) for m in METRICS},
                }
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(float(row[m])) for m in METRICS):
                rows.append(row)
    return rows


def average(rows: list[dict[str, object]], metric: str) -> float:
    values = [float(r[metric]) for r in rows]
    return mean(values) if values else float("nan")


def verdict(metric: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "同等"
    improved = delta > 0 if HIGHER_IS_BETTER[metric] else delta < 0
    return "B が良い" if improved else "A が良い"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a_label", default="A")
    parser.add_argument("--b_label", default="B")
    parser.add_argument("--block", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.a, args.b):
        if not path.is_file():
            print(f"[エラー] {path} がありません")
            return 1

    rows_a, rows_b = read(args.a), read(args.b)
    if not rows_a or not rows_b:
        print("[エラー] 有効な行がありません")
        return 1

    # 共通する (frame, camera) だけを残す。
    keys_a = {(int(r["frame"]), r["camera"]) for r in rows_a}
    keys_b = {(int(r["frame"]), r["camera"]) for r in rows_b}
    shared = keys_a & keys_b
    if not shared:
        print("[エラー] 共通する (frame, camera) がありません")
        return 1
    dropped = len(keys_a - shared) + len(keys_b - shared)
    rows_a = [r for r in rows_a if (int(r["frame"]), r["camera"]) in shared]
    rows_b = [r for r in rows_b if (int(r["frame"]), r["camera"]) in shared]

    print("=" * 78)
    print(f"  A = {args.a_label}   B = {args.b_label}")
    print("=" * 78)
    print(f"  共通画像 {len(shared):,} 枚 "
          f"(フレーム {len({f for f, _ in shared})} 件)")
    if dropped:
        print(f"  [注意] 片方にしか無い {dropped} 枚は比較から除外")
    print()

    cameras = sorted({str(r["camera"]) for r in rows_a})
    for metric in METRICS:
        print(f"  --- {metric.upper()} {ARROWS[metric]} ---")
        print(f"    {'camera':<10}{args.a_label:>16}{args.b_label:>16}"
              f"{'差 (B-A)':>14}   判定")
        for camera in cameras:
            subset_a = [r for r in rows_a if r["camera"] == camera]
            subset_b = [r for r in rows_b if r["camera"] == camera]
            value_a, value_b = average(subset_a, metric), average(subset_b, metric)
            delta = value_b - value_a
            print(f"    {camera:<10}{value_a:>16.4f}{value_b:>16.4f}"
                  f"{delta:>+14.4f}   {verdict(metric, delta)}")
        value_a, value_b = average(rows_a, metric), average(rows_b, metric)
        delta = value_b - value_a
        print(f"    {'全体':<9}{value_a:>16.4f}{value_b:>16.4f}"
              f"{delta:>+14.4f}   {verdict(metric, delta)}")
        print()

    if args.block > 0:
        frames = sorted({int(r["frame"]) for r in rows_a})
        lowest, highest = frames[0], frames[-1]
        print(f"  --- PSNR: {args.block} フレームブロックごと ---")
        print(f"    {'frames':<14}{args.a_label:>16}{args.b_label:>16}{'差 (B-A)':>14}")
        start = lowest - ((lowest - 1) % args.block)
        while start <= highest:
            stop = start + args.block - 1
            subset_a = [r for r in rows_a if start <= int(r["frame"]) <= stop]
            subset_b = [r for r in rows_b if start <= int(r["frame"]) <= stop]
            if subset_a:
                value_a = average(subset_a, "psnr")
                value_b = average(subset_b, "psnr")
                label = f"{start}-{stop}"
                print(f"    {label:<14}{value_a:>16.4f}{value_b:>16.4f}"
                      f"{value_b - value_a:>+14.4f}")
            start += args.block
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
