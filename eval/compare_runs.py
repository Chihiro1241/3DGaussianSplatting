"""
eval/compare_runs.py
2 つのランの評価 CSV を突き合わせて、カメラ別・ブロック別に並べる。

使い方:
    # フレーム別 CSV (per_frame.csv) 同士 — カメラ別 + フレームブロック別
    python eval/compare_runs.py \
        --a output/4DGS/neu3d/coffee_martini/baseline_30k/results/per_frame.csv --a_label "30,000 iter" \
        --b output/4DGS/neu3d/coffee_martini/baseline_7k/results/per_frame.csv  --b_label "7,000 iter" \
        --block 10

    # per-image CSV (metrics.csv) 同士 — カメラ別のみ
    python eval/compare_runs.py \
        --a output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/results/metrics.csv --a_label warm-start \
        --b output/4DGS/neu3d/coffee_martini/baseline_neu3d_full/results/metrics.csv  --b_label baseline

入力は列から自動判別する。

* ``frame,camera,psnr,...``   — evaluate_per_frame.py の出力。フレーム番号があるので
  同じ (frame, camera) の集合だけを比べ、フレームブロック別の推移も出せる。
* ``filename,psnr,...``       — evaluate.py の出力。1 行 1 画像で最終行が全体平均。
  filename (cam00.png など) からカメラを取り、カメラ別に平均する。フレーム番号が
  無いのでブロック別は出せない。

フレーム別 CSV では同じフレーム集合だけを比べる。片方にしか無いフレームを混ぜると
「どちらが良いか」ではなく「どのフレームを含んだか」の比較になってしまう。

Neu3D の公式指定では cam00 が held-out の中央参照カメラ。本体の COLMAP ローダーは
8 枚ごと固定分割なので test は cam00 / cam09 / cam19 になる。evaluate.py の neu3d
プリセットは MS-SSIM ではなく D-SSIM を出す。
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


def read(path: Path) -> tuple[list[dict[str, object]], bool]:
    """CSV を読み、(行, フレーム番号を持つか) を返す。

    frame 列があれば per_frame.csv、無ければ evaluate.py の per-image CSV とみなす。
    後者は filename の語幹 (cam00.png → cam00) をカメラ名に使い、集計行は落とす。
    """
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        has_frame = "frame" in fields and "camera" in fields
        for record in reader:
            try:
                if has_frame:
                    key = {"frame": int(record["frame"]), "camera": record["camera"]}
                else:
                    name = (record.get("filename") or "").strip()
                    if not name.lower().endswith(".png"):   # *** AVERAGE *** 行など
                        continue
                    key = {"frame": None, "camera": Path(name).stem}
                row = {**key, **{m: float(record[m]) for m in METRICS}}
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(float(row[m])) for m in METRICS):
                rows.append(row)
    return rows, has_frame


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
    parser.add_argument("--a", type=Path, required=True, help="比較元の CSV")
    parser.add_argument("--b", type=Path, required=True, help="比較先の CSV")
    parser.add_argument("--a_label", default="A")
    parser.add_argument("--b_label", default="B")
    parser.add_argument("--block", type=int, default=10,
                        help="フレームブロック別 PSNR の幅。0 で出さない "
                             "(フレーム番号を持つ CSV のときだけ有効)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.a, args.b):
        if not path.is_file():
            print(f"[エラー] {path} がありません")
            return 1

    rows_a, frames_a = read(args.a)
    rows_b, frames_b = read(args.b)
    if not rows_a or not rows_b:
        print("[エラー] 有効な行がありません")
        return 1
    by_frame = frames_a and frames_b

    dropped = 0
    if by_frame:
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
    if by_frame:
        frame_count = len({int(r["frame"]) for r in rows_a})
        print(f"  共通画像 {len(rows_a):,} 枚 (フレーム {frame_count} 件)")
        if dropped:
            print(f"  [注意] 片方にしか無い {dropped} 枚は比較から除外")
    else:
        print(f"  画像数: A {len(rows_a):,} 枚 / B {len(rows_b):,} 枚")
        if frames_a != frames_b:
            print("  [注意] 片方だけフレーム番号を持つため、カメラ別平均のみで比較する")
        print("  [注意] フレーム番号が無いので、フレーム集合が同じかは検証できない")
    print()

    cameras = sorted({str(r["camera"]) for r in rows_a} & {str(r["camera"]) for r in rows_b})
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

    if by_frame and args.block > 0:
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
