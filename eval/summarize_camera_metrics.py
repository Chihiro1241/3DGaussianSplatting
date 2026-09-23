"""
eval/summarize_camera_metrics.py
eval/evaluate.py が書いた per-image CSV を、カメラ別・全体で集計する。

使い方:
    python eval/summarize_camera_metrics.py --csv output/4DGS/neu3d/coffee_martini/baseline_30k/results/metrics.csv

----------------------------------------------------------------------------
なぜ summarize_metrics_4d.py を使わないのか
----------------------------------------------------------------------------
``eval/summarize_metrics_4d.py`` は ``--warm`` と ``--baseline`` の 2 本を
必ず要求する warm-start 比較専用の道具で、単独のランを要約できない。
同じ CSV を両方に渡せば動きはするが、「warm-start と baseline が完全に一致」
という誤読を招く表が出てしまう。ここでは 1 本の CSV をそのまま集計する。

Neu3D の COLMAP ローダーは 8 枚ごとの固定分割なので test は cam00 / cam09 /
cam19 になる (cam00 が公式指定の held-out 中央参照カメラ)。
evaluate.py の CSV は 1 行 1 画像で filename がベース名 (cam00.png) のため、
そこでグループ分けすればカメラ別平均が出る。最終行の全体平均は集計対象から
外す (二重計上になる)。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

METRICS = ("psnr", "d_ssim", "lpips")
ARROWS = {"psnr": "↑", "d_ssim": "↓", "lpips": "↓"}


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            name = (record.get("filename") or "").strip()
            # evaluate.py は最終行に "*** AVERAGE ***" という集計行を足す
            # (eval/evaluate.py の writer.writerow を参照)。これを 1 枚の画像と
            # して数えると平均が二重計上になるので落とす。
            normalised = name.strip("* ").lower()
            if not normalised or normalised in {"mean", "average", "all", "overall"}:
                continue
            try:
                values = {key: float(record[key]) for key in METRICS}
            except (KeyError, TypeError, ValueError):
                continue
            values["camera"] = Path(name).stem
            rows.append(values)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json_out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.csv.is_file():
        print(f"[エラー] {args.csv} がありません")
        return 1

    rows = read_rows(args.csv)
    if not rows:
        print(f"[エラー] {args.csv} に有効な行がありません")
        return 1

    cameras = sorted({str(row["camera"]) for row in rows})
    print("=" * 66)
    print(f"  カメラ別サマリー  |  {args.csv}")
    print("=" * 66)
    print(f"  画像 {len(rows):,} 枚 / カメラ {len(cameras)} 台 "
          f"({len(rows) // max(len(cameras), 1):,} フレーム相当)")
    print()
    header = f"  {'camera':<10}{'n':>7}" + "".join(
        f"{metric + ' ' + ARROWS[metric]:>16}" for metric in METRICS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    payload: dict[str, object] = {"csv": str(args.csv), "cameras": {}}

    for camera in cameras:
        subset = [row for row in rows if row["camera"] == camera]
        cells = ""
        entry: dict[str, object] = {"count": len(subset)}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            average = mean(values)
            deviation = pstdev(values) if len(values) > 1 else 0.0
            cells += f"{average:>10.4f}±{deviation:<5.3f}"
            entry[metric] = {"mean": average, "std": deviation,
                             "min": min(values), "max": max(values)}
        payload["cameras"][camera] = entry  # type: ignore[index]
        print(f"  {camera:<10}{len(subset):>7}{cells}")

    print("  " + "-" * (len(header) - 2))
    overall: dict[str, object] = {"count": len(rows)}
    cells = ""
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        average = mean(values)
        deviation = pstdev(values) if len(values) > 1 else 0.0
        cells += f"{average:>10.4f}±{deviation:<5.3f}"
        overall[metric] = {"mean": average, "std": deviation,
                           "min": min(values), "max": max(values)}
    payload["overall"] = overall
    print(f"  {'全体':<9}{len(rows):>7}{cells}")
    print()
    print("  ± は per-image の標準偏差 (フレーム間のばらつき)。")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
