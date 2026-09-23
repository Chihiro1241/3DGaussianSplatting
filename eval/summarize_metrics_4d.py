"""
eval/summarize_metrics_4d.py
eval/evaluate.py が書いた per-image CSV を、カメラ別・全体で集計する。

使い方:
    python eval/summarize_metrics_4d.py \
        --warm     eval/results/warmstart_full/neu3d_coffee_martini.csv \
        --baseline eval/results/baseline_full/neu3d_coffee_martini.csv

evaluate.py の CSV は 1 行 1 画像 (filename,psnr,d_ssim,lpips) で、最終行に
全体平均が入る。ここでは filename (cam00.png など) でグループ分けして
カメラ別平均を出し、warm-start と baseline を並べる。

Neu3D の公式指定では cam00 が held-out の中央参照カメラ。本体の COLMAP
ローダーは 8 枚ごと固定分割なので test は cam00 / cam09 / cam19 になる。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

METRICS = ("psnr", "d_ssim", "lpips")
BETTER = {"psnr": "↑", "d_ssim": "↓", "lpips": "↓"}


def read_rows(path: Path) -> list[dict]:
    """per-image 行だけを返す (集計行や空行は落とす)。"""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("filename") or "").strip()
            if not name or not name.lower().endswith(".png"):
                continue
            try:
                rows.append({"camera": Path(name).stem,
                             **{m: float(row[m]) for m in METRICS}})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warm = read_rows(args.warm)
    base = read_rows(args.baseline)
    if not warm or not base:
        print("[エラー] CSV に per-image 行がありません")
        return 1

    cameras = sorted({r["camera"] for r in warm} | {r["camera"] for r in base})
    print(f"画像数: warm-start {len(warm)} / baseline {len(base)}")
    print()
    header = f"{'カメラ':<10}"
    for metric in METRICS:
        header += f"{metric.upper() + ' ' + BETTER[metric]:>24}"
    print(header)
    print(f"{'':10}" + f"{'warm / base / 差':>24}" * len(METRICS))
    print("-" * (10 + 24 * len(METRICS)))

    for camera in cameras + ["全体"]:
        if camera == "全体":
            w_rows, b_rows = warm, base
        else:
            w_rows = [r for r in warm if r["camera"] == camera]
            b_rows = [r for r in base if r["camera"] == camera]
        if not w_rows or not b_rows:
            continue
        line = f"{camera:<10}"
        for metric in METRICS:
            w, b = mean([r[metric] for r in w_rows]), mean([r[metric] for r in b_rows])
            line += f"{w:7.3f} /{b:7.3f} /{w - b:+7.3f}"
        print(line)

    print()
    print("※ 差は warm-start − baseline。PSNR は正が、D-SSIM / LPIPS は負が warm-start 優位。")
    print("※ evaluate.py の neu3d プリセットは MS-SSIM ではなく D-SSIM を出す。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
