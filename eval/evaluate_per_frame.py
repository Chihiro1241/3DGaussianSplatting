"""
eval/evaluate_per_frame.py
フレーム番号つきで PSNR / D-SSIM / LPIPS を出し、カメラ別集計と
N フレームブロックごとの推移を表示する。

使い方:
    python eval/evaluate_per_frame.py \
        --render_dir output/4DGS/neu3d/renders/baseline_30k/renders \
        --gt_dir     output/4DGS/neu3d/renders/baseline_30k/gt \
        --output_csv eval/results/baseline_30k_per_frame.csv \
        --device cuda --rgba_background black --block 10

----------------------------------------------------------------------------
なぜ eval/evaluate.py だけでは足りないのか
----------------------------------------------------------------------------
``eval/evaluate.py`` は CSV の 1 列目に ``pred_path.name`` しか書かない
(同ファイル 278 行目)。Neu3D は 300 フレームすべてが cam00 / cam09 / cam19 と
いう同じ名前なので、出来上がる CSV は **どの行がどのフレームか判別できない**。
カメラ別平均までは出せるが、「フレームが進むと劣化していないか」は見られない。

行の並び順から復元する手もあるが、GT が見つからないペアは黙ってスキップ
されるため (同 232-239 行)、1 つでも欠けると以降のフレーム対応が丸ごと
ずれる。そこで **相対パスの親ディレクトリ名 (frame_0042) から frame を取る**。

数値が evaluate.py と食い違っては比較にならないので、ペア探索・画像読み込み・
指標計算はすべて evaluate.py の関数をそのまま import して使う。
evaluate.py 自体には手を入れない。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate import (  # noqa: E402
    MetricsCalculator,
    collect_image_pairs,
    load_image_np,
)

METRICS = ("psnr", "d_ssim", "lpips")
ARROWS = {"psnr": "↑", "d_ssim": "↓", "lpips": "↓"}


def frame_of(render_dir: Path, path: Path) -> int:
    """renders/frame_0042/cam00.png -> 42。取れなければ -1。"""
    try:
        relative = path.relative_to(render_dir)
    except ValueError:
        return -1
    for part in relative.parts:
        match = re.fullmatch(r"frame_(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def finite(values: list[float]) -> list[float]:
    import math
    return [v for v in values if math.isfinite(v)]


def average(values: list[float]) -> float:
    kept = finite(values)
    return sum(kept) / len(kept) if kept else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--dataset", default="neu3d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips_net", default="vgg", choices=("vgg", "alex", "squeeze"))
    parser.add_argument("--rgba_background", default="black", choices=("white", "black"),
                        help="学習 config の data.rgba_background と揃えること")
    parser.add_argument("--block", type=int, default=10,
                        help="推移表示のブロック幅 (フレーム数)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pairs = collect_image_pairs(args.render_dir, args.gt_dir)
    if not pairs:
        print("[エラー] 有効な画像ペアがありません")
        return 1
    print(f"画像ペア: {len(pairs):,} 組")

    calculator = MetricsCalculator(
        dataset_type=args.dataset, device=args.device, lpips_net=args.lpips_net
    )

    rows: list[dict[str, object]] = []
    for index, (pred_path, gt_path) in enumerate(pairs, start=1):
        predicted = load_image_np(pred_path, args.rgba_background)
        truth = load_image_np(gt_path, args.rgba_background)
        if predicted.shape != truth.shape:
            from PIL import Image
            height, width = truth.shape[:2]
            import numpy as np
            predicted = np.array(
                Image.fromarray(predicted).resize((width, height), Image.LANCZOS)
            )
        values = calculator.compute(predicted, truth)
        rows.append(
            {
                "frame": frame_of(args.render_dir, pred_path),
                "camera": pred_path.stem,
                **{metric: values.get(metric, float("nan")) for metric in METRICS},
            }
        )
        if index % 100 == 0 or index == len(pairs):
            print(f"  {index}/{len(pairs)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "camera", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.output_csv}")

    # ------------------------------------------------------------ カメラ別
    cameras = sorted({str(row["camera"]) for row in rows})
    print()
    print("=" * 70)
    print("  カメラ別")
    print("=" * 70)
    print(f"  {'camera':<10}{'n':>7}" + "".join(f"{m + ' ' + ARROWS[m]:>14}" for m in METRICS))
    print("  " + "-" * 66)
    for camera in cameras:
        subset = [r for r in rows if r["camera"] == camera]
        cells = "".join(
            f"{average([float(r[m]) for r in subset]):>14.4f}" for m in METRICS
        )
        print(f"  {camera:<10}{len(subset):>7}{cells}")
    print("  " + "-" * 66)
    overall = "".join(f"{average([float(r[m]) for r in rows]):>14.4f}" for m in METRICS)
    print(f"  {'全体':<9}{len(rows):>7}{overall}")

    # -------------------------------------------------------- ブロック推移
    numbered = [r for r in rows if int(r["frame"]) > 0]
    if numbered and args.block > 0:
        lowest = min(int(r["frame"]) for r in numbered)
        highest = max(int(r["frame"]) for r in numbered)
        print()
        print("=" * 70)
        print(f"  {args.block} フレームブロックごとの推移")
        print("=" * 70)
        print(f"  {'frames':<14}{'n':>6}" + "".join(f"{m + ' ' + ARROWS[m]:>14}" for m in METRICS))
        print("  " + "-" * 66)
        start = lowest - ((lowest - 1) % args.block)
        block_psnr: list[tuple[str, float]] = []
        while start <= highest:
            stop = start + args.block - 1
            subset = [r for r in numbered if start <= int(r["frame"]) <= stop]
            if subset:
                cells = "".join(
                    f"{average([float(r[m]) for r in subset]):>14.4f}" for m in METRICS
                )
                label = f"{start}-{stop}"
                print(f"  {label:<14}{len(subset):>6}{cells}")
                block_psnr.append(
                    (label, average([float(r["psnr"]) for r in subset]))
                )
            start += args.block

        if len(block_psnr) > 1:
            first_label, first_value = block_psnr[0]
            last_label, last_value = block_psnr[-1]
            best = max(block_psnr, key=lambda item: item[1])
            worst = min(block_psnr, key=lambda item: item[1])
            print("  " + "-" * 66)
            print(f"  先頭ブロック {first_label}: {first_value:.4f} dB")
            print(f"  末尾ブロック {last_label}: {last_value:.4f} dB")
            print(f"  差分 (末尾 - 先頭): {last_value - first_value:+.4f} dB")
            print(f"  最良 {best[0]}: {best[1]:.4f} dB / 最低 {worst[0]}: {worst[1]:.4f} dB")
            print(f"  ブロック間の振れ幅: {best[1] - worst[1]:.4f} dB")
            print()
            print("  baseline は毎フレーム独立学習なので、ここが右肩下がりに")
            print("  ならなければ「フレームが進んでも劣化しない」と言える。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
