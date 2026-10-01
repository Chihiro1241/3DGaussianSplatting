"""
scripts/4DGS/tracking/evaluate_tracking.py
予測軌跡を正解軌跡と比べ、Dynamic 3D Gaussians (arXiv:2308.09713) Table 1 の
3D / 2D 追跡指標 (MTE, δ, Survival) をシーンごとと全体平均で出す。
指標の定義は extensions/4dgs/tracking_metrics.py (論文当時の PIPs++ に準拠)。

使い方:
    python scripts/4DGS/tracking/evaluate_tracking.py \
        --scene juggle  pred/juggle.npz  gt/juggle.npz \
        --scene boxes   pred/boxes.npz   gt/boxes.npz \
        --unit-to-cm 100 --output-csv results/tracking.csv

予測は scripts/4DGS/tracking/track_points.py の出力でも、他手法 (3GS-O など) が
同じキーで書いたものでもよい。

----------------------------------------------------------------------------
入力 (.npz) のキー
----------------------------------------------------------------------------
予測と正解は同じキーを使い、軌跡の本数・順番とフレーム数を揃えておくこと
(先頭が基準フレーム)。

3D (tracks_3d があれば評価する)
    tracks_3d    (Q, T, 3)  ワールド座標。--unit-to-cm で cm に直す
    valid_3d     (Q, T)     正解のみ・任意。無ければ全フレーム有効
2D (tracks_2d があれば評価する)
    tracks_2d    (M, T, 2)  画素 (画素中心が整数)
    image_size   (2,) か (M, 2)  (width, height)。正規化に使う
    valid_2d     (M, T)     正解のみ・任意。無ければ画像内 ([1, W-2] x [1, H-2]) を有効
    visible_2d   (M, T)     正解のみ・任意。あれば visible_2d[:, 0] の軌跡だけを
                            評価する (「最初の点がそのカメラで可視」)

2D の軌跡は「3D の正解軌跡 × それが見えるカメラ」ごとに 1 本。1 シーンの全カメラの
軌跡を 1 つにまとめて指標を出す (PIPs++ の 1 動画に当たる)。Mean はシーン平均。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTENSION_ROOT = _REPO_ROOT / "extensions" / "4dgs"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from tracking_metrics import (  # noqa: E402
    DELTA_THRESHOLDS,
    SURVIVAL_THRESHOLD_2D,
    SURVIVAL_THRESHOLD_3D_CM,
    TrackMetrics,
    errors_2d,
    errors_3d,
    in_image_mask,
    scene_mean,
    track_metrics,
)


#: Table 1 の行 (課題, 指標名, TrackMetrics の属性, 小数桁)。
ROWS = (
    ("3D Tracking", "3D MTE↓", "mte", 2),
    ("3D Tracking", "3D δ↑", "delta_avg", 1),
    ("3D Tracking", "3D Surv↑", "survival", 1),
    ("2D Tracking", "2D MTE↓", "mte", 2),
    ("2D Tracking", "2D δ↑", "delta_avg", 1),
    ("2D Tracking", "2D Surv↑", "survival", 1),
)


def evaluate_scene(
    predicted: np.lib.npyio.NpzFile,
    truth: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
) -> dict[str, TrackMetrics]:
    results: dict[str, TrackMetrics] = {}
    if "tracks_3d" in truth:
        errors = errors_3d(predicted["tracks_3d"], truth["tracks_3d"], args.unit_to_cm)
        valid = truth["valid_3d"] if "valid_3d" in truth else None
        results["3D"] = track_metrics(
            errors, valid, thresholds=args.thresholds,
            survival_threshold=args.survival_3d,
        )
    if "tracks_2d" in truth:
        predicted_2d = predicted["tracks_2d"]
        truth_2d = truth["tracks_2d"]
        sizes = np.broadcast_to(truth["image_size"], (truth_2d.shape[0], 2))
        valid = truth["valid_2d"] if "valid_2d" in truth else in_image_mask(truth_2d, sizes)
        if "visible_2d" in truth:
            keep = truth["visible_2d"][:, 0].astype(bool)
            predicted_2d, truth_2d = predicted_2d[keep], truth_2d[keep]
            sizes, valid = sizes[keep], valid[keep]
        results["2D"] = track_metrics(
            errors_2d(predicted_2d, truth_2d, sizes), valid,
            thresholds=args.thresholds, survival_threshold=args.survival_2d,
        )
    if not results:
        raise ValueError("正解に tracks_3d も tracks_2d もありません")
    return results


def format_table(scenes: list[str], table: dict[str, dict[str, TrackMetrics]]) -> str:
    columns = [*scenes, "Mean"]
    width = max(10, *(len(c) for c in columns))
    lines = [f"{'Task':12s} {'Metric':10s} " + " ".join(f"{c:>{width}s}" for c in columns)]
    for task, name, attribute, digits in ROWS:
        kind = name[:2]
        if not any(kind in table[scene] for scene in scenes):
            continue
        cells = []
        for column in columns:
            result = table[column].get(kind)
            cells.append(f"{getattr(result, attribute):>{width}.{digits}f}" if result
                         else f"{'-':>{width}s}")
        lines.append(f"{task:12s} {name:10s} " + " ".join(cells))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", nargs=3, action="append", required=True,
                        metavar=("NAME", "PRED", "GT"), help="シーン名と予測・正解の .npz")
    parser.add_argument("--unit-to-cm", type=float, default=1.0,
                        help="3D 座標の 1 単位が何 cm か (メートルなら 100)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DELTA_THRESHOLDS))
    parser.add_argument("--survival-2d", type=float, default=SURVIVAL_THRESHOLD_2D,
                        help="2D Survival の閾値 (正規化 px)。論文当時の PIPs++ は 50、現行は 16")
    parser.add_argument("--survival-3d", type=float, default=SURVIVAL_THRESHOLD_3D_CM,
                        help="3D Survival の閾値 (cm)")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table: dict[str, dict[str, TrackMetrics]] = {}
    scenes = []
    for name, predicted_path, truth_path in args.scene:
        with np.load(predicted_path) as predicted, np.load(truth_path) as truth:
            table[name] = evaluate_scene(predicted, truth, args)
        scenes.append(name)
    table["Mean"] = {
        kind: scene_mean([table[s][kind] for s in scenes if kind in table[s]])
        for kind in ("3D", "2D") if any(kind in table[s] for s in scenes)
    }

    print(format_table(scenes, table))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scene", "kind", "num_tracks", "mte", "delta_avg", "survival",
                             *(f"delta_{t:g}" for t in args.thresholds)])
            for scene in [*scenes, "Mean"]:
                for kind, result in table[scene].items():
                    writer.writerow([scene, kind, result.num_tracks, result.mte,
                                     result.delta_avg, result.survival,
                                     *(result.delta[float(t)] for t in args.thresholds)])
        print(f"CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
