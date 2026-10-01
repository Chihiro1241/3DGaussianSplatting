"""
scripts/4DGS/tracking/plot_tracks.py
2D 軌跡を 1 視点の画像に重ねて描く (Dynamic 3D Gaussians, arXiv:2308.09713 の
Fig. 6 相当: 予測を青、正解を赤)。正解を渡さなければ予測だけを描く。

使い方:
    python scripts/4DGS/tracking/plot_tracks.py \
        --pred tracks_cam00.npz \
        --image data/dynamic/neu3d/cook_spinach/converted_4d/frame_0050/images/cam00.png \
        --frame 50 --trail 15 --out tracks_cam00_f050.png
    # 正解と重ねるとき:  --gt gt_cam00.npz

--frame は .npz の frames に入っているフレーム番号。その時刻までの直近
--trail フレームを線で、その時刻の位置を点で描く。背景画像はそのフレームの
同じカメラの画像を渡すこと (画素座標がそのまま重なる)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PREDICTION_COLOUR = (40, 90, 255)
TRUTH_COLOUR = (255, 40, 40)


def draw_tracks(
    draw: ImageDraw.ImageDraw,
    tracks: np.ndarray,
    start: int,
    end: int,
    colour: tuple[int, int, int],
    width: int,
) -> None:
    """Draw ``tracks[:, start:end + 1]`` as polylines with a dot at ``end``."""

    radius = width + 1
    for track in tracks[:, start:end + 1]:
        points = [tuple(p) for p in track if np.isfinite(p).all()]
        if len(points) >= 2:
            draw.line(points, fill=colour, width=width)
        if np.isfinite(track[-1]).all():
            x, y = track[-1]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", type=Path, required=True, help="予測の .npz (tracks_2d)")
    parser.add_argument("--gt", type=Path, default=None, help="正解の .npz (tracks_2d)")
    parser.add_argument("--image", type=Path, required=True, help="背景にする画像")
    parser.add_argument("--frame", type=int, required=True, help="描く時刻のフレーム番号")
    parser.add_argument("--trail", type=int, default=15, help="線で描く直近のフレーム数")
    parser.add_argument("--every", type=int, default=1, help="この間隔で軌跡を間引く")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with np.load(args.pred) as predicted:
        frames = predicted["frames"]
        prediction = predicted["tracks_2d"][::args.every]
    matches = np.flatnonzero(frames == args.frame)
    if not len(matches):
        print(f"[エラー] frame {args.frame} は予測にありません ({frames[0]}..{frames[-1]})")
        return 1
    end = int(matches[0])
    start = max(0, end - args.trail)

    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    if args.gt is not None:
        with np.load(args.gt) as truth:
            ground_truth = truth["tracks_2d"][::args.every]
        if ground_truth.shape != prediction.shape:
            print(f"[エラー] 予測と正解の形が違います: {prediction.shape} vs {ground_truth.shape}")
            return 1
        draw_tracks(draw, ground_truth, start, end, TRUTH_COLOUR, args.line_width)
    draw_tracks(draw, prediction, start, end, PREDICTION_COLOUR, args.line_width)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"{len(prediction)} 本 (frame {frames[start]}..{frames[end]}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
