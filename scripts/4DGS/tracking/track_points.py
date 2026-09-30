"""
scripts/4DGS/tracking/track_points.py
4D ラン (frame_NNNN/ ごとのチェックポイント) から、クエリ点の軌跡を作る。
Dynamic 3D Gaussians (Luiten et al., arXiv:2308.09713) Sec. 3 の
"Tracking with Dynamic 3D Gaussians" の手順 (extensions/4dgs/point_tracking.py)。

使い方:
    # 3D クエリ (基準フレームのワールド座標, (Q,3) の .npy)
    python scripts/4DGS/tracking/track_points.py \
        --run_dir output/4DGS/neu3d/cook_spinach/cook_spinach_dynreg_7k_250_300f \
        --data_dir data/dynamic/neu3d/cook_spinach/converted_4d \
        --queries-3d queries.npy --out tracks.npz

    # 2D クエリ (カメラ cam00 の画素。--query-pixels (Q,2) の .npy か --query-grid)
    python scripts/4DGS/tracking/track_points.py \
        --run_dir ... --data_dir ... \
        --query-camera cam00 --query-grid 40 --out tracks_cam00.npz

----------------------------------------------------------------------------
出力 (.npz)
----------------------------------------------------------------------------
    frames        (T,)       フレーム番号 (先頭が基準フレーム)
    tracks_3d     (Q, T, 3)  ワールド座標の軌跡
    gaussian      (Q,)       担当ガウシアンの番号 (静的背景は -1)
    influence     (Q,)       基準フレームでの最大影響 f
    2D クエリのときはさらに
    camera        ()         カメラ名
    image_size    (2,)       (width, height)
    query_pixels  (Q, 2)     クエリ画素
    tracks_2d     (Q, T, 2)  カメラへの投影 (画素中心が整数)
    depths        (Q, T)     カメラ座標の深度
    query_alpha   (Q,)       クエリ画素の累積不透明度 (深度の信頼度)

scripts/4DGS/tracking/evaluate_tracking.py がこの形式 (と同じキーを持つ他手法の
予測) を読んで評価する。

----------------------------------------------------------------------------
前提
----------------------------------------------------------------------------
* 基準フレームは最小のフレーム番号 (通常 frame_0001)。
* カメラは全フレームで動かない (Neu3D は固定リグ) ものとし、基準フレームの
  カメラを全フレームの投影に使う。
* 3D クエリは論文どおり、最大影響 f のガウシアンに付け、全ガウシアンで
  f < --background-threshold (既定 0.5) なら静的背景として固定する。
* 2D クエリは中央値深度 (光線の透過率が 0.5 を切るガウシアンの中心深度) で
  3D 点にし、そのガウシアンに付ける。論文の文面 (平均深度 + f の最大) では
  なく、公式の深度描画器と著者の説明 (issue #20) に合わせた。学習済みの
  ガウシアンは平たく不透明度も低いので、f >= 0.5 の規則では大半の点が
  背景になるため。
* 1 フレームずつチェックポイントを読み、担当ガウシアンの位置と回転だけを取り
  出す。全フレーム × 全ガウシアンの回転は持たない。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _import_root in (_REPO_ROOT / "src", _REPO_ROOT / "extensions" / "4dgs"):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from gaussian_splatting.config import (  # noqa: E402
    config_from_mapping,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.data import load_dataset  # noqa: E402
from gaussian_splatting.io.checkpoint import (  # noqa: E402
    model_from_checkpoint_state,
    read_checkpoint,
)

from point_tracking import (  # noqa: E402
    ReferenceGaussians,
    anchor_points,
    anchor_to_gaussians,
    compact_anchors,
    frame_subset,
    median_depth_gaussians,
    project_points,
    propagate,
    render_depth,
    sample_map,
    unproject_pixels,
)


def frame_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """(フレーム番号, 最終チェックポイント) を昇順に返す。"""
    entries = []
    for directory in sorted(run_dir.glob("frame_*")):
        checkpoint = directory / "checkpoints" / "latest.pt"
        if checkpoint.is_file():
            entries.append((int(directory.name.split("_")[1]), checkpoint))
    if not entries:
        raise FileNotFoundError(f"frame_NNNN/checkpoints/latest.pt がありません: {run_dir}")
    return entries


def load_model(checkpoint: Path, device_override: str | None):
    state = read_checkpoint(checkpoint, map_location="cpu")
    config = config_from_mapping(state["config"])
    device = torch.device(device_override) if device_override else resolve_device(config.runtime)
    model = model_from_checkpoint_state(
        state, config, device=device, dtype=resolve_dtype(config.runtime)
    )
    return model, config


def find_camera(data_dir: Path, frame: int, config, name: str, image_directory: str):
    splits = load_dataset(
        data_dir / f"frame_{frame:04d}", config, splits=("train", "val", "test"),
        load_points=False, image_directory=image_directory,
    )
    for camera in [*splits.train, *splits.val, *splits.test]:
        if Path(camera.image_name or "").stem == name:
            return camera
    raise ValueError(f"カメラ {name} が frame {frame} にありません")


def grid_pixels(width: int, height: int, stride: int) -> np.ndarray:
    us = np.arange(stride / 2, width, stride)
    vs = np.arange(stride / 2, height, stride)
    return np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="4D ラン root (frame_NNNN/checkpoints/latest.pt を含む)")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="frame_NNNN/ を含むデータ root (2D クエリで必須)")
    parser.add_argument("--out", type=Path, required=True, help="出力 .npz")
    queries = parser.add_mutually_exclusive_group(required=True)
    queries.add_argument("--queries-3d", type=Path, help="(Q,3) のワールド座標 .npy")
    queries.add_argument("--query-pixels", type=Path, help="(Q,2) の画素 .npy (u, v)")
    queries.add_argument("--query-grid", type=int, help="この間隔 (px) の格子をクエリにする")
    parser.add_argument("--query-camera", default=None, help="2D クエリのカメラ名 (例 cam00)")
    parser.add_argument("--min-alpha", type=float, default=0.5,
                        help="--query-grid で、累積不透明度がこれ未満の画素を捨てる")
    parser.add_argument("--background-threshold", type=float, default=0.5,
                        help="3D クエリで、最大影響 f がこれ未満なら静的背景 (0 で判定なし)")
    parser.add_argument("--image-directory", default="images")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--device", default=None, help="既定はチェックポイントの設定")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    two_d = args.queries_3d is None
    if two_d and (args.query_camera is None or args.data_dir is None):
        print("[エラー] 2D クエリには --query-camera と --data_dir が要ります")
        return 1

    entries = frame_checkpoints(args.run_dir)
    if args.start_frame is not None:
        entries = [e for e in entries if e[0] >= args.start_frame]
    if args.end_frame is not None:
        entries = [e for e in entries if e[0] <= args.end_frame]
    reference_frame, reference_checkpoint = entries[0]
    model, config = load_model(reference_checkpoint, args.device)
    device = model.means_world.device
    dtype = model.means_world.dtype

    output: dict[str, np.ndarray] = {}
    if two_d:
        camera = find_camera(args.data_dir, reference_frame, config,
                             args.query_camera, args.image_directory)
        camera = replace(
            camera,
            rotation_cw=camera.rotation_cw.to(device=device, dtype=dtype),
            translation_cw=camera.translation_cw.to(device=device, dtype=dtype),
            camera_center_world=camera.camera_center_world.to(device=device, dtype=dtype),
            image=None,
        )
        depth, alpha = render_depth(model, camera, config.rendering)
        if args.query_grid is not None:
            pixels_np = grid_pixels(camera.width, camera.height, args.query_grid)
        else:
            pixels_np = np.load(args.query_pixels).reshape(-1, 2)
        pixels = torch.as_tensor(pixels_np, dtype=dtype, device=device)
        query_alpha = sample_map(alpha, pixels)
        if args.query_grid is not None:
            keep = query_alpha >= args.min_alpha
            pixels, query_alpha = pixels[keep], query_alpha[keep]
            print(f"格子 {len(pixels_np)} 点のうち alpha >= {args.min_alpha} の {len(pixels)} 点を使う")
        # 担当は中央値深度を決めたガウシアン。光線が半透明にもならない画素だけ
        # 静的背景にし、その位置は平均深度で置く (累積不透明度 0 なら NaN)。
        carriers, median = median_depth_gaussians(model, camera, config.rendering, pixels)
        mean_depth = torch.where(query_alpha > 0, sample_map(depth, pixels), torch.nan)
        points = unproject_pixels(
            pixels, torch.where(carriers >= 0, median, mean_depth), camera
        )
        output.update(
            camera=np.array(args.query_camera),
            image_size=np.array([camera.width, camera.height]),
            query_pixels=pixels.cpu().numpy(),
            query_alpha=query_alpha.cpu().numpy(),
        )
    else:
        points = torch.as_tensor(np.load(args.queries_3d).reshape(-1, 3),
                                 dtype=dtype, device=device)

    reference = ReferenceGaussians.from_model(model)
    if two_d:
        anchors = anchor_to_gaussians(points, carriers, reference)
        rule = "光線の透過率が 0.5 を切らない"
    else:
        anchors = anchor_points(points, reference,
                                background_threshold=args.background_threshold)
        rule = f"全ガウシアンで f < {args.background_threshold}"
    background = anchors.is_background
    print(f"クエリ {len(points)} 点: 静的背景 {int(background.sum())} 点 ({rule})")
    used, compact = compact_anchors(anchors)
    del model

    started = time.time()
    tracks = []
    for index, (number, checkpoint) in enumerate(entries, start=1):
        model, _ = load_model(checkpoint, args.device)
        with torch.no_grad():
            tracks.append(propagate(compact, frame_subset(model, used)))
        del model
        if index % 50 == 0 or index == len(entries):
            print(f"  {index}/{len(entries)} フレーム ({time.time() - started:.0f}秒)", flush=True)
    tracks_3d = torch.stack(tracks, dim=1)

    output.update(
        frames=np.array([number for number, _ in entries]),
        tracks_3d=tracks_3d.cpu().numpy(),
        gaussian=anchors.gaussian_indices.cpu().numpy(),
        influence=anchors.influences.cpu().numpy(),
    )
    if two_d:
        tracks_2d, depths = project_points(tracks_3d, camera)
        output.update(tracks_2d=tracks_2d.cpu().numpy(), depths=depths.cpu().numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **output)
    print(f"軌跡 {tracks_3d.shape[0]} 本 x {tracks_3d.shape[1]} フレーム -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
