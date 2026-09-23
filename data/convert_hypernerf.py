"""
data/convert_hypernerf.py
Nerfies/HyperNeRF 形式 -> NeRF Synthetic 形式 (transforms_*.json) への変換。

本体の train.py / render.py / evaluate.py には一切手を入れず、データ側だけを
このリポジトリが読める形へ寄せるためのアダプタ。

使い方:
  python data/convert_hypernerf.py \
      --scene_dir data/dynamic/hypernerf/vrig-chicken \
      --out_dir   data/dynamic/hypernerf/vrig-chicken/converted \
      --resolution 2x

----------------------------------------------------------------------------
入力形式 (実データで確認済み)
----------------------------------------------------------------------------
<scene>/
  dataset.json   : {count, num_exemplars, ids, train_ids, val_ids}
  scene.json     : {scale, scene_to_metric, center, near, far}
  camera/<id>.json :
      orientation(3x3), position(3), focal_length, principal_point(2),
      skew, pixel_aspect_ratio, radial_distortion(3),
      tangential_distortion(2), image_size(2)=[W,H]
  rgb/{1x,2x,4x,8x,16x}/<id>.png
  points.npy     : (N,3) メトリック空間の SfM 点群

----------------------------------------------------------------------------
座標系 (points.npy の再投影で実証済み)
----------------------------------------------------------------------------
* ``orientation`` は world->camera 回転 R、``position`` はカメラ中心 C。
  カメラ座標は ``x_cam = R (x_world - C)`` で、+z が前方 (OpenCV 系)。
  vrig-chicken の全点で深度が正になることを確認した。
* ``position`` と ``points.npy`` はどちらもメトリック空間にある。
  正規化は ``p_norm = (p - scene.center) * scene.scale`` で、これを適用すると
  点群の深度が scene.json の near/far 区間に 100% 収まる (生のままでは 0%)。
* NeRF Synthetic の ``transform_matrix`` は Blender 系 camera-to-world なので
  ``c2w_blender = [R^T | C_norm] @ diag(1,-1,-1,1)`` として書き出す。
  これは本体の ``blender_c2w_to_opencv_w2c`` の逆変換にあたる。

----------------------------------------------------------------------------
内部パラメータの扱い (ここが本スクリプトの主眼)
----------------------------------------------------------------------------
NeRF Synthetic 形式は「全フレーム共通の camera_angle_x ひとつ」しか持てず、
本体のローダー (dataset.py: load_nerf_synthetic_dataset) は
``cx = width/2``, ``cy = height/2``, ``fx = fy`` を強制する。
一方 HyperNeRF の実データは:

  * 主点が中心からずれている  (vrig-chicken は x で 160px、
    vrig-3dprinter は y で 241px。1072x1920 に対して 15% / 13%)
  * pixel_aspect_ratio != 1  (fy = focal_length * pixel_aspect_ratio)
  * フレームごとに焦点距離が違う場合がある (broom2 で 5.4% の変動)

これらを無視して書き出すと投影が系統的にずれる。そこで **画像側を
クロップ/リサンプルして、捨てられる内部パラメータが厳密に成立する形へ
持ち込む**:

  1. 各フレームで主点まわりに対称な最大領域を取る
       hx = min(cx, W - cx),  hy = min(cy, H - cy)
     これで主点が切り出し領域の中心に来る。
  2. 正規化画角の共通値を取る
       tan_x = min_i (hx_i / fx_i),  tan_y = min_i (hy_i / fy_i)
     全フレームが同じ画角になり、単一の camera_angle_x が厳密に成立する。
  3. 出力サイズを W_out : H_out = tan_x : tan_y として正方画素にする。
     これで fx = fy が厳密に成立する。
  4. 切り出しと拡縮は PIL の ``resize(size, box=...)`` に浮動小数の box を
     渡して 1 回のリサンプルで行う (整数クロップによる丸め誤差を持ち込まない)。

結果として (画像, 内部パラメータ) の組は厳密に整合する。代償は画角が
わずかに狭まること (下の変換サマリーに削減率を出力する)。

**未対応**: radial_distortion / tangential_distortion。本体のレンダラーに
歪みモデルが無く、配布画像も歪んだままなので、この残差は補正できない。
vrig-chicken の k1=0.175 が最大で、これは再構成品質の上限を制限する。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_SOURCE = {"train": "train_ids", "test": "val_ids", "val": "val_ids"}


def load_scene(scene_dir: Path):
    """dataset.json / scene.json / 全 camera JSON を読む。"""
    dataset = json.loads((scene_dir / "dataset.json").read_text(encoding="utf-8"))
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    for key in ("train_ids", "val_ids"):
        if key not in dataset:
            raise ValueError(f"{scene_dir/'dataset.json'} に {key} がありません")
    for key in ("center", "scale"):
        if key not in scene:
            raise ValueError(f"{scene_dir/'scene.json'} に {key} がありません")
    return dataset, scene


def read_camera(scene_dir: Path, frame_id: str) -> dict:
    path = scene_dir / "camera" / f"{frame_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"カメラ JSON がありません: {path}")
    cam = json.loads(path.read_text(encoding="utf-8"))
    required = ("orientation", "position", "focal_length", "principal_point",
                "pixel_aspect_ratio", "image_size")
    missing = [k for k in required if k not in cam]
    if missing:
        raise ValueError(f"{path} に {missing} がありません")
    return cam


def intrinsics_at_level(cam: dict, level_scale: float):
    """指定解像度における fx, fy, cx, cy, W, H を返す。

    camera/*.json の内部パラメータは 1x (image_size) 基準なので、
    rgb/<level> の実サイズに合わせて一様に拡縮する。
    """
    focal = float(cam["focal_length"])
    fx = focal * level_scale
    fy = focal * float(cam["pixel_aspect_ratio"]) * level_scale
    cx = float(cam["principal_point"][0]) * level_scale
    cy = float(cam["principal_point"][1]) * level_scale
    width = float(cam["image_size"][0]) * level_scale
    height = float(cam["image_size"][1]) * level_scale
    return fx, fy, cx, cy, width, height


def blender_c2w(cam: dict, center: np.ndarray, scale: float) -> list[list[float]]:
    """Blender 系 camera-to-world 行列 (4x4) を返す。"""
    rotation_cw = np.asarray(cam["orientation"], dtype=np.float64)
    position = np.asarray(cam["position"], dtype=np.float64)
    if abs(float(np.linalg.det(rotation_cw)) - 1.0) > 1e-4:
        raise ValueError("orientation が回転行列ではありません (det != 1)")
    camera_center = (position - center) * scale
    c2w = np.eye(4, dtype=np.float64)
    # OpenCV 系 c2w: 回転は R^T、並進はカメラ中心。
    c2w[:3, :3] = rotation_cw.T
    c2w[:3, 3] = camera_center
    # Blender 系へ: y と z の符号を反転 (本体 blender_c2w_to_opencv_w2c の逆)。
    c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    return [[float(v) for v in row] for row in c2w]


def convert(scene_dir: Path, out_dir: Path, resolution: str,
            val_monitor_count: int) -> dict:
    dataset, scene = load_scene(scene_dir)
    center = np.asarray(scene["center"], dtype=np.float64)
    scale = float(scene["scale"])

    rgb_dir = scene_dir / "rgb" / resolution
    if not rgb_dir.is_dir():
        available = sorted(p.name for p in (scene_dir / "rgb").iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"解像度 {resolution} がありません: {rgb_dir}\n  利用可能: {available}"
        )

    train_ids = list(dataset["train_ids"])
    test_ids = list(dataset["val_ids"])
    # 学習中の監視用 val は先頭の一部だけにする (本評価は test 側)。
    val_ids = test_ids[:max(1, min(val_monitor_count, len(test_ids)))]

    # 実画像から解像度倍率を求める (rgb/<level> のサイズ / image_size)。
    used_ids = sorted(set(train_ids) | set(test_ids))
    first_cam = read_camera(scene_dir, used_ids[0])
    with Image.open(rgb_dir / f"{used_ids[0]}.png") as probe:
        actual_w, actual_h = probe.size
    level_scale = actual_w / float(first_cam["image_size"][0])
    if abs(actual_h / float(first_cam["image_size"][1]) - level_scale) > 1e-3:
        raise ValueError("rgb 画像の縦横比が camera JSON の image_size と一致しません")

    # --- 1) 主点対称の最大領域から共通画角を求める ---
    cameras: dict[str, dict] = {}
    tan_x = math.inf
    tan_y = math.inf
    for frame_id in used_ids:
        cam = read_camera(scene_dir, frame_id)
        cameras[frame_id] = cam
        fx, fy, cx, cy, width, height = intrinsics_at_level(cam, level_scale)
        tan_x = min(tan_x, min(cx, width - cx) / fx)
        tan_y = min(tan_y, min(cy, height - cy) / fy)
    if not (math.isfinite(tan_x) and math.isfinite(tan_y)) or tan_x <= 0 or tan_y <= 0:
        raise ValueError("共通画角を決定できませんでした")

    # --- 2) 正方画素になる出力サイズ ---
    # 画素密度は元画像と揃える: 先頭フレームの fx を基準に幅を決める。
    fx0, _, _, _, _, _ = intrinsics_at_level(first_cam, level_scale)
    out_w = max(16, int(round(2.0 * fx0 * tan_x)))
    out_h = max(16, int(round(out_w * tan_y / tan_x)))
    camera_angle_x = 2.0 * math.atan(tan_x)
    # 丸めによる fx/fy の残差 (本体は fx=fy を強制するため、その誤差)
    fx_out = (out_w / 2.0) / tan_x
    fy_out = (out_h / 2.0) / tan_y
    aspect_residual = abs(fy_out - fx_out) / fx_out

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "scene": scene_dir.name,
        "resolution_level": resolution,
        "source_size": [actual_w, actual_h],
        "output_size": [out_w, out_h],
        "camera_angle_x": camera_angle_x,
        "fx_fy_residual": aspect_residual,
        "fov_kept_x": None,
        "fov_kept_y": None,
        "counts": {},
    }

    # 画角の保持率 (元画像全体に対して、共通画角が占める割合)
    keep_x = []
    keep_y = []
    for frame_id in used_ids:
        fx, fy, cx, cy, width, height = intrinsics_at_level(cameras[frame_id], level_scale)
        keep_x.append((2.0 * fx * tan_x) / width)
        keep_y.append((2.0 * fy * tan_y) / height)
    stats["fov_kept_x"] = float(np.mean(keep_x))
    stats["fov_kept_y"] = float(np.mean(keep_y))

    for split, source_key in SPLIT_SOURCE.items():
        ids = {"train": train_ids, "test": test_ids, "val": val_ids}[split]
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for frame_id in ids:
            cam = cameras[frame_id]
            fx, fy, cx, cy, _, _ = intrinsics_at_level(cam, level_scale)
            # 主点まわりに共通画角ぶんを切り出す (浮動小数 box、単一リサンプル)
            box = (cx - fx * tan_x, cy - fy * tan_y,
                   cx + fx * tan_x, cy + fy * tan_y)
            source = rgb_dir / f"{frame_id}.png"
            if not source.is_file():
                raise FileNotFoundError(f"画像がありません: {source}")
            destination = split_dir / f"{frame_id}.png"
            if not destination.exists():
                with Image.open(source) as image:
                    image.convert("RGB").resize(
                        (out_w, out_h), Image.LANCZOS, box=box
                    ).save(destination)
            frames.append(
                {
                    "file_path": f"./{split}/{frame_id}",
                    "transform_matrix": blender_c2w(cam, center, scale),
                }
            )
        payload = {"camera_angle_x": camera_angle_x, "frames": frames}
        (out_dir / f"transforms_{split}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        stats["counts"][split] = len(frames)

    (out_dir / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--resolution", default="2x",
        help="rgb/ 配下の解像度レベル (既定 2x = 536x960。"
             "論文の 960x540 に相当し、vrig-peel-banana は 1x を持たない)",
    )
    parser.add_argument(
        "--val-monitor-count", type=int, default=8,
        help="transforms_val.json に入れる監視用フレーム数 (既定 8)。"
             "本評価は transforms_test.json 側 (val_ids 全件) で行う",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = convert(args.scene_dir, args.out_dir, args.resolution,
                    args.val_monitor_count)
    print(f"=== {stats['scene']} 変換完了 ===")
    print(f"  入力 {stats['source_size'][0]}x{stats['source_size'][1]} "
          f"({stats['resolution_level']}) -> 出力 {stats['output_size'][0]}x{stats['output_size'][1]}")
    print(f"  camera_angle_x = {math.degrees(stats['camera_angle_x']):.2f} 度")
    print(f"  画角保持率  x={stats['fov_kept_x']:.3f}  y={stats['fov_kept_y']:.3f}")
    print(f"  fx/fy 残差 = {stats['fx_fy_residual']*100:.4f}%")
    print(f"  フレーム数: {stats['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
