"""
data/convert_neu3d.py
Neural 3D Video (Neu3D) 形式 -> 本体が読める形式への変換。

本体の scripts/train.py / scripts/render.py / eval/evaluate.py には一切手を入れず、
データ側だけをこのリポジトリが読める形へ寄せるためのアダプタ。

使い方:
  # COLMAP ネイティブ配置 (推奨)
  python data/convert_neu3d.py \
      --colmap_dir data/neu3d/coffee_martini/colmap/sparse/0 \
      --frames_dir data/neu3d/coffee_martini/frames \
      --out_dir    data/neu3d/coffee_martini/converted

  # NeRF Synthetic 形式も併せて出す場合
  python data/convert_neu3d.py ... --nerf_out_dir data/neu3d/coffee_martini/converted_nerf

----------------------------------------------------------------------------
入力形式 (実データで確認済み: coffee_martini)
----------------------------------------------------------------------------
<scene>/
  cam00.mp4 .. cam20.mp4  : 同期済み多視点動画。2704x2028, 300 フレーム, 30fps。
                            欠番 (cam03/15/17) は公式が除外した不良ストリーム。
  poses_bounds.npy        : (N, 17) LLFF 形式。N = 有効ストリーム数 (18)。
                            全カメラで focal=1460.754, H=2028, W=2704。
  frames/<cam>/frame_%04d.png : 本スクリプトの前段 (ffmpeg) で展開したもの。

カメラポーズは poses_bounds.npy にも入っているが、3DGS は SfM 初期点群を必要と
するため、第 1 フレームの多視点画像に COLMAP をかけた結果 (--colmap_dir) を
正とする。poses_bounds.npy は --verify_poses で整合性チェックにのみ使う。

----------------------------------------------------------------------------
なぜ出力を 2 系統に分けるか
----------------------------------------------------------------------------
本体の ``load_dataset`` は分岐の順序が

    1. transforms_train.json があれば NeRF Synthetic ローダー
    2. camera_poses_blender.json があれば Blender ローダー
    3. sparse/0/ があれば COLMAP ローダー

なので、同じディレクトリに transforms_train.json と sparse/ を両方置くと
**COLMAP 経路が手前で遮られる**。そして両者は等価ではない:

  * NeRF Synthetic 経路 (dataset.py: load_nerf_synthetic_dataset)
      - cx = W/2, cy = H/2, fx = fy を強制する
      - initial_points / initial_colors を返さない (scene_center も原点固定)
        => SfM 点群が捨てられ、AABB ランダム初期化になる
  * COLMAP 経路 (dataset.py: load_colmap_dataset)
      - fx, fy, cx, cy をそのまま使う
      - SfM 点群と色を初期化へ渡す
      - ただし PINHOLE モデルのみ、かつ test 分割は 8 枚ごと固定 (index % 8 == 0)

実シーンで SfM 点群を捨てるのは品質に直結するため、**COLMAP 配置を主**とし、
NeRF Synthetic 形式は別ディレクトリ (--nerf_out_dir) へ副次的に出す。

----------------------------------------------------------------------------
NeRF Synthetic 形式での内部パラメータの扱い
----------------------------------------------------------------------------
convert_hypernerf.py と同じ方針。捨てられる cx/cy/fx≠fy を「画像側を
クロップ/リサンプルして厳密に成立させる」ことで吸収する:

  1. 各カメラで主点まわりに対称な最大領域を取る (主点が中心に来る)
  2. 全カメラ共通の正規化画角 tan_x = min_i(hx_i/fx_i), tan_y = min_i(hy_i/fy_i)
  3. 出力を W:H = tan_x:tan_y として正方画素にする (fx = fy が厳密に成立)
  4. PIL resize(box=...) の浮動小数 box で 1 回だけリサンプルする

**未対応**: COLMAP が PINHOLE で解いた時点で歪みは無視されている。元動画は
GoPro 系の歪みを持つため、この残差は補正できず再構成品質の上限を制限する。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_splatting.data.colmap_loader import (  # noqa: E402
    read_colmap_model,
    qvec_to_rotation_cw,
)

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# COLMAP モデルの読み出し
# ---------------------------------------------------------------------------

def load_colmap(colmap_dir: Path):
    """sparse モデルを読み、名前順のカメラ一覧を返す。

    本体の load_colmap_dataset と同じ ``sorted(images, key=name)`` 順を使う。
    分割 (index % 8) がこの順序に依存するため、ここを揃えておくことが重要。
    """
    cameras, images, points, colors = read_colmap_model(colmap_dir, read_points=True)
    ordered = sorted(images.values(), key=lambda image: image.name)
    if not ordered:
        raise ValueError(f"COLMAP モデルに画像がありません: {colmap_dir}")
    records = []
    for image in ordered:
        if image.camera_id not in cameras:
            raise ValueError(f"{image.name} が未知のカメラ {image.camera_id} を参照しています")
        intrinsics = cameras[image.camera_id]
        if intrinsics.model != "PINHOLE" or len(intrinsics.params) != 4:
            raise ValueError(
                f"カメラモデル {intrinsics.model!r} は非対応です。"
                "feature_extractor に --ImageReader.camera_model PINHOLE を指定してください"
            )
        fx, fy, cx, cy = (float(v) for v in intrinsics.params)
        rotation_cw = qvec_to_rotation_cw(image.qvec).numpy().astype(np.float64)
        translation_cw = np.asarray(image.tvec, dtype=np.float64)
        center = -(rotation_cw.T @ translation_cw)
        records.append(
            {
                "name": image.name,
                "stem": Path(image.name).stem,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "width": int(intrinsics.width), "height": int(intrinsics.height),
                "rotation_cw": rotation_cw,
                "translation_cw": translation_cw,
                "center": center,
            }
        )
    return records, points, colors


def split_indices(count: int) -> dict[str, list[int]]:
    """本体 load_colmap_dataset と同一の 8 枚ごと分割を再現する。

    NeRF Synthetic 側の出力もこれに揃えることで、両経路の train/test が
    同じカメラ集合になり、PSNR を直接比較できる。
    """
    test = [index for index in range(count) if index % 8 == 0]
    train = [index for index in range(count) if index % 8 != 0]
    return {"train": train, "test": test, "val": test}


# ---------------------------------------------------------------------------
# 出力 1: COLMAP ネイティブ配置 (主)
# ---------------------------------------------------------------------------

def source_image(
    record: dict, frames_dir: Path | None, images_dir: Path | None, frame_index: int,
) -> Path:
    """このカメラの入力画像を返す。

    ``--images_dir`` が指定されていればその平坦なディレクトリを優先する
    (image_undistorter の出力がこの形)。無ければ frames/<cam>/frame_%04d.png。
    """
    if images_dir is not None:
        path = images_dir / record["name"]
        if not path.is_file():
            raise FileNotFoundError(f"画像がありません: {path}")
        return path
    if frames_dir is None:
        raise ValueError("--frames_dir と --images_dir のどちらかが必要です")
    path = frames_dir / record["stem"] / f"frame_{frame_index:04d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"フレーム画像がありません: {path}")
    return path


def write_colmap_layout(
    colmap_dir: Path, frames_dir: Path | None, images_dir: Path | None,
    out_dir: Path, frame_index: int, downscales: tuple[int, ...] = (),
) -> dict:
    """<out_dir>/sparse/0/ と <out_dir>/images[_N]/ を作る。

    images/ には入力画像の実体へのシンボリックリンクを張る。COLMAP の
    images.bin が持つ画像名 (cam00.png) をそのまま使う必要があるため、
    リンク名は COLMAP 側の名前に合わせる。

    ``downscales`` を与えると images_N/ に 1/N 縮小版も書く (Mip-NeRF360 と
    同じ命名で、本体の ``--image-directory images_2`` がそのまま使える)。
    load_colmap_dataset は実画像サイズと cameras.bin の寸法比から内部
    パラメータを自動で合わせるので、sparse 側は書き換えなくてよい。
    これは ``features.resolution_warmup`` が data.resolution_scale=1.0 を
    要求するための逃げ道でもある: 縮小は読み込み時ではなく事前に行う。
    """
    records, points, _ = load_colmap(colmap_dir)

    sparse_out = out_dir / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    for entry in sorted(colmap_dir.iterdir()):
        if entry.is_file():
            shutil.copy2(entry, sparse_out / entry.name)

    image_out = out_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    linked = []
    for record in records:
        source = source_image(record, frames_dir, images_dir, frame_index)
        destination = image_out / record["name"]
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(source.resolve())
        linked.append(record["name"])

    scaled_sizes: dict[str, list[int]] = {}
    for factor in downscales:
        if factor < 2:
            raise ValueError(f"--downscale は 2 以上にしてください: {factor}")
        scaled_out = out_dir / f"images_{factor}"
        scaled_out.mkdir(parents=True, exist_ok=True)
        for record in records:
            source = source_image(record, frames_dir, images_dir, frame_index)
            destination = scaled_out / record["name"]
            if destination.exists():
                continue
            with Image.open(source) as image:
                width = max(1, image.width // factor)
                height = max(1, image.height // factor)
                image.convert("RGB").resize((width, height), Image.LANCZOS).save(
                    destination
                )
        with Image.open(scaled_out / records[0]["name"]) as probe:
            scaled_sizes[f"images_{factor}"] = list(probe.size)

    splits = split_indices(len(records))
    return {
        "layout": "colmap",
        "frame_index": frame_index,
        "cameras": len(records),
        "points": int(points.shape[0]) if points is not None else 0,
        "train": [records[i]["stem"] for i in splits["train"]],
        "test": [records[i]["stem"] for i in splits["test"]],
        "images": linked,
        "downscaled": scaled_sizes,
    }


# ---------------------------------------------------------------------------
# 出力 2: NeRF Synthetic 形式 (副)
# ---------------------------------------------------------------------------

def blender_c2w(record: dict) -> list[list[float]]:
    """Blender 系 camera-to-world 行列 (4x4)。

    本体 blender_c2w_to_opencv_w2c の逆変換にあたる。
    """
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = record["rotation_cw"].T
    c2w[:3, 3] = record["center"]
    return [[float(v) for v in row] for row in c2w @ np.diag([1.0, -1.0, -1.0, 1.0])]


def write_nerf_layout(
    colmap_dir: Path, frames_dir: Path | None, images_dir: Path | None,
    out_dir: Path, frame_index: int,
) -> dict:
    """transforms_{train,val,test}.json と、画角を揃えた画像を書き出す。

    image_undistorter の出力はカメラごとに画像サイズが異なる。NeRF Synthetic
    ローダーは全フレーム同一サイズを要求するので、共通画角への切り出しが
    ここでサイズ統一も兼ねる。
    """
    records, _, _ = load_colmap(colmap_dir)

    # --- 1) 主点対称の最大領域から共通画角を求める ---
    tan_x = min(
        min(r["cx"], r["width"] - r["cx"]) / r["fx"] for r in records
    )
    tan_y = min(
        min(r["cy"], r["height"] - r["cy"]) / r["fy"] for r in records
    )
    if not (tan_x > 0 and tan_y > 0):
        raise ValueError("共通画角を決定できませんでした (主点が画像外の可能性)")

    # --- 2) 正方画素になる出力サイズ (画素密度は先頭カメラの fx に合わせる) ---
    out_w = max(16, int(round(2.0 * records[0]["fx"] * tan_x)))
    out_h = max(16, int(round(out_w * tan_y / tan_x)))
    camera_angle_x = 2.0 * math.atan(tan_x)
    # 丸めの結果 fx = fy がどれだけ崩れるか (本体は fx=fy を強制する)
    fx_out = (out_w / 2.0) / tan_x
    fy_out = (out_h / 2.0) / tan_y
    aspect_residual = abs(fy_out - fx_out) / fx_out

    splits = split_indices(len(records))
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in splits[split]:
            record = records[index]
            box = (
                record["cx"] - record["fx"] * tan_x,
                record["cy"] - record["fy"] * tan_y,
                record["cx"] + record["fx"] * tan_x,
                record["cy"] + record["fy"] * tan_y,
            )
            source = source_image(record, frames_dir, images_dir, frame_index)
            destination = split_dir / f"{record['stem']}.png"
            if not destination.exists():
                with Image.open(source) as image:
                    image.convert("RGB").resize(
                        (out_w, out_h), Image.LANCZOS, box=box
                    ).save(destination)
            frames.append(
                {
                    "file_path": f"./{split}/{record['stem']}",
                    "transform_matrix": blender_c2w(record),
                }
            )
        (out_dir / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": camera_angle_x, "frames": frames}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        counts[split] = len(frames)

    keep_x = float(np.mean([2.0 * r["fx"] * tan_x / r["width"] for r in records]))
    keep_y = float(np.mean([2.0 * r["fy"] * tan_y / r["height"] for r in records]))
    return {
        "layout": "nerf_synthetic",
        "frame_index": frame_index,
        "source_size": [records[0]["width"], records[0]["height"]],
        "output_size": [out_w, out_h],
        "camera_angle_x": camera_angle_x,
        "fx_fy_residual": aspect_residual,
        "fov_kept_x": keep_x,
        "fov_kept_y": keep_y,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# poses_bounds.npy との整合性チェック
# ---------------------------------------------------------------------------

def verify_against_poses_bounds(colmap_dir: Path, poses_bounds: Path) -> dict:
    """COLMAP の解と配布ポーズが同じ構図か、カメラ中心の相対配置で確認する。

    COLMAP の再構成は相似変換の自由度 (スケール・回転・並進) を持つので絶対値は
    比較できない。そこで「カメラ中心どうしの距離行列」をそれぞれの平均距離で
    正規化し、その相対誤差を見る。構図が一致していれば小さくなる。
    """
    records, _, _ = load_colmap(colmap_dir)
    raw = np.load(poses_bounds)
    if raw.shape[0] != len(records):
        return {
            "checked": False,
            "reason": f"カメラ数が不一致: COLMAP {len(records)} 台 vs "
                      f"poses_bounds {raw.shape[0]} 台",
        }
    # LLFF: 3x5 の 4 列目までが [down, right, backwards | t]、5 列目が hwf。
    llff_centers = raw[:, :15].reshape(-1, 3, 5)[:, :, 3]
    colmap_centers = np.stack([r["center"] for r in records])

    def normalized_distances(points: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        return distances / distances.mean()

    residual = np.abs(
        normalized_distances(colmap_centers) - normalized_distances(llff_centers)
    )
    return {
        "checked": True,
        "cameras": len(records),
        "max_relative_error": float(residual.max()),
        "mean_relative_error": float(residual.mean()),
    }


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--colmap_dir", type=Path, required=True,
                        help="COLMAP sparse モデル (例 .../colmap/sparse/0)")
    parser.add_argument("--frames_dir", type=Path, default=None,
                        help="ffmpeg で展開した frames/<cam>/frame_%%04d.png の親")
    parser.add_argument("--images_dir", type=Path, default=None,
                        help="COLMAP の画像名と同名のファイルが並ぶ平坦なディレクトリ。"
                             "image_undistorter の出力 (undistorted/images) を"
                             "指定する場合はこちら。frames_dir より優先される")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="COLMAP ネイティブ配置の出力先 (主)")
    parser.add_argument("--nerf_out_dir", type=Path, default=None,
                        help="NeRF Synthetic 形式の出力先 (副)。"
                             "out_dir と同じにしてはいけない (COLMAP 経路が遮られる)")
    parser.add_argument("--downscale", type=int, nargs="*", default=(),
                        metavar="N",
                        help="images_N/ に 1/N 縮小版も書き出す (例 --downscale 2 4)。"
                             "本体の --image-directory images_2 で使う。"
                             "resolution_warmup は data.resolution_scale=1.0 を"
                             "要求するため、縮小は事前に済ませておく必要がある")
    parser.add_argument("--frame_index", type=int, default=1,
                        help="使用するフレーム番号 (既定 1 = 第1フレーム)")
    parser.add_argument("--poses_bounds", type=Path, default=None,
                        help="指定すると poses_bounds.npy と構図の整合性を検証する")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames_dir is None and args.images_dir is None:
        raise SystemExit("--frames_dir か --images_dir のどちらかを指定してください")
    if args.nerf_out_dir is not None:
        if args.nerf_out_dir.resolve() == args.out_dir.resolve():
            raise SystemExit(
                "--nerf_out_dir と --out_dir は別にしてください。"
                "同一だと transforms_train.json が COLMAP 経路を遮り、SfM 点群が捨てられます"
            )

    colmap_stats = write_colmap_layout(
        args.colmap_dir, args.frames_dir, args.images_dir,
        args.out_dir, args.frame_index, tuple(args.downscale),
    )
    print(f"[COLMAP 配置] {args.out_dir}")
    print(f"  カメラ {colmap_stats['cameras']} 台 / 点群 {colmap_stats['points']:,} 点")
    print(f"  train ({len(colmap_stats['train'])}): {', '.join(colmap_stats['train'])}")
    print(f"  test  ({len(colmap_stats['test'])}): {', '.join(colmap_stats['test'])}")
    for name, size in colmap_stats["downscaled"].items():
        print(f"  {name}/: {size[0]}x{size[1]}")

    report = {"colmap": colmap_stats}
    if args.nerf_out_dir is not None:
        nerf_stats = write_nerf_layout(
            args.colmap_dir, args.frames_dir, args.images_dir,
            args.nerf_out_dir, args.frame_index,
        )
        report["nerf_synthetic"] = nerf_stats
        print(f"[NeRF Synthetic 配置] {args.nerf_out_dir}")
        print(f"  {nerf_stats['source_size']} -> {nerf_stats['output_size']}")
        print(f"  camera_angle_x = {math.degrees(nerf_stats['camera_angle_x']):.2f} 度")
        print(f"  fx/fy 残差 = {nerf_stats['fx_fy_residual']:.2e}")
        print(f"  画角保持率 x={nerf_stats['fov_kept_x']:.1%} y={nerf_stats['fov_kept_y']:.1%}")
        print(f"  frames: " + ", ".join(
            f"{k}={v}" for k, v in nerf_stats["counts"].items()))

    if args.poses_bounds is not None:
        check = verify_against_poses_bounds(args.colmap_dir, args.poses_bounds)
        report["poses_bounds_check"] = check
        if check["checked"]:
            print("[poses_bounds 照合] カメラ配置の正規化距離行列")
            print(f"  平均相対誤差 {check['mean_relative_error']:.4f} / "
                  f"最大 {check['max_relative_error']:.4f}")
        else:
            print(f"[poses_bounds 照合] スキップ: {check['reason']}")

    (args.out_dir / "conversion_stats.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
