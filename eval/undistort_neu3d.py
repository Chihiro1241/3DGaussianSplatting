"""
eval/undistort_neu3d.py
Neu3D の全フレームを第 1 フレームと同一の処理で undistort し、
scripts/train_4d.py が読める「1 フレーム 1 シーンディレクトリ」配置を作る。

使い方:
    python eval/undistort_neu3d.py \
        --frames_dir data/neu3d/coffee_martini/frames \
        --colmap_dir data/neu3d/coffee_martini/colmap/sparse/0 \
        --out_dir    data/neu3d/coffee_martini/converted_4d \
        --downscale  2 \
        --workers    6

----------------------------------------------------------------------------
なぜ OpenCV で書き直さず colmap image_undistorter を呼ぶのか
----------------------------------------------------------------------------
第 1 フレームの undistort は外部バイナリ ``colmap image_undistorter`` が行った
(eval/convert_neu3d.py は歪み補正済みの出力を受け取るだけ)。

image_undistorter は出力解像度をカメラごとに「黒縁が最小になる」最適化で決めて
おり、coffee_martini では 2674x2005 〜 2698x2023 とカメラごとに異なる値が出て
いる。この決め方を Python 側で再現し損ねると、既存の sparse モデル
(cameras.bin の寸法) と画素が食い違い、内部パラメータのスケール換算がずれる。

そこで **同じバイナリを同じ sparse モデルに対して呼ぶ**。intrinsics は
cameras.bin に固定されているのでフレームによらず結果の幾何は同一になり、
「第 1 フレームと同じ intrinsics・同じ処理」が近似ではなく保証になる。
``--verify_frame`` は実際に第 1 フレームを再生成して既存の images_2/ と
バイト一致するかを確かめる (既定で有効)。

----------------------------------------------------------------------------
出力配置
----------------------------------------------------------------------------
scripts/train_4d.py は「データ root の下にフレームごとの通常の 3DGS シーン」を期待する:

    <out_dir>/
      frame_0001/
        sparse/0/{cameras,images,points3D}.bin   <- 全フレーム共通 (第1フレームの SfM)
        images/cam00.png ... cam20.png           <- そのフレームの undistort 済み画像
      frame_0002/
      ...

sparse は第 1 フレームの再構成を全フレームで共有する。Neu3D はカメラが静止した
多視点リグなので、ポーズはフレーム間で不変であり、これが正しい扱いになる。

``--downscale 2`` を付けると images/ には 1/2 縮小版を書く。cameras.bin は
原寸のままだが、load_colmap_dataset が実画像サイズとの比で内部パラメータを
自動換算するので sparse 側を書き換える必要はない。
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_splatting.data.colmap_loader import read_colmap_model  # noqa: E402


def camera_names(colmap_dir: Path) -> list[str]:
    """COLMAP が登録した画像名を名前順で返す (cam00.png ...)。"""
    _, images, _, _ = read_colmap_model(colmap_dir, read_points=False)
    return sorted(image.name for image in images.values())


def undistort_one_frame(
    frame_index: int,
    frames_dir: Path,
    colmap_dir: Path,
    out_dir: Path,
    names: list[str],
    downscale: int,
) -> tuple[int, int]:
    """1 フレーム分を undistort し、<out_dir>/frame_NNNN/images/ へ書く。

    Returns:
        ``(frame_index, 書き出した画像数)``
    """
    frame_root = out_dir / f"frame_{frame_index:04d}"
    image_out = frame_root / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    # 既に揃っているなら何もしない (再開できるようにする)
    if len(list(image_out.glob("*.png"))) == len(names):
        return frame_index, 0

    with tempfile.TemporaryDirectory(prefix=f"neu3d_{frame_index:04d}_") as scratch:
        staging = Path(scratch) / "input"
        staging.mkdir()
        # image_undistorter は images.bin の画像名で入力を探すので、
        # frames/<cam>/frame_NNNN.png をその名前でリンクし直す。
        for name in names:
            camera = Path(name).stem
            source = frames_dir / camera / f"frame_{frame_index:04d}.png"
            if not source.is_file():
                raise FileNotFoundError(f"フレーム画像がありません: {source}")
            (staging / name).symlink_to(source.resolve())

        undistorted = Path(scratch) / "undistorted"
        completed = subprocess.run(
            [
                "colmap", "image_undistorter",
                "--image_path", str(staging),
                "--input_path", str(colmap_dir),
                "--output_path", str(undistorted),
                "--output_type", "COLMAP",
            ],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"frame {frame_index} の image_undistorter が失敗しました:\n"
                f"{completed.stderr[-2000:]}"
            )

        produced = undistorted / "images"
        written = 0
        for name in names:
            source = produced / name
            if not source.is_file():
                raise FileNotFoundError(f"undistort 出力がありません: {source}")
            destination = image_out / name
            if downscale > 1:
                # convert_neu3d.py の --downscale と同一の手順にすること。
                # 変えると第 1 フレームとバイト一致しなくなる。
                with Image.open(source) as image:
                    width = max(1, image.width // downscale)
                    height = max(1, image.height // downscale)
                    image.convert("RGB").resize(
                        (width, height), Image.LANCZOS
                    ).save(destination)
            else:
                shutil.copy2(source, destination)
            written += 1
    return frame_index, written


def link_sparse(sparse_dir: Path, out_dir: Path, frame_indices: list[int]) -> None:
    """各フレームディレクトリへ共通の sparse/0 を配置する。

    ここに置くのは **undistort 済みの PINHOLE モデル** であって、
    image_undistorter へ --input_path として渡した歪みあり側ではない。
    images/ が undistort 済みなので、対応する内部パラメータもそちらに
    揃っていなければ本体の COLMAP ローダーが弾く (PINHOLE のみ対応)。

    カメラが静止したリグなのでポーズはフレーム間で不変。実体は 1 つで足りる
    ので、frame_0001 にだけ実体を置き、以降はそこへのリンクにする。
    """
    first = out_dir / f"frame_{frame_indices[0]:04d}" / "sparse" / "0"
    if first.is_symlink():
        first.unlink()
    first.mkdir(parents=True, exist_ok=True)
    for entry in sorted(first.iterdir()):
        if entry.is_file():
            entry.unlink()
    for entry in sorted(sparse_dir.iterdir()):
        if entry.is_file():
            shutil.copy2(entry, first / entry.name)
    for index in frame_indices[1:]:
        target = out_dir / f"frame_{index:04d}" / "sparse"
        target.mkdir(parents=True, exist_ok=True)
        link = target / "0"
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        link.symlink_to(first.resolve(), target_is_directory=True)


def verify_against_reference(
    out_dir: Path, reference_dir: Path, names: list[str], frame_index: int,
) -> dict:
    """再生成した第 1 フレームが既存の images_2/ とバイト一致するか確かめる。"""
    produced = out_dir / f"frame_{frame_index:04d}" / "images"
    identical, differing, missing = [], [], []
    for name in names:
        left, right = produced / name, reference_dir / name
        if not right.is_file():
            missing.append(name)
        elif filecmp.cmp(left, right, shallow=False):
            identical.append(name)
        else:
            differing.append(name)
    return {"identical": identical, "differing": differing, "missing": missing}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames_dir", type=Path, required=True,
                        help="frames/<cam>/frame_%%04d.png の親")
    parser.add_argument("--colmap_dir", type=Path, required=True,
                        help="image_undistorter へ渡す COLMAP sparse モデル "
                             "(歪みあり側。SIMPLE_RADIAL)")
    parser.add_argument("--sparse_dir", type=Path, default=None,
                        help="各 frame_NNNN/sparse/0 に置くモデル。"
                             "undistort 済みの PINHOLE 側を指定する "
                             "(既定: --colmap_dir。通常は明示すること)")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="frame_NNNN/ を作る先")
    parser.add_argument("--downscale", type=int, default=1,
                        help="1/N に縮小して書き出す (既定 1 = 原寸)")
    parser.add_argument("--start_frame", type=int, default=1)
    parser.add_argument("--end_frame", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6,
                        help="同時に処理するフレーム数")
    parser.add_argument("--verify_against", type=Path, default=None,
                        help="第 1 フレームのバイト一致を確かめる既存ディレクトリ "
                             "(例 .../converted/images_2)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = camera_names(args.colmap_dir)
    if args.sparse_dir is not None and camera_names(args.sparse_dir) != names:
        raise SystemExit("--colmap_dir と --sparse_dir の画像名が一致しません")
    indices = list(range(args.start_frame, args.end_frame + 1))
    print(f"カメラ {len(names)} 台 / フレーム {len(indices)} 枚 "
          f"({args.start_frame}..{args.end_frame})")
    print(f"出力: {args.out_dir}  (downscale 1/{args.downscale})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                undistort_one_frame, index, args.frames_dir, args.colmap_dir,
                args.out_dir, names, args.downscale,
            ): index
            for index in indices
        }
        for future in as_completed(futures):
            future.result()  # 例外はここで送出させる
            done += 1
            if done % 50 == 0 or done == len(indices):
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (len(indices) - done) / rate if rate else 0.0
                print(f"  {done}/{len(indices)} フレーム "
                      f"({elapsed:.0f}秒経過, 残り約 {remaining:.0f}秒)", flush=True)

    link_sparse(args.sparse_dir or args.colmap_dir, args.out_dir, indices)
    print(f"sparse/0 を {len(indices)} フレームへ配置しました")

    if args.verify_against is not None:
        result = verify_against_reference(
            args.out_dir, args.verify_against, names, args.start_frame
        )
        print(f"[検証] frame_{args.start_frame:04d} vs {args.verify_against}")
        print(f"  バイト一致 {len(result['identical'])} / "
              f"不一致 {len(result['differing'])} / "
              f"参照側に無し {len(result['missing'])}")
        if result["differing"]:
            print(f"  不一致: {', '.join(result['differing'][:5])}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
