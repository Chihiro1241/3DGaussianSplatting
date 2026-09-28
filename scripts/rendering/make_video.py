"""
scripts/rendering/make_video.py
scripts/rendering/render_4d.py が書いた連番 PNG を、カメラごとに 1 本の mp4 にまとめる。
GT との並置や複数カメラの結合はしない (単一視点の描画結果だけを出す)。

使い方:
    python scripts/rendering/make_video.py \
        --render-root output/4DGS/neu3d/coffee_martini/baseline_30k/renders \
        --out-dir     output/4DGS/neu3d/coffee_martini/baseline_30k/videos

出力: ``<out-dir>/<camera>.mp4`` (例: cam00.mp4)

----------------------------------------------------------------------------
なぜ glob ではなく concat リストを使うのか
----------------------------------------------------------------------------
render_4d.py の出力は **フレームごとのディレクトリ**に分かれている:

    renders/frame_0001/cam00.png
    renders/frame_0002/cam00.png

つまり 1 本の動画に必要な画像はディレクトリをまたいで散らばっており、
ffmpeg の ``-pattern_type glob`` (1 ディレクトリ内の連番が前提) では拾えない。
そこで各カメラぶんの concat リストを作って渡す。並び順はフレーム番号で
明示的にソートするので、ファイル名の桁揃えに依存しない。

----------------------------------------------------------------------------
奇数サイズ対策
----------------------------------------------------------------------------
libx264 の yuv420p は幅・高さが偶数である必要がある。奇数のときは
``pad=ceil(iw/2)*2:ceil(ih/2)*2`` で 1 px 足す (描画した画素は削らない)。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

EVEN = "pad=ceil(iw/2)*2:ceil(ih/2)*2"


def frames_for(root: Path, camera: str) -> list[Path]:
    """``frame_NNNN/<camera>.png`` をフレーム番号順に集める。"""
    found: dict[int, Path] = {}
    for directory in root.glob("frame_*"):
        match = re.fullmatch(r"frame_(\d+)", directory.name)
        image = directory / f"{camera}.png"
        if match and image.is_file():
            found[int(match.group(1))] = image
    return [found[number] for number in sorted(found)]


def cameras_in(root: Path) -> list[str]:
    """最初のフレームに在るカメラ名。"""
    directories = [d for d in root.glob("frame_*") if d.is_dir()]
    if not directories:
        raise FileNotFoundError(f"{root} に frame_* がありません")
    first = min(directories, key=lambda d: d.name)
    return sorted(p.stem for p in first.glob("*.png"))


def write_list(images: list[Path], handle) -> str:
    """ffmpeg concat demuxer 用のリストを書く。"""
    for image in images:
        # concat デマクサはシングルクォートをエスケープ記法で受ける。
        path = str(image.resolve()).replace("'", r"'\''")
        handle.write(f"file '{path}'\n")
    handle.flush()
    return handle.name


def encode(images: list[Path], destination: Path, fps: int, crf: int) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as listing:
        write_list(images, listing)
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-r", str(fps), "-f", "concat", "-safe", "0", "-i", listing.name,
            "-vf", EVEN,
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "(no stderr)"
        print(f"  [失敗] {destination.name}: {tail}")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--render-root", type=Path, required=True,
                        help="renders/ を含むディレクトリ (render_4d.py の --out_dir)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="既定: 最初のフレームに在るカメラすべて")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_root = args.render_root / "renders"
    if not render_root.is_dir():
        print(f"[エラー] {render_root} がありません")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cameras = args.cameras or cameras_in(render_root)
    print(f"カメラ: {', '.join(cameras)}")

    made = 0
    for camera in cameras:
        images = frames_for(render_root, camera)
        if not images:
            print(f"[スキップ] {camera}: 画像がありません")
            continue
        destination = args.out_dir / f"{camera}.mp4"
        if encode(images, destination, args.fps, args.crf):
            size = destination.stat().st_size / 1e6
            print(f"  {destination}  ({len(images)} frames, {size:.1f} MB)")
            made += 1
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
