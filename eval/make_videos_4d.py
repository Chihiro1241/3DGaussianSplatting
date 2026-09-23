"""
eval/make_videos_4d.py
eval/render_4d.py が書いた連番 PNG を mp4 にまとめる。

使い方:
    python eval/make_videos_4d.py \
        --render_root output/4DGS/neu3d/coffee_martini/baseline_30k/renders \
        --out_dir     output/4DGS/neu3d/coffee_martini/baseline_30k/videos

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
libx264 の yuv420p は幅・高さが偶数である必要がある。Neu3D の画像は
偶数だが、hstack/vstack した結果が奇数になることがあるため、
``crop=trunc(iw/2)*2:trunc(ih/2)*2`` を全経路に入れておく。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EVEN = "crop=trunc(iw/2)*2:trunc(ih/2)*2"


def frame_number(directory: Path) -> int:
    match = re.search(r"(\d+)", directory.name)
    return int(match.group(1)) if match else -1


def collect(root: Path, camera: str) -> list[Path]:
    """frame_NNNN/<camera>.png をフレーム番号順に集める。"""
    directories = sorted(
        (d for d in root.glob("frame_*") if d.is_dir()), key=frame_number
    )
    return [d / f"{camera}.png" for d in directories if (d / f"{camera}.png").is_file()]


def cameras_present(root: Path) -> list[str]:
    first = min(
        (d for d in root.glob("frame_*") if d.is_dir()), key=frame_number, default=None
    )
    if first is None:
        raise FileNotFoundError(f"{root} に frame_* がありません")
    return sorted(p.stem for p in first.glob("*.png"))


def write_list(images: list[Path], handle) -> str:
    """ffmpeg concat demuxer 用のリストを書く。"""
    for image in images:
        # concat デマクサはシングルクォートをエスケープ記法で受ける。
        path = str(image.resolve()).replace("'", r"'\''")
        handle.write(f"file '{path}'\n")
    handle.flush()
    return handle.name


def probe_width(video: Path) -> int:
    """ffprobe で動画の幅を取る (カメラ間の解像度差を揃えるため)。"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        return 0


def run_ffmpeg(command: list[str], label: str) -> bool:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [失敗] {label}")
        print("    " + result.stderr.strip().splitlines()[-1] if result.stderr else "")
        return False
    return True


def encode_single(images: list[Path], destination: Path, fps: int, crf: int) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as listing:
        write_list(images, listing)
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-r", str(fps), "-f", "concat", "-safe", "0", "-i", listing.name,
            "-vf", EVEN,
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            str(destination),
        ]
        return run_ffmpeg(command, destination.name)


def encode_compare(
    gt_images: list[Path],
    pred_images: list[Path],
    destination: Path,
    fps: int,
    crf: int,
) -> bool:
    """GT を左、予測を右に並べたラベル付きの比較動画。"""
    count = min(len(gt_images), len(pred_images))
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as gt_listing, \
            tempfile.NamedTemporaryFile("w", suffix=".txt") as pred_listing:
        write_list(gt_images[:count], gt_listing)
        write_list(pred_images[:count], pred_listing)
        # drawtext はフォント未指定だと環境によって失敗するので、
        # 文字を焼き込まず色付きの枠だけにする選択肢も残している。
        filters = (
            f"[0:v]drawtext=text='GT':fontsize=48:fontcolor=white:"
            f"box=1:boxcolor=black@0.5:x=16:y=16[left];"
            f"[1:v]drawtext=text='Pred':fontsize=48:fontcolor=white:"
            f"box=1:boxcolor=black@0.5:x=16:y=16[right];"
            f"[left][right]hstack,{EVEN}"
        )
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-r", str(fps), "-f", "concat", "-safe", "0", "-i", gt_listing.name,
            "-r", str(fps), "-f", "concat", "-safe", "0", "-i", pred_listing.name,
            "-filter_complex", filters,
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            str(destination),
        ]
        if run_ffmpeg(command, destination.name):
            return True
        # drawtext が使えない環境向けのフォールバック (ラベルなし)。
        print("    drawtext なしで再試行します")
        command[command.index("-filter_complex") + 1] = f"[0:v][1:v]hstack,{EVEN}"
        return run_ffmpeg(command, destination.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--render_root", type=Path, required=True,
                        help="renders/ と gt/ を含むディレクトリ")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="既定: 最初のフレームに在るカメラすべて")
    parser.add_argument("--skip_stack", action="store_true",
                        help="全カメラ縦積み動画を作らない")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_root = args.render_root / "renders"
    gt_root = args.render_root / "gt"
    if not render_root.is_dir():
        print(f"[エラー] {render_root} がありません")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cameras = args.cameras or cameras_present(render_root)
    print(f"カメラ: {', '.join(cameras)}")

    predicted_videos: list[Path] = []
    for camera in cameras:
        predictions = collect(render_root, camera)
        if not predictions:
            print(f"[スキップ] {camera}: 画像がありません")
            continue
        destination = args.out_dir / f"{camera}_pred.mp4"
        print(f"{camera}: 予測 {len(predictions)} 枚 -> {destination.name}")
        if encode_single(predictions, destination, args.fps, args.crf):
            predicted_videos.append(destination)

        truths = collect(gt_root, camera) if gt_root.is_dir() else []
        if truths:
            compare = args.out_dir / f"{camera}_compare.mp4"
            pairs = min(len(truths), len(predictions))
            print(f"{camera}: GT 比較 {pairs} 組 -> {compare.name}")
            encode_compare(truths, predictions, compare, args.fps, args.crf)
        else:
            print(f"{camera}: GT が無いので比較動画は作りません")

    if not args.skip_stack and len(predicted_videos) > 1:
        stacked = args.out_dir / "all_cameras_pred.mp4"
        # Neu3D の undistort 後の画像はカメラごとに数 px 幅が違う
        # (cam00=1340, cam09=1336, cam19=1342)。vstack は幅の一致を要求するので、
        # 一番狭い幅に揃えてから積む (-2 で高さは偶数を保ったまま自動計算)。
        widths = [probe_width(video) for video in predicted_videos]
        target = min(w for w in widths if w > 0)
        target -= target % 2
        print(f"全カメラ縦積み -> {stacked.name} (幅 {widths} -> {target} に統一)")

        inputs: list[str] = []
        for video in predicted_videos:
            inputs += ["-i", str(video)]
        scaled = "".join(
            f"[{i}:v]scale={target}:-2,setsar=1[s{i}];"
            for i in range(len(predicted_videos))
        )
        labels = "".join(f"[s{i}]" for i in range(len(predicted_videos)))
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex",
            f"{scaled}{labels}vstack=inputs={len(predicted_videos)},{EVEN}",
            "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            str(stacked),
        ]
        run_ffmpeg(command, stacked.name)

    print("\n=== 生成された動画 ===")
    for video in sorted(args.out_dir.glob("*.mp4")):
        print(f"  {video}  ({video.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
