"""
eval/make_compare_runs_video.py
2 つのラン (A/B) の描画結果を GT と 3 面並べた比較動画にする。

    python eval/make_compare_runs_video.py \
        --gt_root   output/4DGS/neu3d/renders/full_baseline_2000 \
        --a_root    output/4DGS/neu3d/renders/full_baseline_2000 --a_label "baseline 2000" \
        --b_root    output/4DGS/neu3d/renders/full_warmstart_2000 --b_label "warm-start 2000" \
        --out_dir   output/4DGS/neu3d/videos/compare_baseline_vs_warmstart \
        --end_frame 100 --fps 30 10

eval/make_videos_4d.py は 1 ランの GT×Pred しか作れないので、ラン同士を
突き合わせるこちらを別に用意する。GT は両ランで同一 (同じ画像への symlink)
なので片方から取れば良い。
同一カメラなら GT/A/B の解像度は一致するため scale は不要。
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

EVEN = "pad=ceil(iw/2)*2:ceil(ih/2)*2"


def frames_for(root: Path, camera: str,
               start_frame: int, end_frame: int) -> dict[int, Path]:
    """フレーム番号 -> 画像パス。位置ではなく番号で揃えるため dict で返す。
    ランごとに開始フレームが違う (fixed_gaussian は 2 から) ので、
    位置で zip するとフレームがずれた比較動画ができてしまう。"""
    out: dict[int, Path] = {}
    for d in sorted(root.glob("frame_*")):
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if not (start_frame <= n <= end_frame):
            continue
        p = d / f"{camera}.png"
        if p.is_file():
            out[n] = p
    return out


def cameras_in(root: Path) -> list[str]:
    first = next(iter(sorted(root.glob("frame_*"))), None)
    if first is None:
        raise FileNotFoundError(f"{root} に frame_* がありません")
    return sorted(p.stem for p in first.glob("*.png"))


def write_list(images: list[Path], handle) -> None:
    for image in images:
        path = str(image.resolve()).replace("'", r"'\''")
        handle.write(f"file '{path}'\n")
    handle.flush()


def label_filter(index: int, text: str, out: str, with_frame: bool) -> str:
    safe = text.replace("'", "").replace(":", " ")
    f = (f"[{index}:v]drawtext=text='{safe}':fontsize=42:fontcolor=white:"
         f"box=1:boxcolor=black@0.55:x=20:y=20")
    if with_frame:
        f += (",drawtext=text='frame %{eif\\:n+1\\:d}':fontsize=34:"
              "fontcolor=yellow:box=1:boxcolor=black@0.55:x=20:y=76")
    return f + f"[{out}]"


def encode(lists: list[str], labels: list[str], dest: Path,
           fps: int, crf: int, use_text: bool) -> bool:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for name in lists:
        cmd += ["-r", str(fps), "-f", "concat", "-safe", "0", "-i", name]
    tags = "abc"[: len(lists)]
    if use_text:
        parts = [label_filter(i, labels[i], tags[i], i == 0) for i in range(len(lists))]
        chain = ";".join(parts) + ";" + "".join(f"[{t}]" for t in tags)
    else:
        chain = "".join(f"[{i}:v]" for i in range(len(lists)))
    chain += f"hstack=inputs={len(lists)},{EVEN}"
    cmd += ["-filter_complex", chain, "-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "(no stderr)"
        print(f"    [失敗] {dest.name}: {tail}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", type=Path, required=True)
    ap.add_argument("--a_root", type=Path, required=True)
    ap.add_argument("--b_root", type=Path, default=None,
                    help="省略すると GT と A の 2 面構成にする")
    ap.add_argument("--a_label", default="A")
    ap.add_argument("--b_label", default="B")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--start_frame", type=int, default=1)
    ap.add_argument("--end_frame", type=int, default=100)
    ap.add_argument("--fps", type=int, nargs="+", default=[30])
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--cameras", nargs="*", default=None)
    args = ap.parse_args()

    gt_dir = args.gt_root / "gt"
    a_dir = args.a_root / "renders"
    b_dir = (args.b_root / "renders") if args.b_root else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cams = args.cameras or cameras_in(a_dir)
    print(f"カメラ: {', '.join(cams)}   フレーム: {args.start_frame}-{args.end_frame}")

    made = 0
    for cam in cams:
        sources = [("GT", frames_for(gt_dir, cam, args.start_frame, args.end_frame)),
                   (args.a_label, frames_for(a_dir, cam, args.start_frame, args.end_frame))]
        if b_dir is not None:
            sources.append((args.b_label,
                            frames_for(b_dir, cam, args.start_frame, args.end_frame)))
        # 位置ではなくフレーム番号の共通集合で揃える。
        common = sorted(set.intersection(*(set(d) for _, d in sources)))
        n = len(common)
        if n == 0:
            print(f"  [スキップ] {cam}: 共通フレームがありません")
            continue
        counts = " ".join(f"{lab}={len(d)}" for lab, d in sources)
        if any(len(d) != n for _, d in sources):
            print(f"  [注意] {cam}: {counts} -> 共通 {n} フレーム "
                  f"({common[0]}-{common[-1]}) で揃えます")
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as lg, \
                tempfile.NamedTemporaryFile("w", suffix=".txt") as la, \
                tempfile.NamedTemporaryFile("w", suffix=".txt") as lb:
            handles = [lg, la, lb][: len(sources)]
            for handle, (_, mapping) in zip(handles, sources):
                write_list([mapping[k] for k in common], handle)
            names = [h.name for h in handles]
            labels = [lab for lab, _ in sources]
            for fps in args.fps:
                dest = args.out_dir / f"{cam}_compare_{fps}fps.mp4"
                ok = encode(names, labels, dest, fps, args.crf, use_text=True)
                if not ok:
                    print(f"    drawtext なしで再試行: {dest.name}")
                    ok = encode(names, labels, dest, fps, args.crf, use_text=False)
                if ok:
                    size = dest.stat().st_size / 1e6
                    print(f"  {dest.name}  ({n} frames, {n/fps:.1f}s, {size:.1f} MB)")
                    made += 1
    print(f"\n動画 {made} 本 -> {args.out_dir}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
