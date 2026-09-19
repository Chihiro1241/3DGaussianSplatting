"""既存の学習済みチェックポイントから可視化用スナップショットを抽出する。

学習時に ``--snapshot-interval`` を付け忘れた run や、すでに完了している
``output/`` 配下の run を、再学習なしで ``eval/snapshot_viewer.py`` にかける
ための補助スクリプト。``checkpoints/iteration_*.pt`` を走査し、中心座標と
不透明度だけを取り出して ``snapshots/iteration_*.npz`` を書き出す。

チェックポイントは optimizer の moment や RNG 状態まで含む重いファイルなので、
``model_state_dict`` の ``means_world`` と ``raw_opacities`` だけを読み、
Config の再構築や model の再構築は行わない。

使い方::

    # 単一シーンの run
    python eval/extract_snapshots.py --run output/4DGS/neu3d/neu3d_coffee_martini_frame1

    # 4D run root (frame_0001/ ... を自動で走査)
    python eval/extract_snapshots.py --run output/4DGS/neu3d/warmstart_neu3d_trial

``latest.pt`` / ``best.pt`` / ``recovery.pt`` は番号付きチェックポイントの
複製または別系統なので既定では無視する。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from gaussian_splatting.training.snapshot import (  # noqa: E402
    SNAPSHOT_DIRECTORY_NAME,
    SNAPSHOT_FORMAT_VERSION,
    atomic_write_json,
    snapshot_filename,
    write_snapshot_npz,
)

_NUMBERED_CHECKPOINT = re.compile(r"^iteration_(\d+)\.pt$")
_FRAME_DIRECTORY = re.compile(r"^frame_(\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="単一シーンの run ディレクトリ、または 4D run root",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        metavar="K",
        help="1スナップショットあたりの最大点数。0 で全ガウシアン (既定: 20000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="S",
        help="番号付きチェックポイントを S 個おきに抽出する (既定: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の npz を上書きする (既定: 既存ファイルはスキップ)",
    )
    return parser


def numbered_checkpoints(run_directory: Path) -> list[tuple[int, Path]]:
    """``checkpoints/iteration_*.pt`` を iteration 昇順で返す。"""

    directory = run_directory / "checkpoints"
    if not directory.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _NUMBERED_CHECKPOINT.match(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    found.sort()
    return found


def discover_runs(root: Path) -> list[tuple[int, Path]]:
    """``(frame番号, run ディレクトリ)`` を返す。単一 run なら frame=1。"""

    if not root.is_dir():
        raise NotADirectoryError(f"ディレクトリではありません: {root}")
    if (root / "checkpoints").is_dir():
        return [(1, root)]
    frames: list[tuple[int, Path]] = []
    for directory in sorted(root.iterdir()):
        match = _FRAME_DIRECTORY.match(directory.name)
        if match is not None and (directory / "checkpoints").is_dir():
            frames.append((int(match.group(1)), directory))
    if not frames:
        raise FileNotFoundError(
            f"{root} の直下にも frame_*/ 配下にも checkpoints/ が見つかりません"
        )
    return frames


def extract_run(
    run_directory: Path,
    *,
    frame: int,
    max_points: int | None,
    stride: int,
    overwrite: bool,
) -> int:
    """1 run 分を抽出し、書き出したスナップショット数を返す。"""

    checkpoints = numbered_checkpoints(run_directory)
    if not checkpoints:
        print(f"  [skip] 番号付きチェックポイントなし: {run_directory}", flush=True)
        return 0
    selected = checkpoints[:: max(stride, 1)]
    if selected[-1] != checkpoints[-1]:
        selected.append(checkpoints[-1])

    destination = run_directory / SNAPSHOT_DIRECTORY_NAME
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    written = 0
    for iteration, path in selected:
        target = destination / snapshot_filename(iteration)
        if target.exists() and not overwrite:
            print(f"  [keep] {target.name}", flush=True)
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_state = state["model_state_dict"]
        means = model_state["means_world"].to(torch.float32).numpy()
        opacities = (
            torch.sigmoid(model_state["raw_opacities"].to(torch.float32))
            .reshape(-1)
            .numpy()
        )
        entry = write_snapshot_npz(
            target,
            means=means,
            opacities=opacities,
            iteration=iteration,
            frame=frame,
            max_points=max_points,
        )
        entries.append(entry)
        written += 1
        print(
            f"  [write] {target.name}: {entry['num_gaussians']} Gaussians "
            f"-> {entry['stored_points']} points",
            flush=True,
        )

    if entries or not (destination / "index.json").is_file():
        _write_index(destination, frame=frame, entries=entries, overwrite=overwrite)
    return written


def _write_index(
    destination: Path,
    *,
    frame: int,
    entries: list[dict[str, object]],
    overwrite: bool,
) -> None:
    """既存 index を壊さないよう、iteration をキーにマージして書き出す。"""

    import json

    merged: dict[int, dict[str, object]] = {}
    index_path = destination / "index.json"
    if index_path.is_file() and not overwrite:
        try:
            previous = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in previous.get("snapshots", []):
                if (destination / str(entry.get("file"))).is_file():
                    merged[int(entry["iteration"])] = entry
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            merged = {}
    for entry in entries:
        merged[int(entry["iteration"])] = entry
    atomic_write_json(
        index_path,
        {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "frame": frame,
            "interval": 0,
            "max_points": None,
            "source": "eval/extract_snapshots.py",
            "snapshots": [merged[key] for key in sorted(merged)],
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stride < 1:
        raise ValueError("--stride は 1 以上である必要があります")
    if args.max_points < 0:
        raise ValueError("--max-points は負にできません")

    runs = discover_runs(args.run)
    total = 0
    for frame, run_directory in runs:
        print(f"[frame {frame}] {run_directory}", flush=True)
        total += extract_run(
            run_directory,
            frame=frame,
            max_points=args.max_points or None,
            stride=args.stride,
            overwrite=args.overwrite,
        )
    print(f"\n{len(runs)} run から {total} スナップショットを書き出しました。")
    print(
        "ビューワ: "
        f"streamlit run eval/snapshot_viewer.py -- --run {args.run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
