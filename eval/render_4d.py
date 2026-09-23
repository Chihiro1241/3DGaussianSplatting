"""
eval/render_4d.py
4D ラン (frame_NNNN/ ごとのチェックポイント) を全フレームぶんレンダリングし、
eval/evaluate.py がそのまま食える render/GT のディレクトリ対を作る。

使い方:
    python eval/render_4d.py \
        --run_dir  output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
        --data_dir data/neu3d/coffee_martini/converted_4d \
        --out_dir  output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/renders \
        --split    test

----------------------------------------------------------------------------
なぜ scripts/render.py をループで呼ばないのか
----------------------------------------------------------------------------
scripts/render.py はチェックポイント 1 つを受け取る設計なので、300 フレーム x
2 アームだと 600 回のプロセス起動と 600 回の CUDA 初期化が要る。ここでは
同じ関数 (``render_camera_set``) を 1 プロセス内で呼ぶだけにする。
本体のコードには手を入れない。

----------------------------------------------------------------------------
出力の形と、なぜ GT をミラーするのか
----------------------------------------------------------------------------
eval/evaluate.py のペア照合は

    1. render_dir からの相対パスが gt_dir 配下にそのまま在れば、それを使う
    2. 無ければファイル名 (stem) で引く

の順。Neu3D は 300 フレームすべてが cam00/cam09/cam19 という同じ名前を持つので、
stem 照合に落ちると 300 個の候補が出て「一意に定まらない」として全件スキップ
される。そこで **相対パスが一致する形**にしておく:

    <out_dir>/renders/frame_0001/cam00.png     <- レンダリング結果
    <out_dir>/gt/frame_0001/cam00.png          <- converted_4d/.../images/cam00.png へのリンク

GT 側は実体をコピーせずシンボリックリンクにする (300 x 3 枚ぶんの二重保持を避ける)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gaussian_splatting.config import (  # noqa: E402
    config_from_mapping,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.data import load_dataset  # noqa: E402
from gaussian_splatting.evaluation.runner import render_camera_set  # noqa: E402
from gaussian_splatting.io.checkpoint import (  # noqa: E402
    model_from_checkpoint_state,
    read_checkpoint,
)
from gaussian_splatting.renderer import GaussianRenderer  # noqa: E402


def frame_entries(run_dir: Path) -> list[tuple[int, Path]]:
    """frames_4d.json から (フレーム番号, 最終チェックポイント) を順に返す。"""
    manifest = run_dir / "frames_4d.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"マニフェストがありません: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries: list[tuple[int, Path]] = []
    for record in sorted(payload.get("frames", []), key=lambda r: int(r["frame"])):
        if record.get("status") != "COMPLETED":
            print(f"  [スキップ] frame {record['frame']}: status={record.get('status')}")
            continue
        output = Path(record["output"])
        checkpoint = output / "checkpoints" / "latest.pt"
        if not checkpoint.is_file():
            named = sorted((output / "checkpoints").glob("iteration_*.pt"))
            if not named:
                raise FileNotFoundError(f"チェックポイントがありません: {output}")
            checkpoint = named[-1]
        entries.append((int(record["frame"]), checkpoint))
    return entries


def mirror_ground_truth(
    frame_number: int, data_dir: Path, gt_root: Path,
    image_directory: str, names: list[str],
) -> None:
    """GT を renders 側と同じ相対パスに並べ直す (実体はリンク)。"""
    destination = gt_root / f"frame_{frame_number:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    source_root = data_dir / f"frame_{frame_number:04d}" / image_directory
    for name in names:
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(f"GT 画像がありません: {source}")
        link = destination / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="4D ラン root (frames_4d.json を含む)")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="frame_NNNN/ を含むデータ root")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="renders/ と gt/ を作る先")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image-directory", default="images")
    parser.add_argument("--render-backend", choices=("reference", "cuda"),
                        default="cuda")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = frame_entries(args.run_dir)
    if args.start_frame is not None:
        entries = [e for e in entries if e[0] >= args.start_frame]
    if args.end_frame is not None:
        entries = [e for e in entries if e[0] <= args.end_frame]
    if not entries:
        print("[エラー] 対象フレームがありません")
        return 1

    render_root = args.out_dir / "renders"
    gt_root = args.out_dir / "gt"
    render_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    print(f"{len(entries)} フレームを {args.split} split でレンダリングします")
    started = time.time()
    written_total = 0
    for index, (number, checkpoint) in enumerate(entries, start=1):
        state = read_checkpoint(checkpoint, map_location="cpu")
        config = config_from_mapping(state["config"])
        device = resolve_device(config.runtime)
        dtype = resolve_dtype(config.runtime)
        model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
        dataset = load_dataset(
            args.data_dir / f"frame_{number:04d}",
            config,
            splits=(args.split,),
            load_points=False,
            image_directory=args.image_directory,
        )
        cameras = getattr(dataset, args.split)
        if not cameras:
            raise ValueError(f"frame {number}: {args.split} split が空です")
        destination = render_root / f"frame_{number:04d}"
        written = render_camera_set(
            model, GaussianRenderer(config.rendering, backend=args.render_backend),
            cameras, destination,
        )
        written_total += len(written)
        mirror_ground_truth(
            number, args.data_dir, gt_root, args.image_directory,
            [Path(c.image_name).with_suffix(".png").name for c in cameras],
        )
        del model, dataset, cameras, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if index % 50 == 0 or index == len(entries):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0.0
            print(f"  {index}/{len(entries)} フレーム "
                  f"({elapsed:.0f}秒経過, 残り約 "
                  f"{(len(entries) - index) / rate if rate else 0:.0f}秒)", flush=True)

    print(f"レンダリング {written_total} 枚 -> {render_root}")
    print(f"GT ミラー           -> {gt_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
