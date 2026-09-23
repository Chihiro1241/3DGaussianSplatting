"""
eval/rebuild_manifest_4d.py
出力ディレクトリを走査して frames_4d.json を作り直す。

使い方:
    python eval/rebuild_manifest_4d.py --run_dir output/4DGS/neu3d/baseline_30k

----------------------------------------------------------------------------
なぜ必要か
----------------------------------------------------------------------------
``eval/warmstart_trainer.py`` の baseline 経路は、その実行で回したフレームだけを
``records`` に貯めて毎回マニフェストを丸ごと書き直す。つまり途中で落ちて
``--start_frame 150`` で再開すると、出来上がる frames_4d.json は **150 番以降
しか載らない**。``eval/render_4d.py`` はこのマニフェストを唯一の入力に
するので、そのままだと前半 149 フレームが描画対象から消える。

学習そのものはフレームごとに独立 (baseline は毎回 SfM 点群から始まる) なので、
ディスク上の ``frame_NNNN/checkpoints`` が揃っていれば真実は復元できる。
ここではそれを読み直して、全フレームを載せたマニフェストを書き戻す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_records(run_dir: Path, source_root: Path | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for frame_dir in sorted(run_dir.glob("frame_*")):
        if not frame_dir.is_dir():
            continue
        checkpoints = frame_dir / "checkpoints"
        has_checkpoint = checkpoints.is_dir() and (
            (checkpoints / "latest.pt").is_file()
            or any(checkpoints.glob("iteration_*.pt"))
        )
        if not has_checkpoint:
            print(f"  [除外] {frame_dir.name}: チェックポイントが無い (未完了)")
            continue

        number = int(frame_dir.name.split("_")[1])
        telemetry_path = frame_dir / "training_telemetry.json"
        telemetry = (
            json.loads(telemetry_path.read_text(encoding="utf-8"))
            if telemetry_path.is_file() else {}
        )
        if telemetry.get("status") not in (None, "COMPLETED"):
            print(f"  [除外] {frame_dir.name}: status={telemetry.get('status')}")
            continue

        source = (
            source_root / f"frame_{number:04d}" if source_root is not None else None
        )
        records.append(
            {
                "frame": number,
                "source": str(source) if source is not None else "",
                "output": str(frame_dir),
                "status": telemetry.get("status", "COMPLETED"),
                "initialization": "sfm_points",
                "iterations": telemetry.get("iteration"),
                "num_gaussians_end": telemetry.get("gaussian_count"),
                "wall_seconds": telemetry.get("training_wall_time_seconds"),
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--source_path", type=Path, default=None,
                        help="データ root。省略すると source フィールドは空になる "
                             "(render_4d.py は使わないので実害は無い)")
    parser.add_argument("--frame_count", type=int, default=None,
                        help="データ側の総フレーム数。省略すると検出数を使う")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"[エラー] {args.run_dir} がありません")
        return 1

    manifest_path = args.run_dir / "frames_4d.json"
    previous = 0
    if manifest_path.is_file():
        try:
            previous = len(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("frames", [])
            )
        except json.JSONDecodeError:
            previous = 0

    records = build_records(args.run_dir, args.source_path)
    if not records:
        print("[エラー] 完了したフレームが 1 つもありません")
        return 1

    numbers = [int(r["frame"]) for r in records]
    payload = {
        "frames": records,
        "status": "COMPLETED",
        "frame_count": args.frame_count if args.frame_count else len(records),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    missing = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
    print(f"マニフェスト更新: {manifest_path}")
    print(f"  収録フレーム: {len(records)} 件 "
          f"(frame {min(numbers)}..{max(numbers)} / 更新前は {previous} 件)")
    if missing:
        print(f"  [警告] 範囲内の欠番 {len(missing)} 件: {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
