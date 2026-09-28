"""
eval/warmstart_trainer.py
warm-start あり / なし を同じ条件で回すためのドライバ。

使い方:
    # warm-start あり
    python eval/warmstart_trainer.py \
        --source_path data/neu3d/coffee_martini/converted_4d \
        --output_dir  output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial \
        --config      configs/neu3d/trial_5000.yaml \
        --start_frame 1 --end_frame 10 \
        --image-directory images --render-backend cuda

    # warm-start なし (baseline)
    python eval/warmstart_trainer.py ... --no_warmstart

----------------------------------------------------------------------------
なぜ新しい学習ループを書かないのか
----------------------------------------------------------------------------
フレーム間の受け渡しは既に ``scripts/train_4d.py`` + ``extensions/4dgs/trainer_4d.py``
に実装されている。しかも要求どおり **ガウシアンパラメータだけ**を引き継ぎ、
optimizer の Adam モーメント・位置 LR スケジュール・ADC 統計は毎フレーム
ゼロから作り直す (``build_frame_training_state`` の docstring と、それを
実際に検査する ``assert_frame_state_is_reset`` を参照)。

よって本スクリプトは学習ループを持たず、既存の入口を subprocess で呼ぶだけに
する。``scripts/train.py`` にも ``scripts/train_4d.py`` にも変更を加えない。

  * warm-start あり : ``scripts/train_4d.py`` をフレーム範囲に対して 1 回呼ぶ
  * warm-start なし : ``scripts/train.py`` をフレームごとに独立に呼ぶ
                      (各フレームが第 1 フレームと同じ SfM 点群から始まる)

どちらも出力を ``<output_dir>/frame_NNNN/`` に揃え、warm-start なし側でも
``frames_4d.json`` 互換のマニフェストを書く。こうすると後段の
``eval/loss_logger.py`` が両者を同じ手順で CSV 化できる。

----------------------------------------------------------------------------
5000 iter で回すときの注意
----------------------------------------------------------------------------
``scripts/train_4d.py`` の ``--subsequent-frame-iterations`` は **2 フレーム目以降にしか
効かない** (``_frame_config`` が ``carried_over`` のときだけ上書きする)。
第 1 フレームを含めて全フレームを 5000 iter にしたいので、本スクリプトは
``training.iterations`` が既に 5000 の設定ファイル (configs/neu3d/trial_5000.yaml) を
渡すことを前提にし、``--subsequent-frame-iterations`` は使わない。

その設定では ``densify_until_iteration: 15000`` が学習長 5000 を上回るため、
密度制御が最後まで止まらない。これは warm-start / baseline の**両方に等しく
効く**ので比較の妥当性は保たれるが、ガウシアン数の推移は報告すること。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "frames_4d.json"


def frame_directories(source: Path, pattern: str = "frame_*") -> list[Path]:
    """データ root 直下のフレームディレクトリを番号順に返す。"""
    found = sorted(path for path in source.glob(pattern) if path.is_dir())
    if not found:
        raise FileNotFoundError(f"{source} に {pattern} がありません")
    return found


def run(command: list[str], log_path: Path) -> None:
    """子プロセスを実行し、標準出力をログへ流しつつ画面にも出す。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"失敗 (exit {code}): {' '.join(command)}")


def train_warmstart(args: argparse.Namespace, frames: list[Path]) -> None:
    """scripts/train_4d.py に丸ごと任せる (フレーム間の受け渡しはあちら側の責務)。"""
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_4d.py"),
        "--data", str(args.source_path),
        "--config", str(args.config),
        "--output", str(args.output_dir),
        "--start-frame", str(args.start_frame),
        "--end-frame", str(args.end_frame),
        "--image-directory", args.image_directory,
        "--render-backend", args.render_backend,
        # ドライバが driver.log を置くために出力ルートを先に作るので、
        # scripts/train_4d.py 側の exist_policy=error と衝突させない。
        "--allow-existing-output",
    ]
    if args.carry_over_checkpoint is not None:
        command += ["--carry-over-checkpoint", str(args.carry_over_checkpoint)]
    if args.disable_training_evaluation:
        command.append("--disable-training-evaluation")
    run(command, args.output_dir / "driver.log")


def train_baseline(args: argparse.Namespace, frames: list[Path]) -> None:
    """フレームごとに scripts/train.py を独立に呼ぶ。

    受け渡しが無いので各フレームは SfM 点群から始まる。出力とマニフェストは
    warm-start 側と同じ形にして、後段の解析を共通化する。
    """
    records: list[dict[str, object]] = []
    manifest_path = args.output_dir / MANIFEST_NAME
    for number in range(args.start_frame, args.end_frame + 1):
        source = frames[number - 1]
        frame_output = args.output_dir / f"frame_{number:04d}"
        started = time.time()
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
            "--data", str(source),
            "--config", str(args.config),
            "--output", str(frame_output),
            "--image-directory", args.image_directory,
            "--render-backend", args.render_backend,
        ]
        if args.disable_training_evaluation:
            command.append("--disable-training-evaluation")
        print(f"[frame {number}/{args.end_frame}] {source.name}: "
              f"SfM 点群から (warm-start なし) -> {frame_output}", flush=True)
        run(command, args.output_dir / "driver.log")

        telemetry_path = frame_output / "training_telemetry.json"
        telemetry = (
            json.loads(telemetry_path.read_text(encoding="utf-8"))
            if telemetry_path.is_file() else {}
        )
        records.append(
            {
                "frame": number,
                "source": str(source),
                "output": str(frame_output),
                "status": telemetry.get("status", "COMPLETED"),
                "initialization": "sfm_points",
                "iterations": telemetry.get("iteration"),
                "num_gaussians_end": telemetry.get("gaussian_count"),
                "wall_seconds": time.time() - started,
            }
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "frames": records,
                    "status": "COMPLETED" if number == args.end_frame else "RUNNING",
                    "frame_count": len(frames),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_path", type=Path, required=True,
                        help="frame_NNNN/ を含むデータ root")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="training.iterations を目的の値にした設定ファイル")
    parser.add_argument("--start_frame", type=int, default=1,
                        help="1 始まりの開始フレーム")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="1 始まりの終了フレーム (既定: 最後)")
    parser.add_argument("--carry_over_checkpoint", type=Path, default=None,
                        help="--start_frame > 1 で再開するとき、前フレームの"
                             "チェックポイントから初期ガウシアンを引き継ぐ。"
                             "省略すると再開フレームが SfM 点群から始まってしまう")
    parser.add_argument("--no_warmstart", action="store_true",
                        help="フレームごとに独立学習する (baseline)")
    parser.add_argument("--frame-pattern", default="frame_*")
    parser.add_argument("--image-directory", default="images")
    parser.add_argument("--render-backend", choices=("reference", "cuda"),
                        default="cuda")
    parser.add_argument("--disable-training-evaluation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames = frame_directories(args.source_path, args.frame_pattern)
    if args.end_frame is None:
        args.end_frame = len(frames)
    if not 1 <= args.start_frame <= args.end_frame <= len(frames):
        raise SystemExit(
            f"フレーム範囲が不正です: {args.start_frame}..{args.end_frame} "
            f"(利用可能 1..{len(frames)})"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = "baseline (warm-start なし)" if args.no_warmstart else "warm-start あり"
    print(f"=== {mode} / frame {args.start_frame}..{args.end_frame} "
          f"/ config {args.config} ===", flush=True)
    started = time.time()
    if args.no_warmstart:
        train_baseline(args, frames)
    else:
        train_warmstart(args, frames)
    elapsed = time.time() - started
    count = args.end_frame - args.start_frame + 1
    print(f"=== 完了: {count} フレーム / {elapsed:.0f}秒 "
          f"(平均 {elapsed / count:.1f}秒/frame) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
