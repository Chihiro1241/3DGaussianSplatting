"""
学習中の損失を CSV に記録するロガー。

本体の Trainer は既に ``<run>/train_log.jsonl`` へ 100 iter ごとに
``loss_total`` / ``loss_l1`` / ``loss_dssim`` / ``psnr`` などを出力している
(src/gaussian_splatting/training/trainer.py)。そのため通常は本モジュールを
学習ループへ差し込む必要はなく、既存ログを CSV へ変換すれば足りる。
scripts/train.py も scripts/train_4d.py も変更しないのはこのため。

用途は 2 つ:

1. 既存の train_log.jsonl から CSV を作る (推奨経路)::

       python eval/loss_logger.py \
           --run_dir output/4DGS/dnerf/experiments/warmstart/lego \
           --out_dir eval/loss_logs/_default --scene lego

   ``--run_dir`` が frames_4d.json を持つ 4D ラン root なら、
   frame_0001/ ... を走査して frame_0000.csv ... を書き出す。
   単一シーンの run ディレクトリなら frame_0000.csv を 1 本だけ書く。

2. 任意の呼び出し元から逐次記録する (LossLogger API)::

       logger = LossLogger(out_dir="eval/loss_logs/_default", scene="lego", frame=0)
       logger.log(iteration=100, loss=0.042)
       logger.close()

出力: ``<out_dir>/<scene>/frame_{:04d}.csv`` (iter, loss の 2 列)。
``close()`` は最終行へ ``*** LAST100_MEAN ***`` 行を追記する。この行の loss は
「最後の 100 iteration 区間の損失平均」で、記録間隔が 100 iter の場合は
最終記録点そのものになるため、実際には最後に記録された点から遡って
100 iteration 分に入るサンプルの平均を取る。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUMMARY_MARKER = "*** LAST100_MEAN ***"


def _mean_last_window(rows: list[tuple[int, float]], window: int = 100) -> float:
    """最終 iteration から遡って ``window`` iteration 分の損失平均を返す。"""
    if not rows:
        return float("nan")
    last_iteration = rows[-1][0]
    selected = [loss for it, loss in rows if it > last_iteration - window]
    if not selected:
        selected = [rows[-1][1]]
    return sum(selected) / len(selected)


class LossLogger:
    """1 フレーム分の (iter, loss) を CSV へ記録する。"""

    def __init__(self, out_dir: str | Path, scene: str, frame: int) -> None:
        if not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        self.path = Path(out_dir) / scene / f"frame_{frame:04d}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[tuple[int, float]] = []
        self._closed = False

    def log(self, iteration: int, loss: float) -> None:
        if self._closed:
            raise RuntimeError("logger is already closed")
        self._rows.append((int(iteration), float(loss)))

    def close(self) -> Path:
        """CSV を書き出し、最終行にサマリーを追記する。"""
        if self._closed:
            return self.path
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["iter", "loss"])
            for iteration, loss in self._rows:
                writer.writerow([iteration, f"{loss:.8g}"])
            writer.writerow([SUMMARY_MARKER, f"{_mean_last_window(self._rows):.8g}"])
        self._closed = True
        return self.path

    def __enter__(self) -> "LossLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def rows_from_train_log(path: Path, loss_key: str = "loss_total") -> list[tuple[int, float]]:
    """train_log.jsonl から (iteration, loss) を読む。"""
    rows: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "iteration" in record and loss_key in record:
            rows.append((int(record["iteration"]), float(record[loss_key])))
    rows.sort(key=lambda item: item[0])
    return rows


def _frame_run_directories(run_dir: Path) -> list[Path]:
    """4D ラン root なら frame_* を順に、そうでなければ自身を返す。"""
    manifest = run_dir / "frames_4d.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        ordered = sorted(payload.get("frames", []), key=lambda r: int(r["frame"]))
        directories = [Path(record["output"]) for record in ordered]
        if directories:
            return directories
    globbed = sorted(p for p in run_dir.glob("frame_*") if p.is_dir())
    return globbed or [run_dir]


def convert(run_dir: Path, out_dir: Path, scene: str,
            loss_key: str = "loss_total") -> list[Path]:
    """run_dir 配下の train_log.jsonl を frame_NNNN.csv 群へ変換する。"""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run ディレクトリがありません: {run_dir}")
    written: list[Path] = []
    for index, frame_dir in enumerate(_frame_run_directories(run_dir)):
        log_path = frame_dir / "train_log.jsonl"
        if not log_path.is_file():
            print(f"  [スキップ] {log_path} がありません")
            continue
        rows = rows_from_train_log(log_path, loss_key)
        if not rows:
            print(f"  [スキップ] {log_path} に {loss_key} がありません")
            continue
        logger = LossLogger(out_dir, scene, index)
        for iteration, loss in rows:
            logger.log(iteration, loss)
        written.append(logger.close())
        print(f"  frame_{index:04d}.csv  ({len(rows)} 点, "
              f"最終100iter平均 {_mean_last_window(rows):.6g})")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="4D ラン root、または単一シーンの run ディレクトリ")
    parser.add_argument("--out_dir", type=Path, default=Path("eval/loss_logs/_default"))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--loss_key", default="loss_total",
                        choices=["loss_total", "loss_l1", "loss_dssim", "psnr"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"=== {args.scene}: {args.run_dir} -> {args.out_dir/args.scene} ===")
    written = convert(args.run_dir, args.out_dir, args.scene, args.loss_key)
    print(f"{len(written)} 本の CSV を書き出しました")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
