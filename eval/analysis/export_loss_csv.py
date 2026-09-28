"""
eval/analysis/export_loss_csv.py
学習ログ ``train_log.jsonl`` の損失を、フレームごとの CSV へ書き出す。

本体の Trainer は ``<run>/train_log.jsonl`` へ 100 iter ごとに
``loss_total`` / ``loss_l1`` / ``loss_dssim`` / ``psnr`` などを出力している
(src/gaussian_splatting/training/trainer.py)。本スクリプトはそれを
``plot_loss.py`` / ``summarize_warmstart.py`` が読む CSV へ変換するだけで、
学習には関与しない。

使い方::

    python eval/analysis/export_loss_csv.py \
        --run_dir output/4DGS/dnerf/lego/experiments/warmstart \
        --out_dir output/4DGS/dnerf/lego/experiments/warmstart/loss_logs

``--run_dir`` が frames_4d.json を持つ 4D ラン root なら、
frame_0001/ ... を走査して frame_0000.csv ... を書き出す。
単一シーンの run ディレクトリなら frame_0000.csv を 1 本だけ書く。

出力: ``<out_dir>/frame_{:04d}.csv`` (iter, loss の 2 列)。
シーンは出力パスの階層で表すので (``output/4DGS/neu3d/<scene>/loss_logs/<variant>/``)、
``--out_dir`` はそのまま書き出し先になる。``--scene`` は進捗表示のラベルだけに使う。
最終行には ``*** LAST100_MEAN ***`` 行を追記する。この行の loss は
「最後の 100 iteration 区間の損失平均」で、記録間隔が 100 iter の場合は
最終記録点そのものになるため、実際には最後に記録された点から遡って
100 iteration 分に入るサンプルの平均を取る。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def write_loss_csv(path: Path, rows: list[tuple[int, float]]) -> Path:
    """(iter, loss) を書き、最終行にサマリーを追記する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iter", "loss"])
        for iteration, loss in rows:
            writer.writerow([iteration, f"{loss:.8g}"])
        writer.writerow([SUMMARY_MARKER, f"{_mean_last_window(rows):.8g}"])
    return path


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


# ---- 読み込み側 (plot_loss.py / summarize_warmstart.py が使う) ----
def read_frame_csv(path: Path) -> tuple[list[int], list[float], float]:
    """(iters, losses, last100_mean) を返す。"""
    iters: list[int] = []
    losses: list[float] = []
    summary = float("nan")
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_iter = (row.get("iter") or "").strip()
            raw_loss = (row.get("loss") or "").strip()
            if not raw_loss:
                continue
            if raw_iter == SUMMARY_MARKER:
                summary = float(raw_loss)
                continue
            try:
                iters.append(int(raw_iter))
                losses.append(float(raw_loss))
            except ValueError:
                continue
    if math.isnan(summary) and losses:
        last = iters[-1]
        window = [l for i, l in zip(iters, losses) if i > last - 100] or [losses[-1]]
        summary = sum(window) / len(window)
    return iters, losses, summary


def load_series(log_dir: Path) -> list[dict]:
    """frame_*.csv を番号順に読む。"""
    if not log_dir.is_dir():
        raise FileNotFoundError(f"ログディレクトリがありません: {log_dir}")
    series = []
    for path in sorted(log_dir.glob("frame_*.csv")):
        iters, losses, summary = read_frame_csv(path)
        if not iters:
            print(f"  [スキップ] {path.name}: データ点なし")
            continue
        series.append(
            {
                "name": path.stem,
                "frame": int(path.stem.split("_")[-1]),
                "iters": iters,
                "losses": losses,
                "final": summary,
                "converged_iter": convergence_iteration(iters, losses),
            }
        )
    return series


def convergence_iteration(iters: list[int], losses: list[float],
                          fraction: float = 0.10) -> int | None:
    """損失が初期値の ``fraction`` 以下へ最初に落ちた iteration。

    warm-start では初期損失そのものが低いため、この指標は
    「その run の中でどれだけ下がったか」の相対量であり、
    run 間の絶対比較には使えない点に注意 (HTML 側にも注記を出す)。
    """
    if not losses:
        return None
    threshold = losses[0] * fraction
    for iteration, loss in zip(iters, losses):
        if loss <= threshold:
            return iteration
    return None


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


def convert(run_dir: Path, out_dir: Path,
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
        written.append(write_loss_csv(out_dir / f"frame_{index:04d}.csv", rows))
        print(f"  frame_{index:04d}.csv  ({len(rows)} 点, "
              f"最終100iter平均 {_mean_last_window(rows):.6g})")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="4D ラン root、または単一シーンの run ディレクトリ")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="CSV の書き出し先。シーンは階層で表すのでここに含める")
    parser.add_argument("--scene", default=None, help="進捗表示に使うラベル (任意)")
    parser.add_argument("--loss_key", default="loss_total",
                        choices=["loss_total", "loss_l1", "loss_dssim", "psnr"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label = f"{args.scene}: " if args.scene else ""
    print(f"=== {label}{args.run_dir} -> {args.out_dir} ===")
    written = convert(args.run_dir, args.out_dir, args.loss_key)
    print(f"{len(written)} 本の CSV を書き出しました")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
