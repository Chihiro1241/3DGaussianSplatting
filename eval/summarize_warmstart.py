"""
eval/summarize_warmstart.py
warm-start あり / なし の収束を数値で突き合わせる。

使い方:
    python eval/summarize_warmstart.py \
        --warm_dir     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial/loss_logs \
        --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial/loss_logs \
        --warm_run     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial \
        --baseline_run output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial

出す指標:
  * 収束損失 = 最終 100 iter の平均 (plot_loss.py と同じ定義)
  * 収束 iter = 損失がその run の初期値の 10% 以下へ最初に落ちた iteration
  * フレームあたりの所要時間とガウシアン数

**収束 iter の読み方に注意**: これは run 内の相対量であり、run 間で直接
比較してはいけない。warm-start は初期損失そのものが低いので、同じ「10% に
落ちる」でも意味する仕事量が違う。run 間で比べてよいのは収束損失と所要時間。
plot_loss.py も HTML 側に同じ注記を出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_loss import convergence_iteration, load_series  # noqa: E402


def run_records(run_dir: Path) -> dict[int, dict]:
    """frames_4d.json をフレーム番号で引ける形にする。"""
    manifest = run_dir / "frames_4d.json"
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {int(r["frame"]): r for r in payload.get("frames", [])}


def frame_seconds(record: dict) -> float | None:
    """そのフレームの所要時間。4D 側は telemetry を読みに行く。"""
    if "wall_seconds" in record:
        return float(record["wall_seconds"])
    telemetry = Path(record.get("output", "")) / "training_telemetry.json"
    if telemetry.is_file():
        payload = json.loads(telemetry.read_text(encoding="utf-8"))
        value = payload.get("training_wall_time_seconds")
        if value is not None:
            return float(value)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warm_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--warm_run", type=Path, default=None)
    parser.add_argument("--baseline_run", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="収束 iter のしきい値 (初期損失に対する比)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warm = load_series(args.warm_dir)
    base = load_series(args.baseline_dir)
    if not warm or not base:
        print("[エラー] frame_*.csv が読めません")
        return 1

    warm_records = run_records(args.warm_run) if args.warm_run else {}
    base_records = run_records(args.baseline_run) if args.baseline_run else {}

    print("フレームごとの収束損失 (最終 100 iter 平均):")
    wins = 0
    for index, (w, b) in enumerate(zip(warm, base)):
        difference = w["final"] - b["final"]
        if difference < 0:
            wins += 1
        mark = "warm↓" if difference < 0 else "base↓"
        print(f"  frame_{index:03d}: warm-start={w['final']:.4f}, "
              f"baseline={b['final']:.4f}, 差={difference:+.4f}  {mark}")

    print()
    print(f"warm-start が baseline より損失が低いフレーム数: {wins}/{len(warm)}")

    warm_iters = [convergence_iteration(w["iters"], w["losses"], args.fraction)
                  for w in warm]
    base_iters = [convergence_iteration(b["iters"], b["losses"], args.fraction)
                  for b in base]

    def mean(values: list[int | None]) -> str:
        present = [v for v in values if v is not None]
        if not present:
            return "到達せず"
        suffix = "" if len(present) == len(values) else \
            f" ({len(present)}/{len(values)} フレームのみ到達)"
        return f"{sum(present) / len(present):.0f}{suffix}"

    print(f"warm-start の平均収束 iter (初期損失の {args.fraction:.0%}): "
          f"{mean(warm_iters)}")
    print(f"baseline   の平均収束 iter (初期損失の {args.fraction:.0%}): "
          f"{mean(base_iters)}")
    print("  ※ この指標は run 内の相対量。warm-start は初期損失自体が低いため、")
    print("     run 間の直接比較には使えない。")

    warm_times = [frame_seconds(warm_records[k]) for k in sorted(warm_records)]
    base_times = [frame_seconds(base_records[k]) for k in sorted(base_records)]
    warm_times = [t for t in warm_times if t is not None]
    base_times = [t for t in base_times if t is not None]
    if warm_times and base_times:
        print()
        print(f"1 フレームあたりの平均所要時間: "
              f"warm-start={sum(warm_times) / len(warm_times):.1f}秒, "
              f"baseline={sum(base_times) / len(base_times):.1f}秒")

    if warm_records and base_records:
        print()
        print("ガウシアン数 (フレーム終了時):")
        for key in sorted(set(warm_records) & set(base_records)):
            print(f"  frame {key}: warm-start="
                  f"{warm_records[key].get('num_gaussians_end')}, "
                  f"baseline={base_records[key].get('num_gaussians_end')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
