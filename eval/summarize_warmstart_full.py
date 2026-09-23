"""
eval/summarize_warmstart_full.py
300 フレーム実験の集計。フェーズ 2 の報告項目をそのまま出す。

使い方:
    python eval/summarize_warmstart_full.py \
        --warm_dir     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/loss_logs \
        --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full/loss_logs \
        --warm_run     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
        --baseline_run output/4DGS/neu3d/coffee_martini/baseline_neu3d_full

出す項目:
  1. 損失推移 (フレーム平均) を iter 500/1000/1500/2000 で
  2. 収束損失 (最終 100 iter 平均) を 50 フレームごとのブロック平均で
  3. ガウシアン数の推移 (warm-start)
  4. 所要時間

注意: 10 フレーム試行で確かめた通り、第 1 フレームは warm-start / baseline とも
SfM 点群から同一条件で学習するので、その差は CUDA ラスタライザ backward の
atomicAdd 非決定性によるノイズ下限になる。ブロック平均を読むときはこの下限を
下回る差を有意と見なさないこと。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_loss import load_series  # noqa: E402


def run_records(run_dir: Path) -> dict[int, dict]:
    manifest = run_dir / "frames_4d.json"
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {int(r["frame"]): r for r in payload.get("frames", [])}


def frame_seconds(record: dict) -> float | None:
    if "wall_seconds" in record:
        return float(record["wall_seconds"])
    telemetry = Path(record.get("output", "")) / "training_telemetry.json"
    if telemetry.is_file():
        payload = json.loads(telemetry.read_text(encoding="utf-8"))
        value = payload.get("training_wall_time_seconds")
        if value is not None:
            return float(value)
    return None


def loss_at(series: list[dict], iteration: int) -> float | None:
    """全フレームでその iteration の損失を平均する。"""
    values = [
        loss
        for entry in series
        for it, loss in zip(entry["iters"], entry["losses"])
        if it == iteration
    ]
    return sum(values) / len(values) if values else None


def block_mean(series: list[dict], start: int, end: int) -> float | None:
    """frame 番号 [start, end] (0 始まり表示) の収束損失を平均する。"""
    values = [e["final"] for e in series if start <= e["frame"] <= end]
    return sum(values) / len(values) if values else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warm_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--warm_run", type=Path, required=True)
    parser.add_argument("--baseline_run", type=Path, required=True)
    parser.add_argument("--block", type=int, default=50,
                        help="ブロック平均の幅 (既定 50 フレーム)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warm = load_series(args.warm_dir)
    base = load_series(args.baseline_dir)
    if not warm or not base:
        print("[エラー] frame_*.csv が読めません")
        return 1

    print(f"【損失推移（フレーム平均）】  warm {len(warm)} フレーム / "
          f"baseline {len(base)} フレーム")
    for probe in (500, 1000, 1500, 2000):
        print(f"  iter {probe:4d}: warm-start={fmt(loss_at(warm, probe))}, "
              f"baseline={fmt(loss_at(base, probe))}")

    print()
    print("【収束損失（最終100iter平均）のフレーム別推移】")
    highest = max(max(e["frame"] for e in warm), max(e["frame"] for e in base))
    start = 0
    while start <= highest:
        end = start + args.block - 1
        w, b = block_mean(warm, start, end), block_mean(base, start, end)
        if w is not None or b is not None:
            print(f"  frame_{start:03d}〜{min(end, highest):03d} 平均: "
                  f"warm-start={fmt(w)}, baseline={fmt(b)}")
        start += args.block
    all_w = sum(e["final"] for e in warm) / len(warm)
    all_b = sum(e["final"] for e in base) / len(base)
    print(f"  全体平均:            warm-start={fmt(all_w)}, baseline={fmt(all_b)}")

    warm_records = run_records(args.warm_run)
    base_records = run_records(args.baseline_run)

    print()
    print("【Gaussian 数の推移（warm-start)】")
    keys = sorted(warm_records)
    probes = [k for k in keys if (k - 1) % args.block == 0] + ([keys[-1]] if keys else [])
    seen = set()
    previous = None
    for key in probes:
        if key in seen:
            continue
        seen.add(key)
        value = warm_records[key].get("num_gaussians_end")
        delta = "" if previous is None else f"  (前回比 {value - previous:+,})"
        print(f"  frame_{key - 1:03d}: {value:,}{delta}")
        previous = value
    if len(keys) >= 2:
        tail = [warm_records[k]["num_gaussians_end"] for k in keys[-min(20, len(keys)):]]
        span = max(tail) - min(tail)
        print(f"  → 直近 {len(tail)} フレームの変動幅 {span:,} "
              f"({'飽和' if span < 0.02 * max(tail) else '増加中'})")
    if base_records:
        bvals = [r["num_gaussians_end"] for r in base_records.values()
                 if r.get("num_gaussians_end")]
        if bvals:
            print(f"  (参考) baseline: {min(bvals):,} 〜 {max(bvals):,} "
                  f"平均 {sum(bvals) // len(bvals):,}")

    print()
    print("【所要時間】")
    for label, records in (("warm-start", warm_records), ("baseline  ", base_records)):
        times = [t for t in (frame_seconds(r) for r in records.values()) if t is not None]
        if times:
            print(f"  {label} 合計: {sum(times) / 60:.1f}分 / "
                  f"1フレーム平均 {sum(times) / len(times):.1f}秒 "
                  f"({len(times)} フレーム)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
