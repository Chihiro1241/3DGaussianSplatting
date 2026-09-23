"""
eval/summarize_warmstart.py
warm-start あり / なし の収束を数値で突き合わせる。

使い方:
    python eval/summarize_warmstart.py \
        --warm_dir     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/loss_logs \
        --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full/loss_logs \
        --warm_run     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
        --baseline_run output/4DGS/neu3d/coffee_martini/baseline_neu3d_full

出す項目:
  1. 損失推移 (フレーム平均) を iter 500/1000/1500/2000 で
  2. 収束損失 (最終 100 iter 平均) をフレーム別またはブロック平均で
  3. 収束 iter (損失が初期値の 10% へ最初に落ちた iteration)
  4. ガウシアン数の推移
  5. 所要時間

``--block`` が集計の粒度を決める。既定の ``auto`` は 20 フレーム以下なら
フレーム別に全部出し、それより長いランでは 50 フレームごとのブロック平均にする
(10 フレーム試行と 300 フレーム本番のどちらでも読める分量になるように)。
``--block 0`` で常にフレーム別、``--block N`` で N フレームごと。

読み方の注意が 2 つある。

* **収束 iter は run 内の相対量**であり、run 間で直接比較してはいけない。
  warm-start は初期損失そのものが低いので、同じ「10% に落ちる」でも意味する
  仕事量が違う。run 間で比べてよいのは収束損失と所要時間。plot_loss.py も
  HTML 側に同じ注記を出す。
* 第 1 フレームは warm-start / baseline とも SfM 点群から同一条件で学習するので、
  その差は CUDA ラスタライザ backward の atomicAdd 非決定性によるノイズ下限に
  なる。ブロック平均を読むときはこの下限を下回る差を有意と見なさないこと。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_loss import convergence_iteration, load_series  # noqa: E402

AUTO_BLOCK_THRESHOLD = 20   # これ以下のフレーム数ならフレーム別に全部出す
DEFAULT_BLOCK = 50


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


def mean_iter(values: list[int | None]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "到達せず"
    suffix = "" if len(present) == len(values) else \
        f" ({len(present)}/{len(values)} フレームのみ到達)"
    return f"{sum(present) / len(present):.0f}{suffix}"


def resolve_block(spec: str, frame_count: int) -> int:
    """--block の指定を実際の幅に落とす。0 はフレーム別の意味。"""
    if spec == "auto":
        return 0 if frame_count <= AUTO_BLOCK_THRESHOLD else DEFAULT_BLOCK
    try:
        value = int(spec)
    except ValueError:
        raise SystemExit(f"[中止] --block は auto か整数: {spec!r}")
    if value < 0:
        raise SystemExit("[中止] --block は 0 以上")
    return value


def print_convergence_loss(warm: list[dict], base: list[dict], block: int) -> None:
    """収束損失をフレーム別 (block=0) かブロック平均で並べる。"""
    print("【収束損失（最終100iter平均）】")
    if block == 0:
        wins = 0
        for index, (w, b) in enumerate(zip(warm, base)):
            difference = w["final"] - b["final"]
            wins += difference < 0
            mark = "warm↓" if difference < 0 else "base↓"
            print(f"  frame_{index:03d}: warm-start={w['final']:.4f}, "
                  f"baseline={b['final']:.4f}, 差={difference:+.4f}  {mark}")
        print(f"  warm-start が baseline より損失が低いフレーム数: "
              f"{wins}/{min(len(warm), len(base))}")
    else:
        highest = max(max(e["frame"] for e in warm), max(e["frame"] for e in base))
        start = 0
        while start <= highest:
            end = start + block - 1
            w, b = block_mean(warm, start, end), block_mean(base, start, end)
            if w is not None or b is not None:
                print(f"  frame_{start:03d}〜{min(end, highest):03d} 平均: "
                      f"warm-start={fmt(w)}, baseline={fmt(b)}")
            start += block
    all_w = sum(e["final"] for e in warm) / len(warm)
    all_b = sum(e["final"] for e in base) / len(base)
    print(f"  全体平均: warm-start={fmt(all_w)}, baseline={fmt(all_b)}")


def print_gaussian_counts(warm_records: dict[int, dict],
                          base_records: dict[int, dict], block: int) -> None:
    """ガウシアン数。フレーム別か、block ごとに間引いて出す。"""
    if not warm_records:
        return
    print()
    print("【Gaussian 数の推移（warm-start）】")
    keys = sorted(warm_records)
    if block == 0:
        probes = keys
    else:
        probes = [k for k in keys if (k - 1) % block == 0] + [keys[-1]]
    previous = None
    for key in dict.fromkeys(probes):          # 重複を除きつつ順序を保つ
        value = warm_records[key].get("num_gaussians_end")
        if value is None:
            continue
        delta = "" if previous is None else f"  (前回比 {value - previous:+,})"
        line = f"  frame_{key - 1:03d}: {value:,}{delta}"
        if block == 0 and key in base_records:
            line += f"   baseline={base_records[key].get('num_gaussians_end'):,}"
        print(line)
        previous = value
    if len(keys) >= 2:
        tail = [warm_records[k]["num_gaussians_end"] for k in keys[-min(20, len(keys)):]]
        span = max(tail) - min(tail)
        print(f"  → 直近 {len(tail)} フレームの変動幅 {span:,} "
              f"({'飽和' if span < 0.02 * max(tail) else '増加中'})")
    if base_records and block != 0:
        bvals = [r["num_gaussians_end"] for r in base_records.values()
                 if r.get("num_gaussians_end")]
        if bvals:
            print(f"  (参考) baseline: {min(bvals):,} 〜 {max(bvals):,} "
                  f"平均 {sum(bvals) // len(bvals):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warm_dir", type=Path, required=True,
                        help="warm-start 側の loss_logs ディレクトリ")
    parser.add_argument("--baseline_dir", type=Path, required=True,
                        help="baseline 側の loss_logs ディレクトリ")
    parser.add_argument("--warm_run", type=Path, default=None,
                        help="warm-start のラン (所要時間とガウシアン数に使う)")
    parser.add_argument("--baseline_run", type=Path, default=None)
    parser.add_argument("--block", default="auto",
                        help=f"集計の粒度。auto (既定, {AUTO_BLOCK_THRESHOLD} フレーム以下なら"
                             f"フレーム別/以上なら {DEFAULT_BLOCK}) / 0 (フレーム別) / N")
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
    block = resolve_block(args.block, min(len(warm), len(base)))

    print(f"【損失推移（フレーム平均）】  warm {len(warm)} フレーム / "
          f"baseline {len(base)} フレーム")
    for probe in (500, 1000, 1500, 2000):
        print(f"  iter {probe:4d}: warm-start={fmt(loss_at(warm, probe))}, "
              f"baseline={fmt(loss_at(base, probe))}")

    print()
    print_convergence_loss(warm, base, block)

    print()
    print(f"【収束 iter（初期損失の {args.fraction:.0%} へ落ちた iteration）】")
    print(f"  warm-start の平均: "
          f"{mean_iter([convergence_iteration(w['iters'], w['losses'], args.fraction) for w in warm])}")
    print(f"  baseline   の平均: "
          f"{mean_iter([convergence_iteration(b['iters'], b['losses'], args.fraction) for b in base])}")
    print("  ※ この指標は run 内の相対量。warm-start は初期損失自体が低いため、")
    print("     run 間の直接比較には使えない。")

    warm_records = run_records(args.warm_run) if args.warm_run else {}
    base_records = run_records(args.baseline_run) if args.baseline_run else {}
    print_gaussian_counts(warm_records, base_records, block)

    if warm_records or base_records:
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
