"""
eval/analysis/plot_gaussian_count.py
各フレームの training_telemetry.json から Gaussian 数・学習時間・VRAM を集め、
集計の表示・CSV の書き出し・推移の図 (HTML) を作る。

使い方:
    # 1 ラン: 集計表示 + CSV (runs/4DGS_*.sh の評価工程から自動で呼ばれる形)
    python eval/analysis/plot_gaussian_count.py \
        --run output/4DGS/neu3d/coffee_martini/baseline_30k \
        --block 10 --output_csv output/4DGS/neu3d/coffee_martini/baseline_30k/results/gaussian_counts.csv

    # 任意のランを 1 枚の図に重ねる (ラベル=パス。ラベル省略時はランのディレクトリ名)
    python eval/analysis/plot_gaussian_count.py \
        --run "30k=output/4DGS/neu3d/coffee_martini/baseline_30k" \
        --run "warm 250=output/4DGS/neu3d/cook_spinach/warmstart_250_300f" \
        --out_html output/4DGS/neu3d/gaussian_plots/overlay.html

図は横軸がフレーム、縦軸は Gaussian 数 / 学習時間 / VRAM をボタンで切り替える。
ランごとに色を分け、どのランも同じ扱い (基準ランや差分は持たない)。
``--output_csv`` は 1 ラン分の形式なので、``--run`` が 1 本のときだけ使える。

baseline は毎フレームを SfM 点群から独立に学習するので、Gaussian 数の
フレーム間の振れがそのまま「密度制御の再現性」を表す。warm-start と違って
前フレームを引き継がないぶん、ここが大きく振れるほどフレーム間で
別々の表現に収束していることになる。

出力の HTML は単一の自己完結ファイル。Chart.js だけ CDN から読む。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean, pstdev

CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"
CSV_FIELDS = ["frame", "gaussian_count", "iteration", "wall_seconds", "peak_cuda_MiB"]
# 図で切り替えられる縦軸: (CSV の列, ボタン/軸の表示名)
METRICS = [
    ("gaussian_count", "Gaussian 数"),
    ("wall_seconds", "学習時間 (秒)"),
    ("peak_cuda_MiB", "VRAM ピーク (MiB)"),
]


def collect(run_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("frame_*/training_telemetry.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") != "COMPLETED":
            continue
        count = data.get("gaussian_count")
        if not isinstance(count, int):
            continue
        records.append(
            {
                "frame": int(path.parent.name.split("_")[1]),
                "gaussian_count": count,
                "iteration": data.get("iteration"),
                "wall_seconds": data.get("training_wall_time_seconds"),
                "peak_cuda_MiB": data.get("peak_cuda_memory_allocated_MiB"),
            }
        )
    return records


# ------------------------------------------------------------------ 集計表示
def print_summary(run_dir: Path, records: list[dict[str, object]], block: int) -> None:
    counts = [int(r["gaussian_count"]) for r in records]
    times = [float(r["wall_seconds"] or 0.0) for r in records]

    print("=" * 70)
    print(f"  Gaussian 数の推移  |  {run_dir}")
    print("=" * 70)
    print(f"  完了フレーム : {len(records)}")
    print(f"  最小         : {min(counts):,}  (frame "
          f"{records[counts.index(min(counts))]['frame']:04d})")
    print(f"  最大         : {max(counts):,}  (frame "
          f"{records[counts.index(max(counts))]['frame']:04d})")
    print(f"  平均         : {int(mean(counts)):,}")
    print(f"  標準偏差     : {pstdev(counts):,.0f}"
          f"  (平均比 {100 * pstdev(counts) / mean(counts):.2f}%)")
    print(f"  振れ幅       : {max(counts) - min(counts):,} "
          f"({100 * (max(counts) - min(counts)) / min(counts):.2f}%)")
    if any(times):
        print(f"  学習時間     : 平均 {mean(times):.0f}s / 合計 "
              f"{sum(times) / 3600:.1f} 時間")

    if block > 0 and len(records) > block:
        print()
        print(f"  --- {block} フレームブロックごと ---")
        print(f"  {'frames':<14}{'n':>5}{'平均':>12}{'最小':>12}{'最大':>12}")
        print("  " + "-" * 55)
        lowest = min(int(r["frame"]) for r in records)
        highest = max(int(r["frame"]) for r in records)
        start = lowest - ((lowest - 1) % block)
        while start <= highest:
            stop = start + block - 1
            subset = [
                int(r["gaussian_count"]) for r in records
                if start <= int(r["frame"]) <= stop
            ]
            if subset:
                label = f"{start}-{stop}"
                print(f"  {label:<14}{len(subset):>5}{int(mean(subset)):>12,}"
                      f"{min(subset):>12,}{max(subset):>12,}")
            start += block


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------- 図
def _run_color(index: int, total: int, alpha: float = 1.0) -> str:
    """ランごとに色相を回して割り当てる (plot_loss.py と同じ配色規則)。"""
    hue = (215 + index * 360.0 / max(total, 1)) % 360.0
    return f"hsla({hue:.0f}, 70%, 45%, {alpha})"


def _stats_row(label: str, records: list[dict[str, object]]) -> str:
    counts = [int(r["gaussian_count"]) for r in records]
    times = [float(r["wall_seconds"]) for r in records if r["wall_seconds"] is not None]
    memory = [float(r["peak_cuda_MiB"]) for r in records if r["peak_cuda_MiB"] is not None]
    cells = [
        html.escape(label), f"{len(records)}",
        f"{min(counts):,}", f"{max(counts):,}", f"{int(mean(counts)):,}", f"{pstdev(counts):,.0f}",
        f"{mean(times):.0f}" if times else "&mdash;",
        f"{sum(times) / 3600:.1f}" if times else "&mdash;",
        f"{max(memory):,.0f}" if memory else "&mdash;",
    ]
    return "<tr>" + "".join(
        f"<td{'' if i == 0 else ' class=num'}>{c}</td>" for i, c in enumerate(cells)
    ) + "</tr>"


def build_html(runs: list[tuple[str, list[dict[str, object]]]], title: str) -> str:
    total = len(runs)
    series = [{
        "label": label,
        "color": _run_color(i, total),
        "points": {
            key: [{"x": r["frame"], "y": r[key]} for r in records if r[key] is not None]
            for key, _ in METRICS
        },
    } for i, (label, records) in enumerate(runs)]
    payload = json.dumps(
        {"series": series, "metrics": [{"key": k, "label": v} for k, v in METRICS]},
        ensure_ascii=False,
    )
    rows = "".join(_stats_row(label, records) for label, records in runs)
    header = ["ラン", "フレーム数", "Gaussian 最小", "最大", "平均", "標準偏差",
              "学習時間 平均 (秒)", "合計 (時間)", "VRAM ピーク最大 (MiB)"]

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<script src="{CHART_JS}"></script>
<style>
 :root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#fbfbfa; --line:#d8d8d4; --muted:#666; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e8e8e6; --bg:#191918; --line:#3a3a38; --muted:#a0a09c; }} }}
 body {{ margin:0; padding:28px; background:var(--bg); color:var(--fg);
        font:14px/1.6 system-ui,-apple-system,"Helvetica Neue",sans-serif; }}
 h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:0 0 10px; }}
 .sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
 .card {{ border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:20px; }}
 .wrap {{ position:relative; height:420px; }}
 .btns {{ margin:0 0 12px; display:flex; flex-wrap:wrap; gap:8px; }}
 button {{ font:inherit; padding:4px 12px; border:1px solid var(--line);
           border-radius:6px; background:transparent; color:inherit; cursor:pointer; }}
 button[aria-pressed="true"] {{ border-color:currentColor; font-weight:600; }}
 .toggles {{ display:flex; flex-wrap:wrap; gap:10px 16px; margin:12px 0 0; font-size:13px; }}
 .toggles label {{ display:flex; align-items:center; gap:5px; cursor:pointer; }}
 .swatch {{ width:11px; height:11px; border-radius:2px; display:inline-block; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; white-space:nowrap; }}
 th {{ font-weight:600; color:var(--muted); }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .tablewrap {{ overflow-x:auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">各フレームの training_telemetry.json (COMPLETED のフレームのみ)。</div>

<div class="card">
  <h2>フレームごとの推移</h2>
  <div class="btns" id="metrics"></div>
  <div class="wrap"><canvas id="chart"></canvas></div>
  <div class="toggles" id="toggles"></div>
</div>

<div class="card">
  <h2>ラン別サマリー</h2>
  <div class="tablewrap"><table>
    <thead><tr>{"".join(f"<th>{h}</th>" for h in header)}</tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>

<script>
const D = {payload};
let metric = D.metrics[0];
const chart = new Chart(document.getElementById('chart'), {{
  type:'line', data:{{datasets:[]}},
  options:{{responsive:true, maintainAspectRatio:false, animation:false,
    interaction:{{mode:'nearest', intersect:false}},
    plugins:{{legend:{{display:false}}}},
    scales:{{ x:{{type:'linear', title:{{display:true, text:'frame'}}}},
              y:{{title:{{display:true, text:''}}}} }} }}
}});
function render() {{
  const hidden = chart.data.datasets.map((_, i) => !chart.isDatasetVisible(i));
  chart.data.datasets = D.series.map((s, i) => ({{
    label:s.label, data:s.points[metric.key], borderColor:s.color, backgroundColor:s.color,
    borderWidth:1.6, pointRadius:0, tension:0.1, hidden:hidden[i] || false,
  }}));
  chart.options.scales.y.title.text = metric.label;
  chart.update();
  document.querySelectorAll('#metrics button').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.key === metric.key));
}}
const bar = document.getElementById('metrics');
D.metrics.forEach(m => {{
  const b = document.createElement('button');
  b.textContent = m.label; b.dataset.key = m.key;
  b.onclick = () => {{ metric = m; render(); }};
  bar.appendChild(b);
}});
const box = document.getElementById('toggles');
D.series.forEach((s, i) => {{
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = true;
  cb.onchange = () => {{ chart.setDatasetVisibility(i, cb.checked); chart.update(); }};
  const sw = document.createElement('span');
  sw.className = 'swatch'; sw.style.background = s.color;
  label.append(cb, sw, document.createTextNode(s.label));
  box.appendChild(label);
}});
render();
</script></body></html>
"""


# --------------------------------------------------------------------- CLI
def parse_run(text: str) -> tuple[str, Path]:
    """``ラベル=パス`` か ``パス``。ラベル省略時はランのディレクトリ名を使う。"""
    label, sep, path = text.partition("=")
    if not sep:
        return Path(text).name, Path(text)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--run は ラベル=パス の形で指定してください: {text!r}")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=parse_run, action="append", required=True,
                        metavar="[LABEL=]RUN_DIR",
                        help="frame_*/training_telemetry.json を持つラン。繰り返すと同じ図に重ねる")
    parser.add_argument("--block", type=int, default=10,
                        help="集計表示のブロック幅 (フレーム数)。0 でブロック表示なし")
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="gaussian_counts.csv の書き出し先 (--run が 1 本のときのみ)")
    parser.add_argument("--out_html", type=Path, default=None, help="推移の図 (HTML) の書き出し先")
    parser.add_argument("--title", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_csv is not None and len(args.run) != 1:
        parser.error("--output_csv は --run が 1 本のときだけ指定できます")
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        parser.error(f"ラベルが重複しています: {labels}  (ラベル=パス で区別してください)")

    runs: list[tuple[str, list[dict[str, object]]]] = []
    for index, (label, run_dir) in enumerate(args.run):
        records = collect(run_dir)
        if not records:
            print(f"[エラー] {run_dir} に完了フレームがありません")
            return 1
        if index:
            print()
        print_summary(run_dir, records, args.block)
        runs.append((label, records))

    if args.output_csv:
        write_csv(runs[0][1], args.output_csv)
        print(f"\n  CSV: {args.output_csv}")
    if args.out_html:
        title = args.title or "Gaussian 数の推移: " + " / ".join(labels)
        args.out_html.parent.mkdir(parents=True, exist_ok=True)
        args.out_html.write_text(build_html(runs, title), encoding="utf-8")
        print(f"\n  HTML: {args.out_html} ({args.out_html.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
