"""
<...>/loss_logs/<variant>/frame_*.csv を読み、損失曲線の比較 HTML を生成する。

使い方:
    # warm-start ありのみ
    python eval/analysis/plot_loss.py \
        --log_dir  output/4DGS/dnerf/lego/experiments/warmstart/loss_logs \
        --out_html output/4DGS/dnerf/lego/loss_plots/lego.html

    # warm-start あり vs なし
    python eval/analysis/plot_loss.py \
        --log_dir      output/4DGS/dnerf/lego/experiments/warmstart/loss_logs \
        --baseline_dir output/4DGS/dnerf/lego/experiments/baseline/loss_logs \
        --out_html     output/4DGS/dnerf/lego/loss_plots/lego_compare.html

出力は単一の自己完結 HTML。Chart.js だけ CDN から読む。
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from export_loss_csv import load_series

CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"

# 色相を回して各フレームに色を割り当てる (フレーム数が可変のため固定表は使わない)
def _frame_color(index: int, total: int, alpha: float = 1.0) -> str:
    hue = (index * 360.0 / max(total, 1)) % 360.0
    return f"hsla({hue:.0f}, 70%, 45%, {alpha})"


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, float) and math.isnan(value):
        return "&mdash;"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def build_html(series: list[dict], baseline: list[dict] | None,
               title: str) -> str:
    total = len(series)
    curve_datasets = []
    for index, item in enumerate(series):
        curve_datasets.append({
            "label": f"{item['name']} (warm)" if baseline else item["name"],
            "data": [{"x": i, "y": l} for i, l in zip(item["iters"], item["losses"])],
            "borderColor": _frame_color(index, total),
            "backgroundColor": _frame_color(index, total, 0.25),
            "borderWidth": 1.6, "pointRadius": 0, "tension": 0.1,
        })
    if baseline:
        for index, item in enumerate(baseline):
            curve_datasets.append({
                "label": f"{item['name']} (baseline)",
                "data": [{"x": i, "y": l} for i, l in zip(item["iters"], item["losses"])],
                "borderColor": _frame_color(index, len(baseline), 0.55),
                "backgroundColor": "transparent",
                "borderWidth": 1.2, "borderDash": [5, 4],
                "pointRadius": 0, "tension": 0.1, "hidden": True,
            })

    labels = [item["name"] for item in series]
    base_by_name = {item["name"]: item for item in (baseline or [])}
    bar_datasets = [{
        "label": "warm-start",
        "data": [None if math.isnan(i["final"]) else i["final"] for i in series],
        "backgroundColor": "hsla(215, 75%, 50%, 0.85)",
    }]
    if baseline:
        bar_datasets.append({
            "label": "baseline (warm-start なし)",
            "data": [
                (lambda b: None if b is None or math.isnan(b["final"]) else b["final"])(
                    base_by_name.get(name))
                for name in labels
            ],
            "backgroundColor": "hsla(0, 0%, 55%, 0.85)",
        })

    rows = []
    for item in series:
        base = base_by_name.get(item["name"])
        cells = [
            f"<td>{html.escape(item['name'])}</td>",
            f"<td class='num'>{_fmt(item['final'])}</td>",
            f"<td class='num'>{_fmt(item['converged_iter'], 0)}</td>",
        ]
        if baseline:
            delta = None
            if base and not math.isnan(base["final"]) and not math.isnan(item["final"]) \
               and base["final"] != 0:
                delta = (item["final"] - base["final"]) / base["final"] * 100.0
            cls = "" if delta is None else (" good" if delta < 0 else " bad")
            cells += [
                f"<td class='num'>{_fmt(base['final']) if base else '&mdash;'}</td>",
                f"<td class='num'>{_fmt(base['converged_iter'], 0) if base else '&mdash;'}</td>",
                f"<td class='num{cls}'>{'&mdash;' if delta is None else f'{delta:+.1f}%'}</td>",
            ]
        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = ["フレーム", "収束損失 (warm)", "収束 iter (warm)"]
    if baseline:
        header += ["収束損失 (baseline)", "収束 iter (baseline)", "損失の変化"]

    payload = json.dumps(
        {"curves": curve_datasets, "barLabels": labels, "bars": bar_datasets},
        ensure_ascii=False,
    )

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
 h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:32px 0 10px; }}
 .sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
 .card {{ border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:20px; }}
 .wrap {{ position:relative; height:420px; }}
 .toggles {{ display:flex; flex-wrap:wrap; gap:10px 16px; margin:12px 0 0; font-size:13px; }}
 .toggles label {{ display:flex; align-items:center; gap:5px; cursor:pointer; }}
 .swatch {{ width:11px; height:11px; border-radius:2px; display:inline-block; }}
 .btns {{ margin:10px 0 0; display:flex; gap:8px; }}
 button {{ font:inherit; padding:4px 12px; border:1px solid var(--line);
           border-radius:6px; background:transparent; color:inherit; cursor:pointer; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
 th {{ font-weight:600; color:var(--muted); }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 td.good {{ color:#1a7f37; }} td.bad {{ color:#b3261e; }}
 .note {{ color:var(--muted); font-size:12px; margin-top:10px; }}
 .tablewrap {{ overflow-x:auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">損失曲線 (loss vs iteration)。収束損失は最終 100 iteration の平均。</div>

<div class="card">
  <h2 style="margin-top:0">グラフ 1: 全フレームの損失曲線</h2>
  <div class="wrap"><canvas id="curves"></canvas></div>
  <div class="btns"><button id="all">全表示</button><button id="none">全非表示</button></div>
  <div class="toggles" id="toggles"></div>
  <div class="note">Y 軸は対数スケール。{"破線が baseline (既定では非表示)。" if baseline else ""}</div>
</div>

<div class="card">
  <h2 style="margin-top:0">グラフ 2: フレームごとの収束損失</h2>
  <div class="wrap"><canvas id="bars"></canvas></div>
</div>

<div class="card">
  <h2 style="margin-top:0">フレーム別サマリー</h2>
  <div class="tablewrap"><table>
    <thead><tr>{"".join(f"<th>{h}</th>" for h in header)}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <div class="note">「収束 iter」は損失がその run の初期値の 10% 以下へ最初に落ちた iteration。
  warm-start では初期損失自体が低いため、この値は run 内の相対的な下げ幅を表すもので、
  warm/baseline 間の絶対比較には使えない。速度の比較はグラフ 1 の曲線そのものを見ること。</div>
</div>

<script>
const D = {payload};
const curves = new Chart(document.getElementById('curves'), {{
  type:'line', data:{{datasets:D.curves}},
  options:{{responsive:true, maintainAspectRatio:false, animation:false,
    interaction:{{mode:'nearest', intersect:false}},
    plugins:{{legend:{{display:false}}}},
    scales:{{ x:{{type:'linear', title:{{display:true, text:'iteration'}}}},
              y:{{type:'logarithmic', title:{{display:true, text:'loss'}}}} }} }}
}});
const box = document.getElementById('toggles');
D.curves.forEach((ds, i) => {{
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = !ds.hidden;
  cb.onchange = () => {{ curves.setDatasetVisibility(i, cb.checked); curves.update(); }};
  const sw = document.createElement('span');
  sw.className = 'swatch'; sw.style.background = ds.borderColor;
  label.append(cb, sw, document.createTextNode(ds.label));
  box.appendChild(label);
}});
function setAll(v) {{
  D.curves.forEach((_, i) => curves.setDatasetVisibility(i, v));
  box.querySelectorAll('input').forEach(cb => cb.checked = v);
  curves.update();
}}
document.getElementById('all').onclick = () => setAll(true);
document.getElementById('none').onclick = () => setAll(false);

new Chart(document.getElementById('bars'), {{
  type:'bar', data:{{labels:D.barLabels, datasets:D.bars}},
  options:{{responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{{legend:{{position:'top'}}}},
    scales:{{ y:{{title:{{display:true, text:'収束損失 (最終100iterの平均)'}}}} }} }}
}});
</script></body></html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, default=None)
    parser.add_argument("--out_html", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    series = load_series(args.log_dir)
    if not series:
        print(f"[エラー] {args.log_dir} に frame_*.csv がありません")
        return 1
    baseline = load_series(args.baseline_dir) if args.baseline_dir else None
    title = args.title or f"損失曲線: {args.log_dir.name}" + (
        " (warm-start vs baseline)" if baseline else "")
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(build_html(series, baseline, title), encoding="utf-8")
    print(f"  warm  {len(series)} フレーム" + (f" / baseline {len(baseline)} フレーム" if baseline else ""))
    print(f"  -> {args.out_html} ({args.out_html.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
