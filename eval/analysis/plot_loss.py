"""
eval/analysis/plot_loss.py
``loss_logs/frame_*.csv`` (export_loss_csv.py の出力) を読み、損失曲線を 1 枚の HTML にする。

``--run`` を繰り返すと、指定したランをすべて同じ図に重ねる。ランの本数と
ラベルは自由で、どのランも同じ扱い (基準ランや差分の列は持たない)。

使い方:
    # 1 ラン
    python eval/analysis/plot_loss.py \
        --run output/4DGS/neu3d/coffee_martini/baseline_30k/loss_logs \
        --out_html output/4DGS/neu3d/coffee_martini/loss_plots/baseline_30k.html

    # 任意のランを重ねる (ラベル=パス。ラベル省略時はランのディレクトリ名)
    python eval/analysis/plot_loss.py \
        --run "30k=output/4DGS/neu3d/coffee_martini/baseline_30k/loss_logs" \
        --run "7k=output/4DGS/neu3d/coffee_martini/baseline_7k/loss_logs" \
        --run "warm 250=output/4DGS/neu3d/cook_spinach/warmstart_250_300f/loss_logs" \
        --frames 0 50 100 \
        --out_html output/4DGS/neu3d/loss_plots/overlay.html

``--frames`` は ``frame_NNNN.csv`` の番号で描くフレームを絞る (既定は全フレーム)。
ランごとに色を分け、同じランのフレームは同じ色で描く。表示/非表示はラン単位で切り替えられる。

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

def _run_color(index: int, total: int, alpha: float = 1.0) -> str:
    """ランごとに色相を回して割り当てる (ラン数が可変のため固定表は使わない)。"""
    hue = (215 + index * 360.0 / max(total, 1)) % 360.0
    return f"hsla({hue:.0f}, 70%, 45%, {alpha})"


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, float) and math.isnan(value):
        return "&mdash;"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def build_html(runs: list[tuple[str, list[dict]]], title: str) -> str:
    """``runs`` は (ラベル, load_series の結果) の並び。"""
    total = len(runs)
    curve_datasets = []
    for run_index, (label, series) in enumerate(runs):
        # 同じランのフレームは同じ色。フレームが多いほど細く薄くして重なりを読めるようにする。
        alpha = 1.0 if len(series) <= 5 else 0.55
        for item in series:
            curve_datasets.append({
                "label": f"{label} / {item['name']}",
                "run": run_index,
                "data": [{"x": i, "y": l} for i, l in zip(item["iters"], item["losses"])],
                "borderColor": _run_color(run_index, total, alpha),
                "backgroundColor": _run_color(run_index, total, 0.25),
                "borderWidth": 1.6 if len(series) <= 5 else 1.0,
                "pointRadius": 0, "tension": 0.1,
            })
    run_legend = [{"label": label, "color": _run_color(i, total)} for i, (label, _) in enumerate(runs)]

    frame_names = sorted({item["name"] for _, series in runs for item in series})
    by_run = [{item["name"]: item for item in series} for _, series in runs]
    bar_datasets = [{
        "label": label,
        "data": [
            None if (item := lookup.get(name)) is None or math.isnan(item["final"]) else item["final"]
            for name in frame_names
        ],
        "backgroundColor": _run_color(i, total, 0.85),
    } for i, ((label, _), lookup) in enumerate(zip(runs, by_run))]

    rows = []
    for name in frame_names:
        cells = [f"<td>{html.escape(name)}</td>"]
        for lookup in by_run:
            item = lookup.get(name)
            cells += [
                f"<td class='num'>{_fmt(item['final']) if item else '&mdash;'}</td>",
                f"<td class='num'>{_fmt(item['converged_iter'], 0) if item else '&mdash;'}</td>",
            ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = ["フレーム"]
    for label, _ in runs:
        header += [f"収束損失 ({html.escape(label)})", f"収束 iter ({html.escape(label)})"]

    payload = json.dumps(
        {"curves": curve_datasets, "runs": run_legend,
         "barLabels": frame_names, "bars": bar_datasets},
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
 .note {{ color:var(--muted); font-size:12px; margin-top:10px; }}
 .tablewrap {{ overflow-x:auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">損失曲線 (loss vs iteration)。収束損失は最終 100 iteration の平均。</div>

<div class="card">
  <h2 style="margin-top:0">グラフ 1: 損失曲線</h2>
  <div class="wrap"><canvas id="curves"></canvas></div>
  <div class="btns"><button id="all">全表示</button><button id="none">全非表示</button></div>
  <div class="toggles" id="toggles"></div>
  <div class="note">Y 軸は対数スケール。色はラン、チェックボックスでラン単位に表示を切り替える。</div>
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
  初期損失が違う run 同士 (例: warm-start とそうでないもの) では、この値は run 内の相対的な
  下げ幅を表すだけで、run 間の絶対比較には使えない。速度の比較はグラフ 1 の曲線そのものを見ること。</div>
</div>

<script>
const D = {payload};
const curves = new Chart(document.getElementById('curves'), {{
  type:'line', data:{{datasets:D.curves}},
  options:{{responsive:true, maintainAspectRatio:false, animation:false,
    interaction:{{mode:'nearest', intersect:false}},
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{title:(items) => items.length ? items[0].dataset.label : ''}}}}}},
    scales:{{ x:{{type:'linear', title:{{display:true, text:'iteration'}}}},
              y:{{type:'logarithmic', title:{{display:true, text:'loss'}}}} }} }}
}});
const box = document.getElementById('toggles');
function setRun(run, v) {{
  D.curves.forEach((ds, i) => {{ if (ds.run === run) curves.setDatasetVisibility(i, v); }});
}}
D.runs.forEach((r, run) => {{
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = true;
  cb.onchange = () => {{ setRun(run, cb.checked); curves.update(); }};
  const sw = document.createElement('span');
  sw.className = 'swatch'; sw.style.background = r.color;
  const n = D.curves.filter(ds => ds.run === run).length;
  label.append(cb, sw, document.createTextNode(`${{r.label}} (${{n}} フレーム)`));
  box.appendChild(label);
}});
function setAll(v) {{
  D.runs.forEach((_, run) => setRun(run, v));
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


def parse_run(text: str) -> tuple[str, Path]:
    """``ラベル=パス`` か ``パス``。ラベル省略時はランのディレクトリ名を使う。"""
    label, sep, path = text.partition("=")
    if not sep:
        path_obj = Path(text)
        run_dir = path_obj.parent if path_obj.name == "loss_logs" else path_obj
        return run_dir.name, path_obj
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--run は ラベル=パス の形で指定してください: {text!r}")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=parse_run, action="append", required=True,
                        metavar="[LABEL=]LOG_DIR",
                        help="loss_logs ディレクトリ。繰り返すと同じ図に重ねる")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="描く frame_NNNN.csv の番号 (既定: 全フレーム)")
    parser.add_argument("--out_html", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        print(f"[エラー] ラベルが重複しています: {labels}  (ラベル=パス で区別してください)")
        return 1

    runs: list[tuple[str, list[dict]]] = []
    for label, log_dir in args.run:
        series = load_series(log_dir)
        if args.frames is not None:
            wanted = set(args.frames)
            series = [item for item in series if item["frame"] in wanted]
        if not series:
            print(f"[エラー] {label}: {log_dir} に対象の frame_*.csv がありません")
            return 1
        runs.append((label, series))
        print(f"  {label}: {len(series)} フレーム ({log_dir})")

    title = args.title or "損失曲線: " + " / ".join(labels)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(build_html(runs, title), encoding="utf-8")
    print(f"  -> {args.out_html} ({args.out_html.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
