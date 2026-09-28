"""
eval/plot/plot_loss.py
各フレームの ``train_log.jsonl`` から損失の推移を読み、1 枚の HTML にする。

``--run`` を繰り返すと、指定したランをすべて同じ図に重ねる。ランの本数と
ラベルは自由で、どのランも同じ扱い (基準ランや差分の列は持たない)。

使い方:
    # 1 ラン
    python eval/plot/plot_loss.py \
        --run output/4DGS/neu3d/coffee_martini/baseline_30k \
        --out_html output/4DGS/neu3d/coffee_martini/loss_plots/baseline_30k.html

    # 任意のランを重ねる (ラベル=パス。ラベル省略時はランのディレクトリ名)
    python eval/plot/plot_loss.py \
        --run "30k=output/4DGS/neu3d/coffee_martini/baseline_30k" \
        --run "7k=output/4DGS/neu3d/coffee_martini/baseline_7k" \
        --run "warm 250=output/4DGS/neu3d/cook_spinach/warmstart_250_300f" \
        --frames 1 51 101 \
        --out_html output/4DGS/neu3d/loss_plots/overlay.html

    # 数値も欲しいとき: フレームごとの CSV を書き出す (--run が 1 本のときのみ)
    python eval/plot/plot_loss.py --run output/4DGS/neu3d/coffee_martini/baseline_30k \
        --output_csv_dir output/4DGS/neu3d/coffee_martini/baseline_30k/loss_logs

``--run`` は 4D ラン root (``frame_NNNN/train_log.jsonl`` を持つ) でも、単一シーンの
run ディレクトリ (``train_log.jsonl`` を直下に持つ) でもよい。``--frames`` は
実際のフレーム番号 (``frame_NNNN`` の NNNN) で描くフレームを絞る。単一シーンは frame 0 扱い。
``--loss_key`` で描く値を選ぶ (既定 loss_total)。ランごとに色を分け、同じランの
フレームは同じ色で描く。表示/非表示はラン単位で切り替えられる。

CSV は ``<dir>/frame_NNNN.csv`` (iter と値の 2 列)。最終行に ``*** LAST100_MEAN ***``
行 (最後に記録された点から遡って 100 iteration 分の平均) を追記する。

出力の HTML は単一の自己完結ファイル。Chart.js だけ CDN から読む。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"

LOSS_KEYS = ("loss_total", "loss_l1", "loss_dssim", "psnr")
# 値が大きいほど良いもの。収束 iter (初期値の 10% へ落ちた iteration) は定義できず、Y 軸も線形にする。
HIGHER_IS_BETTER = {"psnr"}
SUMMARY_MARKER = "*** LAST100_MEAN ***"


# ------------------------------------------------------------ 学習ログの読み込み
def frame_logs(run_dir: Path) -> list[tuple[int, Path]]:
    """(フレーム番号, train_log.jsonl)。4D ラン root なら frame_* を、単一シーンなら自身を frame 0 として返す。"""
    found = []
    for log in run_dir.glob("frame_*/train_log.jsonl"):
        try:
            found.append((int(log.parent.name.split("_")[1]), log))
        except (IndexError, ValueError):
            continue
    if found:
        return sorted(found)
    single = run_dir / "train_log.jsonl"
    return [(0, single)] if single.is_file() else []


def rows_from_train_log(path: Path, loss_key: str) -> list[tuple[int, float]]:
    """train_log.jsonl から (iteration, 値) を iteration 順に読む。"""
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


def mean_last_window(rows: list[tuple[int, float]], window: int = 100) -> float:
    """最終 iteration から遡って ``window`` iteration 分の平均。

    記録間隔が 100 iter だと最終記録点 1 つだけになるが、それで正しい。
    """
    if not rows:
        return float("nan")
    last_iteration = rows[-1][0]
    selected = [value for it, value in rows if it > last_iteration - window] or [rows[-1][1]]
    return sum(selected) / len(selected)


def convergence_iteration(iters: list[int], losses: list[float],
                          fraction: float = 0.10) -> int | None:
    """損失が初期値の ``fraction`` 以下へ最初に落ちた iteration。

    初期損失が違う run 同士では「その run の中でどれだけ下がったか」の相対量で
    あり、run 間の絶対比較には使えない点に注意 (HTML 側にも注記を出す)。
    """
    if not losses:
        return None
    threshold = losses[0] * fraction
    for iteration, loss in zip(iters, losses):
        if loss <= threshold:
            return iteration
    return None


def load_series(run_dir: Path, loss_key: str, frames: set[int] | None = None) -> list[dict]:
    """フレームごとの推移をフレーム番号順に読む。"""
    series = []
    for number, log in frame_logs(run_dir):
        if frames is not None and number not in frames:
            continue
        rows = rows_from_train_log(log, loss_key)
        if not rows:
            print(f"  [スキップ] {log} に {loss_key} がありません")
            continue
        iters = [it for it, _ in rows]
        values = [value for _, value in rows]
        series.append({
            "name": f"frame_{number:04d}",
            "frame": number,
            "rows": rows,
            "iters": iters,
            "losses": values,
            "final": mean_last_window(rows),
            "converged_iter": None if loss_key in HIGHER_IS_BETTER
                              else convergence_iteration(iters, values),
        })
    return series


def write_csv(series: list[dict], out_dir: Path, loss_key: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in series:
        with (out_dir / f"{item['name']}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["iter", loss_key])
            for iteration, value in item["rows"]:
                writer.writerow([iteration, f"{value:.8g}"])
            writer.writerow([SUMMARY_MARKER, f"{item['final']:.8g}"])


# ---------------------------------------------------------------------- 図
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


def build_html(runs: list[tuple[str, list[dict]]], title: str, loss_key: str) -> str:
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
        header += [f"最終 100 iter 平均 ({html.escape(label)})", f"収束 iter ({html.escape(label)})"]

    payload = json.dumps(
        {"curves": curve_datasets, "runs": run_legend,
         "barLabels": frame_names, "bars": bar_datasets, "key": loss_key,
         "log": loss_key not in HIGHER_IS_BETTER},
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
<div class="sub">{html.escape(loss_key)} vs iteration (各フレームの train_log.jsonl)。棒グラフと表の値は最終 100 iteration の平均。</div>

<div class="card">
  <h2 style="margin-top:0">グラフ 1: {html.escape(loss_key)} の推移</h2>
  <div class="wrap"><canvas id="curves"></canvas></div>
  <div class="btns"><button id="all">全表示</button><button id="none">全非表示</button></div>
  <div class="toggles" id="toggles"></div>
  <div class="note">{"Y 軸は対数スケール。" if loss_key not in HIGHER_IS_BETTER else ""}色はラン、チェックボックスでラン単位に表示を切り替える。</div>
</div>

<div class="card">
  <h2 style="margin-top:0">グラフ 2: フレームごとの最終値</h2>
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
              y:{{type:D.log ? 'logarithmic' : 'linear', title:{{display:true, text:D.key}}}} }} }}
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
    scales:{{ y:{{title:{{display:true, text:D.key + ' (最終100iterの平均)'}}}} }} }}
}});
</script></body></html>
"""


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
                        help="train_log.jsonl を持つラン。繰り返すと同じ図に重ねる")
    parser.add_argument("--loss_key", default="loss_total", choices=LOSS_KEYS)
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="描くフレーム番号 (frame_NNNN の NNNN。既定: 全フレーム)")
    parser.add_argument("--out_html", type=Path, default=None)
    parser.add_argument("--output_csv_dir", type=Path, default=None,
                        help="フレームごとの CSV の書き出し先 (--run が 1 本のときのみ)")
    parser.add_argument("--title", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out_html is None and args.output_csv_dir is None:
        parser.error("--out_html か --output_csv_dir の少なくとも一方を指定してください")
    if args.output_csv_dir is not None and len(args.run) != 1:
        parser.error("--output_csv_dir は --run が 1 本のときだけ指定できます")
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        parser.error(f"ラベルが重複しています: {labels}  (ラベル=パス で区別してください)")

    frames = set(args.frames) if args.frames is not None else None
    runs: list[tuple[str, list[dict]]] = []
    for label, run_dir in args.run:
        series = load_series(run_dir, args.loss_key, frames)
        if not series:
            print(f"[エラー] {label}: {run_dir} に対象フレームの train_log.jsonl がありません")
            return 1
        runs.append((label, series))
        print(f"  {label}: {len(series)} フレーム ({run_dir})")

    if args.output_csv_dir:
        write_csv(runs[0][1], args.output_csv_dir, args.loss_key)
        print(f"  CSV: {args.output_csv_dir} ({len(runs[0][1])} 本)")
    if args.out_html:
        title = args.title or f"{args.loss_key}: " + " / ".join(labels)
        args.out_html.parent.mkdir(parents=True, exist_ok=True)
        args.out_html.write_text(build_html(runs, title, args.loss_key), encoding="utf-8")
        print(f"  -> {args.out_html} ({args.out_html.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
