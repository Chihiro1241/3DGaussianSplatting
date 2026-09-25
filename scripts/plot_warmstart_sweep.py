"""Turn a warm-start iteration sweep into figures and a saturation analysis.

Reads the ``results.csv`` written by ``scripts/warmstart_iteration_sweep.py``
and produces, under ``<output>``:

``report.html``
    One self-contained page holding the three quantitative figures:

    1. frame number vs held-out PSNR, one line per condition -- this is where
       drift shows up, as a curve that keeps sloping down late in the sequence
       rather than settling;
    2. iteration budget vs sequence-mean PSNR, with the two baselines drawn as
       reference lines -- the saturation curve;
    3. cumulative training time vs sequence-mean PSNR -- the same quality read
       against what it costs.

    Each figure is followed by the numbers it plots, so the page is legible
    without hovering and without colour.

``qualitative_frame_NNNN.png``
    Ground truth beside each condition's render of one held-out camera.

``summary.csv`` and ``analysis.json``
    Per-condition means, and the two decision criteria: the smallest budget
    landing within ``--tolerance`` dB of the largest budget's mean, and a
    per-condition drift check over the tail of the sequence.

Usage::

    python scripts/plot_warmstart_sweep.py \\
        --sweep output/4DGS/neu3d/cook_spinach/warmstart_sweep_stage1 \\
        --data data/neu3d/cook_spinach/converted_4d
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPO_ROOT / "src", _REPO_ROOT / "extensions" / "4dgs"):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from gaussian_splatting.config import (  # noqa: E402
    load_config,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.data import load_dataset  # noqa: E402
from gaussian_splatting.io.checkpoint import (  # noqa: E402
    model_from_checkpoint_state,
    read_checkpoint,
)
from gaussian_splatting.renderer import GaussianRenderer  # noqa: E402

# Categorical slots from the reference palette, in its fixed order.  Hues are
# assigned to conditions by sorted budget and never cycled, so a condition
# keeps its colour across all three figures and across re-runs that add arms.
PALETTE_LIGHT = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
PALETTE_DARK = (
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)
MAX_SERIES = len(PALETTE_LIGHT)


@dataclass(frozen=True)
class Series:
    """One condition's rows, ordered by frame."""

    condition: str
    iterations: int
    kind: str
    frames: list[int]
    psnr: list[float]
    ssim: list[float]
    lpips: list[float]
    train_sec: list[float]
    colour_index: int

    @property
    def mean_psnr(self) -> float:
        return sum(self.psnr) / len(self.psnr)

    @property
    def mean_ssim(self) -> float:
        return sum(self.ssim) / len(self.ssim)

    @property
    def mean_lpips(self) -> float:
        return sum(self.lpips) / len(self.lpips)

    @property
    def total_train_sec(self) -> float:
        return sum(self.train_sec)

    @property
    def mean_train_sec(self) -> float:
        return self.total_train_sec / len(self.frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="sweep root written by scripts/warmstart_iteration_sweep.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="figure directory (default: <sweep>/figures)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset root; required for the qualitative comparison image",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "neu3d" / "warmstart_sweep.yaml",
        help="configuration used to render the qualitative comparison",
    )
    parser.add_argument(
        "--qualitative-frame",
        type=int,
        default=None,
        metavar="F",
        help="frame to render for the comparison image (default: the last)",
    )
    parser.add_argument(
        "--qualitative-camera",
        default=None,
        metavar="NAME",
        help="held-out camera to render (default: the first)",
    )
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help="produce only the quantitative figures",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        metavar="DB",
        help="how close to the largest budget counts as saturated (default: 0.1)",
    )
    parser.add_argument(
        "--drift-frames",
        type=int,
        default=10,
        metavar="N",
        help="how many trailing frames the drift check fits a slope over "
        "(default: 10)",
    )
    parser.add_argument(
        "--image-directory",
        default="images",
        help="COLMAP image subdirectory inside each frame directory",
    )
    parser.add_argument(
        "--render-backend",
        choices=("reference", "cuda"),
        default="cuda",
    )
    return parser


# --------------------------------------------------------------------------
# reading the sweep
# --------------------------------------------------------------------------


def condition_sort_key(condition: str, iterations: int) -> tuple[int, int]:
    """Order conditions as floor, increasing budget, then the scratch ceiling."""

    if condition == "iter0":
        return (0, 0)
    if condition.startswith("scratch"):
        return (2, iterations)
    return (1, iterations)


def condition_kind(condition: str) -> str:
    if condition == "iter0":
        return "iter0"
    if condition.startswith("scratch"):
        return "scratch"
    return "warmstart"


def read_series(results_csv: Path) -> list[Series]:
    """Group the result rows into one ordered series per condition."""

    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows in {results_csv}")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)

    names = sorted(
        grouped,
        key=lambda name: condition_sort_key(
            name, int(grouped[name][0]["iters"])
        ),
    )
    if len(names) > MAX_SERIES:
        raise ValueError(
            f"{len(names)} conditions exceed the {MAX_SERIES} categorical "
            "slots; plot a subset, or facet the figure"
        )
    series: list[Series] = []
    for index, name in enumerate(names):
        ordered = sorted(grouped[name], key=lambda row: int(row["frame"]))
        series.append(
            Series(
                condition=name,
                iterations=int(ordered[0]["iters"]),
                kind=condition_kind(name),
                frames=[int(r["frame"]) for r in ordered],
                psnr=[float(r["psnr"]) for r in ordered],
                ssim=[float(r["ssim"]) for r in ordered],
                lpips=[float(r["lpips"]) for r in ordered],
                train_sec=[float(r["train_sec"]) for r in ordered],
                colour_index=index,
            )
        )
    return series


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def least_squares_slope(x: list[float], y: list[float]) -> float:
    """Return dy/dx of the least-squares line, or 0 for a degenerate fit."""

    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    denominator = sum((xi - mean_x) ** 2 for xi in x)
    if denominator == 0.0:
        return 0.0
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / denominator


def analyse(
    series: list[Series], *, tolerance: float, drift_frames: int
) -> dict[str, object]:
    """Answer the two questions the sweep exists to answer.

    The saturation value is the largest *warm-start* budget's mean: the scratch
    baseline trains differently (its Gaussian count is free to grow), so it is
    a reference point, not the top of this curve.
    """

    warm = [s for s in series if s.kind == "warmstart"]
    saturated = max(warm, key=lambda s: s.iterations) if warm else None
    # The saturation criterion assumes quality rises with the budget.  It need
    # not: past some point a fixed Gaussian count simply overfits the training
    # views.  Report the best-scoring budget too, so a non-monotonic curve is
    # visible rather than hidden behind "within 0.1 dB of the largest".
    best = max(warm, key=lambda s: s.mean_psnr) if warm else None

    smallest_within: Series | None = None
    if saturated is not None:
        reference = saturated.mean_psnr
        for candidate in sorted(warm, key=lambda s: s.iterations):
            if candidate.mean_psnr >= reference - tolerance:
                smallest_within = candidate
                break

    smallest_within_best: Series | None = None
    if best is not None:
        for candidate in sorted(warm, key=lambda s: s.iterations):
            if candidate.mean_psnr >= best.mean_psnr - tolerance:
                smallest_within_best = candidate
                break

    drift = {}
    for s in series:
        tail = min(drift_frames, len(s.frames))
        frames = [float(f) for f in s.frames[-tail:]]
        values = s.psnr[-tail:]
        slope = least_squares_slope(frames, values)
        drift[s.condition] = {
            "tail_frames": tail,
            "slope_db_per_frame": slope,
            "total_db_over_tail": slope * (tail - 1) if tail > 1 else 0.0,
            "first_psnr": s.psnr[0],
            "last_psnr": s.psnr[-1],
            "drop_first_to_last": s.psnr[0] - s.psnr[-1],
        }

    return {
        "tolerance_db": tolerance,
        "saturation_condition": None if saturated is None else saturated.condition,
        "saturation_mean_psnr": None if saturated is None else saturated.mean_psnr,
        "smallest_within_tolerance": (
            None if smallest_within is None else smallest_within.condition
        ),
        "smallest_within_tolerance_iters": (
            None if smallest_within is None else smallest_within.iterations
        ),
        "best_condition": None if best is None else best.condition,
        "best_mean_psnr": None if best is None else best.mean_psnr,
        "monotonic_in_budget": (
            None if best is None else best.condition == saturated.condition
        ),
        "smallest_within_tolerance_of_best": (
            None if smallest_within_best is None else smallest_within_best.condition
        ),
        "smallest_within_tolerance_of_best_iters": (
            None if smallest_within_best is None else smallest_within_best.iterations
        ),
        "drift": drift,
        "conditions": {
            s.condition: {
                "iters": s.iterations,
                "kind": s.kind,
                "frames": len(s.frames),
                "mean_psnr": s.mean_psnr,
                "mean_ssim": s.mean_ssim,
                "mean_lpips": s.mean_lpips,
                "total_train_sec": s.total_train_sec,
                "mean_train_sec": s.mean_train_sec,
            }
            for s in series
        },
    }


def write_summary_csv(path: Path, series: list[Series]) -> None:
    columns = (
        "condition", "kind", "iters", "frames",
        "mean_psnr", "mean_ssim", "mean_lpips",
        "mean_train_sec", "total_train_sec",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for s in series:
            writer.writerow([
                s.condition, s.kind, s.iterations, len(s.frames),
                f"{s.mean_psnr:.4f}", f"{s.mean_ssim:.5f}", f"{s.mean_lpips:.5f}",
                f"{s.mean_train_sec:.2f}", f"{s.total_train_sec:.1f}",
            ])


# --------------------------------------------------------------------------
# the report page
# --------------------------------------------------------------------------


def _trace(
    x: list[float], y: list[float], name: str, index: int, *, mode: str = "lines+markers"
) -> dict[str, object]:
    return {
        "type": "scatter",
        "mode": mode,
        "name": name,
        "x": x,
        "y": y,
        "line": {"width": 2, "color": f"__SERIES_{index}__"},
        "marker": {"size": 8, "color": f"__SERIES_{index}__"},
        "hovertemplate": f"{name}<br>%{{x}} → %{{y:.3f}} dB<extra></extra>",
    }


def _table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f'<details class="table-view"><summary>{caption}</summary>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</details>"
    )


def build_report(
    series: list[Series], analysis: dict[str, object], *, title: str
) -> str:
    """Compose the three figures and their tables into one page."""

    warm = [s for s in series if s.kind == "warmstart"]
    baselines = [s for s in series if s.kind != "warmstart"]

    # Figure 1 -- per-frame quality, one line per condition.
    fig1 = [
        _trace([float(f) for f in s.frames], s.psnr, s.condition, s.colour_index)
        for s in series
    ]

    # Figure 2 -- the saturation curve.  A single series, so no legend box;
    # the baselines are reference lines rather than categorical hues.
    fig2 = [
        {
            "type": "scatter",
            "mode": "lines+markers+text",
            "name": "warm start",
            "x": [s.iterations for s in warm],
            "y": [s.mean_psnr for s in warm],
            "text": [f"{s.iterations}" for s in warm],
            "textposition": "top center",
            "textfont": {"color": "__TEXT_SECONDARY__", "size": 11},
            "line": {"width": 2, "color": "__SERIES_0__"},
            "marker": {"size": 9, "color": "__SERIES_0__"},
            "hovertemplate": "%{x} iterations → %{y:.3f} dB<extra></extra>",
        }
    ]

    # Figure 3 -- quality against what it costs.
    fig3 = [
        {
            "type": "scatter",
            "mode": "lines+markers+text",
            "name": "warm start",
            "x": [s.total_train_sec for s in warm],
            "y": [s.mean_psnr for s in warm],
            "text": [f"{s.iterations}" for s in warm],
            "textposition": "top center",
            "textfont": {"color": "__TEXT_SECONDARY__", "size": 11},
            "line": {"width": 2, "color": "__SERIES_0__"},
            "marker": {"size": 9, "color": "__SERIES_0__"},
            "hovertemplate": (
                "%{text} iterations<br>%{x:.0f} s → %{y:.3f} dB<extra></extra>"
            ),
        }
    ]
    for s in baselines:
        fig3.append({
            "type": "scatter",
            "mode": "markers+text",
            "name": s.condition,
            "x": [s.total_train_sec],
            "y": [s.mean_psnr],
            "text": [s.condition],
            "textposition": "bottom center",
            "textfont": {"color": "__TEXT_SECONDARY__", "size": 11},
            "marker": {
                "size": 11,
                "symbol": "diamond",
                "color": f"__SERIES_{s.colour_index}__",
            },
            "hovertemplate": (
                f"{s.condition}<br>%{{x:.0f}} s → %{{y:.3f}} dB<extra></extra>"
            ),
        })

    reference_lines = [
        {
            "type": "line",
            "xref": "paper", "x0": 0, "x1": 1,
            "yref": "y", "y0": s.mean_psnr, "y1": s.mean_psnr,
            "line": {"color": "__TEXT_MUTED__", "width": 1, "dash": "dot"},
        }
        for s in baselines
    ]
    reference_labels = [
        {
            "xref": "paper", "x": 1, "xanchor": "right",
            "yref": "y", "y": s.mean_psnr, "yanchor": "bottom",
            "text": f"{s.condition} ({s.mean_psnr:.2f} dB)",
            "showarrow": False,
            "font": {"color": "__TEXT_SECONDARY__", "size": 11},
        }
        for s in baselines
    ]

    per_frame_rows = [
        [s.condition, str(s.iterations)]
        + [f"{value:.3f}" for value in s.psnr]
        for s in series
    ]
    frame_headers = ["condition", "iters"] + [
        str(f) for f in max(series, key=lambda s: len(s.frames)).frames
    ]
    summary_rows = [
        [
            s.condition, s.kind, str(s.iterations), str(len(s.frames)),
            f"{s.mean_psnr:.3f}", f"{s.mean_ssim:.4f}", f"{s.mean_lpips:.4f}",
            f"{s.mean_train_sec:.1f}", f"{s.total_train_sec:.0f}",
        ]
        for s in series
    ]

    drift = analysis["drift"]
    drift_rows = [
        [
            condition,
            str(entry["tail_frames"]),
            f"{entry['slope_db_per_frame']:+.4f}",
            f"{entry['total_db_over_tail']:+.3f}",
            f"{entry['drop_first_to_last']:+.3f}",
        ]
        for condition, entry in drift.items()
    ]

    smallest = analysis["smallest_within_tolerance"]
    saturation = analysis["saturation_condition"]
    tolerance = analysis["tolerance_db"]
    if not smallest:
        headline = "no warm-start condition to compare"
    elif analysis["monotonic_in_budget"]:
        headline = (
            f"{smallest} is the smallest budget within {tolerance:.2f} dB of "
            f"{saturation}, the largest"
        )
    else:
        headline = (
            f"Quality is not monotonic in the budget: {analysis['best_condition']} "
            f"scores best ({analysis['best_mean_psnr']:.3f} dB), above the largest "
            f"budget {saturation} ({analysis['saturation_mean_psnr']:.3f} dB). "
            f"{analysis['smallest_within_tolerance_of_best']} is the smallest "
            f"budget within {tolerance:.2f} dB of the best."
        )

    figures = json.dumps(
        {"fig1": fig1, "fig2": fig2, "fig3": fig3,
         "shapes": reference_lines, "annotations": reference_labels}
    )
    palette = json.dumps({"light": PALETTE_LIGHT, "dark": PALETTE_DARK})
    # Frames are integers, so the ticks must be too; aim for about ten of them.
    all_frames = [f for s in series for f in s.frames]
    span = max(all_frames) - min(all_frames)
    frame_dtick = max(1, round(span / 10)) if span else 1

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{
    color-scheme: light;
    --surface-0: #f4f4f1;
    --surface-1: #fcfcfb;
    --border:    #d9d8d2;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #8a8983;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-0: #111110;
      --surface-1: #1a1a19;
      --border:    #35342f;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #8a8983;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-0: #111110;
    --surface-1: #1a1a19;
    --border:    #35342f;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #8a8983;
  }}
  body {{
    margin: 0; padding: 32px 24px 64px;
    background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  main {{ max-width: 1020px; margin: 0 auto; }}
  h1 {{ font-size: 21px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin: 40px 0 2px; }}
  p.lede {{ color: var(--text-secondary); margin: 0 0 8px; max-width: 70ch; }}
  .headline {{
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 16px; margin: 16px 0 8px;
    color: var(--text-primary);
  }}
  .chart {{
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px; margin-top: 8px;
  }}
  details.table-view {{ margin-top: 8px; }}
  details.table-view summary {{
    cursor: pointer; color: var(--text-secondary); font-size: 13px;
  }}
  table {{
    border-collapse: collapse; margin-top: 8px; font-size: 13px;
    display: block; overflow-x: auto; max-width: 100%;
  }}
  th, td {{
    border: 1px solid var(--border); padding: 4px 10px;
    text-align: right; white-space: nowrap;
  }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
    text-align: left;
  }}
  th {{ color: var(--text-secondary); font-weight: 600; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="lede">Held-out cameras only. Training time is the trainer's in-loop
  measurement, which excludes evaluation, checkpointing, and dataset loading.</p>
  <div class="headline"><strong>{headline}</strong></div>

  <h2>1. Quality along the sequence</h2>
  <p class="lede">Drift would appear as a line still sloping downward at the
  right-hand end rather than settling.</p>
  <div class="chart"><div id="fig1" style="height:420px"></div></div>
  {_table(frame_headers, per_frame_rows, "PSNR per frame (dB)")}

  <h2>2. Saturation</h2>
  <p class="lede">Sequence-mean PSNR against the per-frame iteration budget.
  Dotted lines mark the baselines.</p>
  <div class="chart"><div id="fig2" style="height:380px"></div></div>
  {_table(
      ["condition", "kind", "iters", "frames", "mean PSNR", "mean SSIM",
       "mean LPIPS", "s / frame", "total s"],
      summary_rows, "Per-condition means")}

  <h2>3. What the quality costs</h2>
  <p class="lede">The same means against total training time for the sequence.</p>
  <div class="chart"><div id="fig3" style="height:380px"></div></div>
  {_table(
      ["condition", "tail frames", "dB / frame", "dB over tail",
       "first − last"],
      drift_rows, "Drift over the tail of the sequence")}
</main>
<script>
const FIGURES = {figures};
const PALETTE = {palette};
const FRAME_DTICK = {frame_dtick};

function isDark() {{
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped) return stamped === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}}

function ink(name) {{
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
}}

function resolve(value, colours) {{
  if (typeof value === "string") {{
    const series = value.match(/^__SERIES_(\\d+)__$/);
    if (series) return colours[Number(series[1]) % colours.length];
    if (value === "__TEXT_SECONDARY__") return ink("--text-secondary");
    if (value === "__TEXT_MUTED__") return ink("--text-muted");
    return value;
  }}
  if (Array.isArray(value)) return value.map((v) => resolve(v, colours));
  if (value && typeof value === "object") {{
    const out = {{}};
    for (const [k, v] of Object.entries(value)) out[k] = resolve(v, colours);
    return out;
  }}
  return value;
}}

function layout(extra) {{
  const grid = ink("--border");
  // Object.assign is shallow, so a per-figure `xaxis` would otherwise replace
  // the base axis wholesale and take the grid and tick colours with it.
  const base = {{
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {{ color: ink("--text-secondary"), size: 12,
             family: "system-ui, -apple-system, sans-serif" }},
    margin: {{ l: 64, r: 24, t: 30, b: 52 }},
    // No axis lines: the recessive grid already carries the reading, and an
    // axis line lands mid-plot whenever a range spans zero.
    xaxis: {{ gridcolor: grid, zeroline: false, showline: false,
              title: {{ font: {{ color: ink("--text-secondary") }} }} }},
    yaxis: {{ gridcolor: grid, zeroline: false, showline: false,
              title: {{ font: {{ color: ink("--text-secondary") }} }} }},
    legend: {{ orientation: "h", y: -0.18,
               font: {{ color: ink("--text-secondary") }} }},
    hoverlabel: {{ font: {{ family: "system-ui, sans-serif" }} }},
  }};
  const overrides = extra || {{}};
  // Copy into a fresh object: assigning into `base` would overwrite its axes
  // before the merge below could read them.
  const merged = Object.assign({{}}, base, overrides);
  for (const axis of ["xaxis", "yaxis"]) {{
    merged[axis] = Object.assign({{}}, base[axis], overrides[axis]);
  }}
  return merged;
}}

function draw() {{
  const colours = isDark() ? PALETTE.dark : PALETTE.light;
  const config = {{ displayModeBar: false, responsive: true }};
  Plotly.react("fig1", resolve(FIGURES.fig1, colours), layout({{
    hovermode: "x unified",
    xaxis: {{ title: {{ text: "frame" }}, tickformat: "d", dtick: FRAME_DTICK }},
    yaxis: {{ title: {{ text: "PSNR (dB)" }} }},
  }}), config);
  Plotly.react("fig2", resolve(FIGURES.fig2, colours), layout({{
    hovermode: "closest", showlegend: false,
    shapes: resolve(FIGURES.shapes, colours),
    annotations: resolve(FIGURES.annotations, colours),
    xaxis: {{ title: {{ text: "iterations per frame" }} }},
    yaxis: {{ title: {{ text: "mean PSNR (dB)" }} }},
  }}), config);
  Plotly.react("fig3", resolve(FIGURES.fig3, colours), layout({{
    hovermode: "closest", showlegend: false,
    xaxis: {{ title: {{ text: "total training time for the sequence (s)" }} }},
    yaxis: {{ title: {{ text: "mean PSNR (dB)" }} }},
  }}), config);
  // Plotly caches the layout it was handed, so a theme change has to re-apply
  // the ink colours rather than only repainting the page around the plot.
  for (const id of ["fig1", "fig2", "fig3"]) {{
    Plotly.relayout(id, {{ "font.color": ink("--text-secondary") }});
  }}
}}

draw();
window.matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, {{
  attributes: true, attributeFilter: ["data-theme"],
}});
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# qualitative comparison
# --------------------------------------------------------------------------


def _position_lookup(metadata: dict[str, object]):
    """Map a dataset frame number to its position in the trained sequence.

    A sweep run with ``--frame-stride N`` hands the trainer an explicit frame
    list, and the trainer numbers its output directories over that list.  The
    sweep metadata records the list, which is what makes the two numbering
    schemes reconcilable after the fact.
    """

    sequence = metadata.get("sequence")
    if not sequence:
        return lambda frame: frame
    numbers = [int(Path(name).name.split("_")[-1]) for name in sequence]
    index = {number: position for position, number in enumerate(numbers, start=1)}

    def position_of(frame: int) -> int:
        try:
            return index[frame]
        except KeyError:
            raise ValueError(
                f"frame {frame} is not on this sweep's sequence"
            ) from None

    return position_of


_CAPTION_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
)


def _caption_font(image_font, size: int):
    """Return a scalable caption font, falling back to PIL's bitmap default."""

    for candidate in _CAPTION_FONT_CANDIDATES:
        try:
            return image_font.truetype(candidate, size)
        except OSError:
            continue
    return image_font.load_default()


def render_qualitative(
    series: list[Series],
    args: argparse.Namespace,
    sweep: Path,
    destination: Path,
) -> Path | None:
    """Write ground truth beside each condition's render of one camera."""

    from PIL import Image, ImageDraw, ImageFont

    config = load_config(args.config)
    device = resolve_device(config.runtime)
    dtype = resolve_dtype(config.runtime)
    renderer = GaussianRenderer(config.rendering, backend=args.render_backend)

    common = sorted(set.intersection(*(set(s.frames) for s in series)))
    if not common:
        print("no frame is common to every condition; skipping the comparison")
        return None
    frame = args.qualitative_frame or common[-1]
    if frame not in common:
        raise ValueError(
            f"frame {frame} is not present in every condition; common frames "
            f"are {common[0]}..{common[-1]}"
        )

    frame_directory = args.data / f"frame_{frame:04d}"
    dataset = load_dataset(
        frame_directory,
        config,
        splits=("test",),
        load_points=False,
        image_directory=args.image_directory,
    )
    cameras = {c.image_name: c for c in dataset.test}
    name = args.qualitative_camera or sorted(cameras)[0]
    if name not in cameras:
        raise ValueError(
            f"{name} is not a held-out camera; available: {sorted(cameras)}"
        )
    camera = cameras[name]

    def to_image(tensor: torch.Tensor) -> "Image.Image":
        array = (
            tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            .permute(1, 2, 0).cpu().numpy()
        )
        return Image.fromarray(array)

    panels: list[tuple[str, "Image.Image"]] = [
        ("ground truth", to_image(camera.image))
    ]
    from gaussian_splatting.training.trainer import camera_to

    runtime_camera = camera_to(camera, device=device, dtype=dtype, include_image=False)
    # A strided sweep files its checkpoints by position in the sequence, not by
    # the dataset frame number the CSV records; scratch always files by frame.
    for s in series:
        position = frame if s.kind == "scratch" else args.position_of(frame)
        checkpoint = (
            args.frame1_checkpoint
            if s.kind == "iter0"
            else sweep / s.condition / f"frame_{position:04d}" / "checkpoints"
            / f"iteration_{s.iterations:08d}.pt"
        )
        if not Path(checkpoint).is_file():
            print(f"  {s.condition}: no checkpoint at {checkpoint}; skipping")
            continue
        state = read_checkpoint(checkpoint, map_location="cpu")
        model = model_from_checkpoint_state(
            state, config, device=device, dtype=dtype
        )
        model.eval()
        with torch.no_grad():
            rendered = renderer(model, runtime_camera).image
        label = f"{s.condition} ({s.psnr[s.frames.index(frame)]:.2f} dB)"
        panels.append((label, to_image(rendered)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    width, height = panels[0][1].size
    columns = min(4, len(panels))
    rows = (len(panels) + columns - 1) // columns
    # Scale the caption to the panel: the bitmap default is unreadable beside a
    # 1300-pixel-wide render.
    point_size = max(12, width // 42)
    font = _caption_font(ImageFont, point_size)
    caption = int(point_size * 1.8)
    sheet = Image.new(
        "RGB", (columns * width, rows * (height + caption)), (250, 250, 248)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(panels):
        column, row = index % columns, index // columns
        x, y = column * width, row * (height + caption)
        draw.text(
            (x + point_size // 2, y + point_size // 3),
            label, fill=(20, 20, 20), font=font,
        )
        sheet.paste(image, (x, y + caption))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sweep = args.sweep
    output = args.output or sweep / "figures"
    output.mkdir(parents=True, exist_ok=True)

    series = read_series(sweep / "results.csv")
    analysis = analyse(
        series, tolerance=args.tolerance, drift_frames=args.drift_frames
    )

    write_summary_csv(output / "summary.csv", series)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.html").write_text(
        build_report(series, analysis, title=f"Warm-start iteration sweep — {sweep.name}"),
        encoding="utf-8",
    )

    print(f"{'condition':<18} {'iters':>6} {'frames':>7} {'PSNR':>8} "
          f"{'SSIM':>7} {'LPIPS':>7} {'s/frame':>9} {'total s':>9}")
    for s in series:
        print(f"{s.condition:<18} {s.iterations:>6} {len(s.frames):>7} "
              f"{s.mean_psnr:>8.3f} {s.mean_ssim:>7.4f} {s.mean_lpips:>7.4f} "
              f"{s.mean_train_sec:>9.1f} {s.total_train_sec:>9.0f}")
    print()
    if analysis["saturation_condition"]:
        print(f"largest budget:  {analysis['saturation_condition']} at "
              f"{analysis['saturation_mean_psnr']:.3f} dB")
        print(f"  smallest within {args.tolerance:.2f} dB of it: "
              f"{analysis['smallest_within_tolerance']}")
        print(f"best budget:     {analysis['best_condition']} at "
              f"{analysis['best_mean_psnr']:.3f} dB")
        print(f"  smallest within {args.tolerance:.2f} dB of it: "
              f"{analysis['smallest_within_tolerance_of_best']}")
        if not analysis["monotonic_in_budget"]:
            print("  NOTE: quality is not monotonic in the budget -- the "
                  "largest budget is not the best")
    print("\ndrift over the last frames (negative slope = still degrading):")
    for condition, entry in analysis["drift"].items():
        print(f"  {condition:<18} {entry['slope_db_per_frame']:+.4f} dB/frame "
              f"over {entry['tail_frames']} frames "
              f"({entry['total_db_over_tail']:+.3f} dB)")

    if not args.skip_qualitative:
        if args.data is None:
            print("\n--data not given; skipping the qualitative comparison")
        else:
            metadata_path = sweep / "sweep_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            args.frame1_checkpoint = Path(metadata["frame1_checkpoint"])
            args.position_of = _position_lookup(metadata)
            common = sorted(set.intersection(*(set(s.frames) for s in series)))
            frame = args.qualitative_frame or (common[-1] if common else None)
            written = render_qualitative(
                series, args, sweep,
                output / f"qualitative_frame_{frame:04d}.png",
            )
            if written is not None:
                print(f"\nqualitative comparison: {written}")

    print(f"\nreport: {output / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
