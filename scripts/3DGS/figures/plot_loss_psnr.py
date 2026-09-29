"""Plot training loss and PSNR for all 21 benchmark scenes as two figures.

Each figure has the same two-panel layout as plot_gaussian_count.py (real-world
datasets above, Synthetic NeRF below), so that loss_plot.pdf, psnr_plot.pdf and
gaussian_count_plot.pdf read as one family.
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from scenes import COLOURS, MILESTONES, PANELS, REPORT_DIR, ROOT_3DGS, SCENES

# (log key, output file, figure title, axis label, log scale?, legend corner)
FIGURES = [
    ("loss_total", "loss_plot.pdf", "学習損失 $\\mathcal{L}$ の推移",
     "損失値（対数スケール）", True, "upper right"),
    ("psnr", "psnr_plot.pdf", "PSNRの推移", "PSNR (dB)", False, "lower right"),
]

WINDOW = 10  # moving-average width, in logged intervals

# (panel index, metric) -> fixed y-axis lower bound; anything absent is autoscaled.
YLIM_BOTTOM = {(0, "psnr"): 15.0}


def load_log(directory: str) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = {
        "iteration": [], "loss_total": [], "loss_l1": [], "loss_dssim": [], "psnr": []
    }
    with (ROOT_3DGS / directory / "train_log.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            for key in columns:
                columns[key].append(record[key])
    return {key: np.asarray(values) for key, values in columns.items()}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def draw_panel(axes, logs, title, datasets, key, ylabel, log_scale, legend_loc,
               ylim_bottom=None):
    for directory, dataset in SCENES:
        if dataset not in datasets:
            continue
        columns = logs[directory]
        iterations, values = columns["iteration"], columns[key]

        smoothed = moving_average(values, WINDOW)
        centre = iterations[WINDOW - 1:] - (WINDOW - 1) * (iterations[1] - iterations[0]) / 2
        axes.plot(centre, smoothed, color=COLOURS[dataset], linewidth=1.2, alpha=0.9, zorder=2)

    if log_scale:
        axes.set_yscale("log")
    axes.set_xlim(0, 30000)
    if ylim_bottom is not None:
        axes.set_ylim(bottom=ylim_bottom)

    # Read the limits back only after any override, so labels sit inside the frame.
    bottom, top = axes.get_ylim()
    # Log axes need a multiplicative offset, linear axes an additive one.
    label_y = top * 0.92 if log_scale else top - 0.06 * (top - bottom)
    for iteration, label in MILESTONES:
        axes.axvline(iteration, color="gray", linestyle="--", linewidth=1.0, zorder=0)
        axes.text(
            iteration + 250, label_y, label,
            color="gray", fontsize=9, va="top", ha="left",
        )

    axes.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    axes.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}K")
    )
    axes.set_xlabel("反復数")
    axes.set_ylabel(ylabel)
    axes.set_title(title, fontsize=11)
    axes.grid(True, which="major", linewidth=0.4, alpha=0.35)
    if log_scale:
        axes.grid(True, which="minor", linewidth=0.3, alpha=0.18)
    axes.set_axisbelow(True)

    # One representative line per dataset, so the legend stays at dataset level.
    axes.legend(
        handles=[Line2D([], [], color=COLOURS[name], linewidth=1.6, label=name)
                 for name in datasets],
        loc=legend_loc,
        frameon=True,
        framealpha=0.9,
        fontsize=9,
    )


def main() -> None:
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    logs = {directory: load_log(directory) for directory, _ in SCENES}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename, figure_title, ylabel, log_scale, legend_loc in FIGURES:
        figure, panels = plt.subplots(2, 1, figsize=(8, 8))
        for index, (axes, (title, datasets)) in enumerate(zip(panels, PANELS)):
            draw_panel(
                axes,
                logs,
                title,
                datasets,
                key,
                ylabel,
                log_scale,
                legend_loc,
                ylim_bottom=YLIM_BOTTOM.get((index, key)),
            )

        figure.suptitle(f"{figure_title}（{WINDOW}区間移動平均）", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        out = REPORT_DIR / filename
        figure.savefig(out, dpi=300)
        plt.close(figure)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
