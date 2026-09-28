"""Plot training loss and PSNR for all 21 benchmark scenes in one 2x2 figure.

Scene grouping, colours and smoothing are taken from make_loss_plot.py so that
every figure in the report reads as one family.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# データセット別レイアウト: <ROOT_3DGS>/<dataset>/<scene>/  (1 シーン 1 ラン)
ROOT_3DGS = Path("output/3DGS")
OUT = Path("output/3DGS/benchmark_report/report/loss_psnr_plot.pdf")

# One colour per dataset; every scene of a dataset is drawn identically.
COLOURS = {
    "Mip-NeRF360": "#1f5fa8",
    "Tanks&Temples": "#c0392b",
    "Deep Blending": "#2e8b40",
    "Synthetic NeRF": "#7a7a7a",
}

SCENES = [
    ("mipnerf360/bicycle", "Mip-NeRF360"),
    ("mipnerf360/bonsai", "Mip-NeRF360"),
    ("mipnerf360/counter", "Mip-NeRF360"),
    ("mipnerf360/flowers", "Mip-NeRF360"),
    ("mipnerf360/garden", "Mip-NeRF360"),
    ("mipnerf360/kitchen", "Mip-NeRF360"),
    ("mipnerf360/room", "Mip-NeRF360"),
    ("mipnerf360/stump", "Mip-NeRF360"),
    ("mipnerf360/treehill", "Mip-NeRF360"),
    ("tandt/train", "Tanks&Temples"),
    ("tandt/truck", "Tanks&Temples"),
    ("deepblending/drjohnson", "Deep Blending"),
    ("deepblending/playroom", "Deep Blending"),
    ("nerf_synthetic/chair", "Synthetic NeRF"),
    ("nerf_synthetic/drums", "Synthetic NeRF"),
    ("nerf_synthetic/ficus", "Synthetic NeRF"),
    ("nerf_synthetic/hotdog", "Synthetic NeRF"),
    ("nerf_synthetic/lego", "Synthetic NeRF"),
    ("nerf_synthetic/materials", "Synthetic NeRF"),
    ("nerf_synthetic/mic", "Synthetic NeRF"),
    ("nerf_synthetic/ship", "Synthetic NeRF"),
]

ROWS = [
    (
        "実世界データセット（Mip-NeRF360 / Tanks&Temples / Deep Blending）",
        ["Mip-NeRF360", "Tanks&Temples", "Deep Blending"],
    ),
    ("Synthetic NeRF", ["Synthetic NeRF"]),
]

# (log key, axis label, log scale?, legend corner)
COLUMNS = [
    ("loss_total", "損失値（対数スケール）", True, "upper right"),
    ("psnr", "PSNR (dB)", False, "lower right"),
]

MILESTONES = [(7000, "7K"), (15000, "15K")]
WINDOW = 10  # moving-average width, in logged intervals

# (row index, metric) -> fixed y-axis lower bound; anything absent is autoscaled.
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


def draw_panel(axes, title, datasets, key, ylabel, log_scale, legend_loc, show_legend,
               ylim_bottom=None):
    for directory, dataset in SCENES:
        if dataset not in datasets:
            continue
        columns = load_log(directory)
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
    axes.set_title(title, fontsize=10)
    axes.grid(True, which="major", linewidth=0.4, alpha=0.35)
    if log_scale:
        axes.grid(True, which="minor", linewidth=0.3, alpha=0.18)
    axes.set_axisbelow(True)

    if show_legend:
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

    figure, grid = plt.subplots(2, 2, figsize=(12, 8))
    for row, (group_title, datasets) in enumerate(ROWS):
        for column, (key, ylabel, log_scale, legend_loc) in enumerate(COLUMNS):
            metric = "損失" if key == "loss_total" else "PSNR"
            draw_panel(
                grid[row][column],
                f"{metric}｜{group_title}",
                datasets,
                key,
                ylabel,
                log_scale,
                legend_loc,
                show_legend=(row == 0),
                ylim_bottom=YLIM_BOTTOM.get((row, key)),
            )

    figure.suptitle(
        f"学習損失とPSNRの推移（{WINDOW}区間移動平均）", fontsize=12
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
