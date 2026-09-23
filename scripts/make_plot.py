"""Plot the Gaussian-count trajectory of all 21 benchmark scenes."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# データセット別レイアウト: <ROOT_3DGS>/<dataset>/runs/<scene>/
ROOT_3DGS = Path("output/3DGS")
OUT = Path("output/3DGS/benchmark_report/report/gaussian_count_plot.pdf")

# One colour per dataset; every scene of a dataset is drawn identically.
COLOURS = {
    "Mip-NeRF360": "#1f5fa8",
    "Tanks&Temples": "#c0392b",
    "Deep Blending": "#2e8b40",
    "Synthetic NeRF": "#7a7a7a",
}

SCENES = [
    ("mipnerf360/runs/bicycle", "Mip-NeRF360"),
    ("mipnerf360/runs/bonsai", "Mip-NeRF360"),
    ("mipnerf360/runs/counter", "Mip-NeRF360"),
    ("mipnerf360/runs/flowers", "Mip-NeRF360"),
    ("mipnerf360/runs/garden", "Mip-NeRF360"),
    ("mipnerf360/runs/kitchen", "Mip-NeRF360"),
    ("mipnerf360/runs/room", "Mip-NeRF360"),
    ("mipnerf360/runs/stump", "Mip-NeRF360"),
    ("mipnerf360/runs/treehill", "Mip-NeRF360"),
    ("tandt/runs/train", "Tanks&Temples"),
    ("tandt/runs/truck", "Tanks&Temples"),
    ("deepblending/runs/drjohnson", "Deep Blending"),
    ("deepblending/runs/playroom", "Deep Blending"),
    ("nerf_synthetic/runs/chair", "Synthetic NeRF"),
    ("nerf_synthetic/runs/drums", "Synthetic NeRF"),
    ("nerf_synthetic/runs/ficus", "Synthetic NeRF"),
    ("nerf_synthetic/runs/hotdog", "Synthetic NeRF"),
    ("nerf_synthetic/runs/lego", "Synthetic NeRF"),
    ("nerf_synthetic/runs/materials", "Synthetic NeRF"),
    ("nerf_synthetic/runs/mic", "Synthetic NeRF"),
    ("nerf_synthetic/runs/ship", "Synthetic NeRF"),
]

PANELS = [
    (
        "実世界データセット（Mip-NeRF360 / Tanks&Temples / Deep Blending）",
        ["Mip-NeRF360", "Tanks&Temples", "Deep Blending"],
    ),
    ("Synthetic NeRF", ["Synthetic NeRF"]),
]

MILESTONES = [(7000, "7K"), (15000, "15K")]


def load_curve(directory: str) -> tuple[list[int], list[float]]:
    iterations, counts = [], []
    with (ROOT_3DGS / directory / "train_log.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            iterations.append(record["iteration"])
            counts.append(record["gaussian_count"] / 1e4)  # 万個
    return iterations, counts


def draw_panel(axes, title: str, datasets: list[str]) -> None:
    peak = 0.0
    for directory, dataset in SCENES:
        if dataset not in datasets:
            continue
        iterations, counts = load_curve(directory)
        axes.plot(iterations, counts, color=COLOURS[dataset], linewidth=1.2, alpha=0.85)
        peak = max(peak, max(counts))

    # Headroom so the milestone labels never sit on top of a curve.
    top = peak * 1.12
    for iteration, label in MILESTONES:
        axes.axvline(iteration, color="gray", linestyle="--", linewidth=1.0, zorder=0)
        axes.text(
            iteration + 250, top * 0.985, label,
            color="gray", fontsize=9, va="top", ha="left",
        )

    axes.set_xlim(0, 30000)
    axes.set_ylim(0, top)
    axes.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    axes.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}K")
    )
    axes.set_xlabel("反復数")
    axes.set_ylabel("Gaussian数（万個）")
    axes.set_title(title, fontsize=11)
    axes.grid(True, linewidth=0.4, alpha=0.35)
    axes.set_axisbelow(True)

    # One representative line per dataset, so the legend stays at dataset level.
    axes.legend(
        handles=[Line2D([], [], color=COLOURS[name], linewidth=1.6, label=name)
                 for name in datasets],
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
    )


def main() -> None:
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure, (upper, lower) = plt.subplots(2, 1, figsize=(8, 8))
    for axes, (title, datasets) in zip((upper, lower), PANELS):
        draw_panel(axes, title, datasets)

    figure.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
