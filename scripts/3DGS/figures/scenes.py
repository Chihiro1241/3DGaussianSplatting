"""Scene list, paths and styling shared by the report tables and figures.

Every script in this directory imports from here, so that the table and all
figures list the same 21 scenes in the same order and draw each dataset in the
same colour.  Run the scripts from the repository root: the paths are relative.
"""
from pathlib import Path

# データセット別レイアウト: <ROOT_3DGS>/<dataset>/<scene>/  (1 シーン 1 ラン)
ROOT_3DGS = Path("output/3DGS")
REPORT_DIR = ROOT_3DGS / "benchmark_report" / "report"

# Report order of the datasets.
DATASETS = ["Mip-NeRF360", "Tanks&Temples", "Deep Blending", "Synthetic NeRF"]

# (run directory under ROOT_3DGS, dataset) in report order.  The scene name is
# the last component of the run directory.
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

# One colour per dataset; every scene of a dataset is drawn identically.
COLOURS = {
    "Mip-NeRF360": "#1f5fa8",
    "Tanks&Temples": "#c0392b",
    "Deep Blending": "#2e8b40",
    "Synthetic NeRF": "#7a7a7a",
}

# (panel title, datasets drawn in it): real-world scenes above, synthetic below.
PANELS = [
    (
        "実世界データセット（Mip-NeRF360 / Tanks&Temples / Deep Blending）",
        ["Mip-NeRF360", "Tanks&Temples", "Deep Blending"],
    ),
    ("Synthetic NeRF", ["Synthetic NeRF"]),
]

MILESTONES = [(7000, "7K"), (15000, "15K")]
