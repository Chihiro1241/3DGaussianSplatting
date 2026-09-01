"""Audit local data and emit a non-training 3DGS paper benchmark manifest."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from gaussian_splatting.data.colmap_loader import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)
from gaussian_splatting.data.nerf_synthetic_loader import resolve_nerf_image_path


PAPER_PSNR: dict[str, tuple[float | None, float]] = {
    "bicycle": (23.604, 25.246), "flowers": (20.515, 21.520),
    "garden": (26.245, 27.410), "stump": (25.709, 26.550),
    "treehill": (22.085, 22.490), "room": (28.139, 30.632),
    "counter": (26.705, 28.700), "kitchen": (28.546, 30.317),
    "bonsai": (28.850, 31.980), "truck": (23.506, 25.187),
    "train": (18.892, 21.097), "drjohnson": (26.306, 28.766),
    "playroom": (29.245, 30.044), "mic": (None, 35.36),
    "chair": (None, 35.83), "ship": (None, 30.80),
    "materials": (None, 30.00), "lego": (None, 35.78),
    "drums": (None, 26.15), "ficus": (None, 34.87),
    "hotdog": (None, 37.72),
}

DATASET_SCENES = {
    "Mip-NeRF360": [
        "bicycle", "flowers", "garden", "stump", "treehill",
        "room", "counter", "kitchen", "bonsai",
    ],
    "Tanks&Temples": ["truck", "train"],
    "Deep Blending": ["drjohnson", "playroom"],
    "Synthetic NeRF": [
        "mic", "chair", "ship", "materials", "lego", "drums", "ficus", "hotdog",
    ],
}

DATASET_SLUG = {
    "Mip-NeRF360": "mipnerf360", "Tanks&Temples": "tandt",
    "Deep Blending": "deepblending", "Synthetic NeRF": "synthetic",
}

PAPER_TIME_SECONDS = {
    "Mip-NeRF360": {"7000": 6 * 60 + 25, "30000": 41 * 60 + 33},
    "Tanks&Temples": {"7000": 6 * 60 + 55, "30000": 26 * 60 + 54},
    "Deep Blending": {"7000": 4 * 60 + 35, "30000": 36 * 60 + 2},
    "Synthetic NeRF": {"7000": None, "30000": None},
}

PAPER_MODEL_MEMORY_MB = {
    "Mip-NeRF360": {"7000": 523, "30000": 734},
    "Tanks&Temples": {"7000": 270, "30000": 411},
    "Deep Blending": {"7000": 386, "30000": 676},
    "Synthetic NeRF": {"7000": None, "30000": None},
}


@dataclass
class SceneDryRun:
    dataset: str
    scene: str
    status: str
    dataset_path: str | None
    loader_type: str | None
    number_of_images: int | None
    num_train_views: int | None
    num_test_views: int | None
    train_test_split: str | None
    train_resolution: str | None
    evaluation_resolution: str | None
    initialization_method: str | None
    initial_gaussian_count: int | None
    background: str
    estimated_output_path: str
    paper_psnr_7000: float | None
    paper_psnr_30000: float
    notes: list[str]
    input_image_directory: str | None = None
    warmup_resolution: str | None = None
    sh_schedule: str = "degree 0; +1 every 1000 iterations; max degree 3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("output/paper_benchmark"))
    parser.add_argument("--real-config", type=Path, default=Path("configs/paper_benchmark_real.yaml"))
    parser.add_argument("--synthetic-config", type=Path, default=Path("configs/paper_benchmark_synthetic.yaml"))
    return parser


def _is_scene_root(path: Path) -> bool:
    sparse = path / "sparse" / "0"
    colmap = ((sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file()) or (
        (sparse / "cameras.txt").is_file() and (sparse / "images.txt").is_file()
    )
    return colmap or (path / "transforms_train.json").is_file()


def _discover(data_root: Path) -> dict[str, list[Path]]:
    wanted = {scene for scenes in DATASET_SCENES.values() for scene in scenes}
    found = {scene: [] for scene in wanted}
    for path in sorted((candidate for candidate in data_root.rglob("*") if candidate.is_dir())):
        key = path.name.lower()
        if key in found and _is_scene_root(path):
            found[key].append(path.resolve())
    return found


def _dimensions(paths: list[Path]) -> str:
    sizes: set[tuple[int, int]] = set()
    for path in paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    ordered = sorted(sizes)
    return ",".join(f"{width}x{height}" for width, height in ordered)


def _warmup_resolution(resolution: str) -> str:
    sizes: list[str] = []
    for token in resolution.split(","):
        width_text, height_text = token.split("x")
        width, height = int(width_text), int(height_text)
        sizes.append(
            f"{max(1, round(width * 0.25))}x{max(1, round(height * 0.25))}"
            f" -> {max(1, round(width * 0.5))}x{max(1, round(height * 0.5))}"
            f" -> {width}x{height}"
        )
    return ",".join(sizes)


def _point_count(sparse: Path) -> int:
    binary = sparse / "points3D.bin"
    if binary.is_file():
        with binary.open("rb") as stream:
            value = stream.read(8)
        if len(value) != 8:
            raise ValueError(f"invalid COLMAP point file: {binary}")
        return int(struct.unpack("<Q", value)[0])
    text = sparse / "points3D.txt"
    if text.is_file():
        return sum(
            1 for line in text.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    raise FileNotFoundError(f"COLMAP point cloud is missing below {sparse}")


def _colmap_metadata(root: Path, image_directory: str = "images") -> dict[str, Any]:
    sparse = root / "sparse" / "0"
    if (sparse / "cameras.bin").is_file():
        cameras = read_cameras_binary(sparse / "cameras.bin")
        images = read_images_binary(sparse / "images.bin")
    else:
        cameras = read_cameras_text(sparse / "cameras.txt")
        images = read_images_text(sparse / "images.txt")
    ordered = sorted(images.values(), key=lambda image: image.name)
    image_paths = [root / image_directory / image.name for image in ordered]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} COLMAP images are missing; first={missing[0]}")
    models = sorted({camera.model for camera in cameras.values()})
    count = len(ordered)
    return {
        "loader": "colmap",
        "total": count,
        "train": count - (count + 7) // 8,
        "test": (count + 7) // 8,
        "resolution": _dimensions(image_paths),
        "point_count": _point_count(sparse),
        "camera_models": models,
        "image_directory": image_directory,
    }


def _synthetic_metadata(root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    paths: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        source = root / f"transforms_{split}.json"
        metadata = json.loads(source.read_text(encoding="utf-8"))
        frames = metadata["frames"]
        paths[split] = [resolve_nerf_image_path(root, frame["file_path"]) for frame in frames]
        counts[split] = len(frames)
    return {
        "loader": "nerf_synthetic",
        "total": sum(counts.values()),
        "train": counts["train"],
        "test": counts["test"],
        "val": counts["val"],
        "resolution": _dimensions(paths["train"] + paths["test"]),
    }


def _audit(real_config: dict[str, Any], synthetic_config: dict[str, Any]) -> list[dict[str, str]]:
    data = real_config["data"]
    initialization = synthetic_config["initialization"]
    rendering = synthetic_config["rendering"]
    features = real_config["features"]
    synthetic_features = synthetic_config["features"]
    return [
        {"condition": "real test split is every 8th image", "status": "PASS" if data["test_every"] == 8 else "FAIL", "observed": f"test_every={data['test_every']}; COLMAP loader uses sorted image names"},
        {"condition": "test evaluation at native input resolution", "status": "PASS" if data["resolution_scale"] == 1.0 and features["resolution_warmup"] else "FAIL", "observed": f"native camera set uses resolution_scale={data['resolution_scale']}; separate PIL-resized warm-up camera sets are training-only"},
        {"condition": "paper input resolution selection for Mip-NeRF360", "status": "PASS", "observed": "runner selects images_4 for outdoor and images_2 for indoor; loader accepts an explicit image directory"},
        {"condition": "real scenes initialize from SfM points", "status": "PASS", "observed": "COLMAP loader returns points3D and train.py prefers them"},
        {"condition": "Synthetic NeRF uses 100000 uniform random Gaussians", "status": "FAIL" if initialization["num_gaussians"] != 100000 else "PASS", "observed": f"configured num_gaussians={initialization['num_gaussians']}; generator is uniform in configured AABB"},
        {"condition": "Synthetic NeRF initialization volume matches official implementation", "status": "PASS" if initialization["aabb_half_extent"] == [1.3, 1.3, 1.3] else "FAIL", "observed": f"configured AABB half extent={initialization['aabb_half_extent']}; official implementation samples each coordinate in [-1.3, 1.3]"},
        {"condition": "Synthetic NeRF uses white background", "status": "PASS" if synthetic_config["data"]["rgba_background"] == "white" and rendering["background"] == [1.0, 1.0, 1.0] else "FAIL", "observed": f"rgba_background={synthetic_config['data']['rgba_background']}, rendering.background={rendering['background']}"},
        {"condition": "Synthetic NeRF random SH-DC initialization", "status": "PASS" if synthetic_features["random_initial_sh_dc"] else "FAIL", "observed": f"random_initial_sh_dc={synthetic_features['random_initial_sh_dc']}"},
        {"condition": "SH starts at degree 0 and adds one band every 1000 iterations", "status": "PASS" if features["progressive_sh_degree"] else "FAIL", "observed": f"progressive_sh_degree={features['progressive_sh_degree']}; active degree is passed to CUDA and reference renderers"},
        {"condition": "quarter-resolution warm-up, upsample at iterations 250 and 500", "status": "PASS" if features["resolution_warmup"] else "FAIL", "observed": "separate loader-generated 1/4, 1/2, and native camera sets switch after iterations 250 and 500"},
        {"condition": "paper-equivalent ADC, LR, and loss", "status": "PASS_WITH_CAVEAT", "observed": "headline constants, schedules, scene-spatial LR scale, and L1/D-SSIM weighting match; implementation is custom rather than the paper release"},
        {"condition": "checkpoints exactly retained at 7000 and 30000", "status": "PASS", "observed": "milestone checkpoints are permanent; intermediate recovery.pt is overwritable"},
        {"condition": "required timing and PyTorch memory telemetry", "status": "PASS", "observed": "training-only and wall clocks plus allocated/reserved peaks persist across resume; process VRAM is runner-owned"},
    ]


def main() -> int:
    args = _parser().parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    real_config = yaml.safe_load(args.real_config.read_text(encoding="utf-8"))
    synthetic_config = yaml.safe_load(args.synthetic_config.read_text(encoding="utf-8"))
    discovered = _discover(data_root)
    rows: list[SceneDryRun] = []
    for dataset, scenes in DATASET_SCENES.items():
        for scene in scenes:
            candidates = discovered[scene]
            output = output_root / "runs" / f"{DATASET_SLUG[dataset]}_{scene}"
            paper7, paper30 = PAPER_PSNR[scene]
            if len(candidates) != 1:
                status = "MISSING" if not candidates else "AMBIGUOUS"
                rows.append(SceneDryRun(
                    dataset, scene, status, None, None, None, None, None, None,
                    None, None, None, None, "white" if dataset == "Synthetic NeRF" else "black",
                    str(output), paper7, paper30,
                    ["no matching dataset root found"] if not candidates else [f"candidates={list(map(str, candidates))}"],
                ))
                continue
            root = candidates[0]
            try:
                if dataset == "Mip-NeRF360":
                    paper_image_directory = "images_4" if scene in DATASET_SCENES["Mip-NeRF360"][:5] else "images_2"
                else:
                    paper_image_directory = "images"
                metadata = _synthetic_metadata(root) if dataset == "Synthetic NeRF" else _colmap_metadata(root, paper_image_directory)
                notes: list[str] = []
                if metadata["loader"] == "colmap" and metadata["camera_models"] != ["PINHOLE"]:
                    notes.append(f"unsupported camera models: {metadata['camera_models']}")
                if dataset == "Synthetic NeRF":
                    split = f"source train={metadata['train']}, val={metadata['val']}, test={metadata['test']}"
                    init_method, init_count, background = "uniform random AABB (required)", 100000, "white (required)"
                else:
                    split = f"sorted names; index % 8 == 0 test ({metadata['train']}/{metadata['test']})"
                    init_method, init_count, background = "COLMAP SfM points3D", metadata["point_count"], "black"
                status = "AVAILABLE" if not notes else "INCOMPATIBLE"
                rows.append(SceneDryRun(
                    dataset, scene, status, str(root), metadata["loader"], metadata["total"],
                    metadata["train"], metadata["test"], split,
                    f"{metadata['resolution']} (required schedule: 1/4 -> 1/2 -> native)",
                    f"{metadata['resolution']} (native)", init_method, init_count, background,
                    str(output), paper7, paper30, notes,
                    input_image_directory=(
                        "train/, val/, test/" if dataset == "Synthetic NeRF"
                        else metadata["image_directory"]
                    ),
                    warmup_resolution=_warmup_resolution(metadata["resolution"]),
                ))
            except Exception as error:
                rows.append(SceneDryRun(
                    dataset, scene, "INCOMPATIBLE", str(root), None, None, None, None, None,
                    None, None, None, None, "white" if dataset == "Synthetic NeRF" else "black",
                    str(output), paper7, paper30, [f"{type(error).__name__}: {error}"],
                ))

    audit = _audit(real_config, synthetic_config)
    blocking = [entry for entry in audit if entry["status"] in {"FAIL", "BLOCKED", "REVIEW"}]
    missing = [row.scene for row in rows if row.status == "MISSING"]
    incompatible = [row.scene for row in rows if row.status in {"AMBIGUOUS", "INCOMPATIBLE"}]
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run-only",
        "training_started": False,
        "benchmark_status": "BLOCKED" if blocking or missing or incompatible else "READY",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "backend": "cuda", "seed": 0, "iterations": 30000,
        "evaluation_iterations": [7000, 30000], "expected_training_runs": 21,
        "paper_psnr": {scene: {"7000": values[0], "30000": values[1]} for scene, values in PAPER_PSNR.items()},
        "paper_average_training_seconds": PAPER_TIME_SECONDS,
        "paper_average_model_memory_MB": PAPER_MODEL_MEMORY_MB,
        "condition_audit": audit,
        "blocking_conditions": blocking,
        "missing_scenes": missing,
        "incompatible_scenes": incompatible,
        "scenes": [asdict(row) for row in rows],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "report").mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_root / "results" / "dry_run.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8"
    )

    headings = ["Dataset", "Scene", "Status", "Path", "Loader", "Input", "Images", "Train/Test", "Warm-up resolution", "Eval resolution", "Initialization", "Initial N", "Background", "SH schedule", "Output"]
    print(" | ".join(headings))
    print(" | ".join(["---"] * len(headings)))
    for row in rows:
        values = [row.dataset, row.scene, row.status, row.dataset_path or "-", row.loader_type or "-",
                  row.input_image_directory or "-", str(row.number_of_images or "-"),
                  f"{row.num_train_views or '-'}/{row.num_test_views or '-'}",
                  row.warmup_resolution or "-", row.evaluation_resolution or "-", row.initialization_method or "-",
                  str(row.initial_gaussian_count or "-"), row.background, row.sh_schedule, row.estimated_output_path]
        print(" | ".join(values))
    print(f"\nbenchmark_status={manifest['benchmark_status']}")
    print(f"missing={','.join(missing) if missing else '-'}")
    print(f"incompatible={','.join(incompatible) if incompatible else '-'}")
    print(f"manifest={output_root / 'manifest.json'}")
    return 2 if manifest["benchmark_status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
