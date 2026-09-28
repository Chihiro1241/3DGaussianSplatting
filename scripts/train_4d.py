"""Train the 4D extension: per-frame 3DGS with frame-to-frame Gaussian hand-off.

The frame loop wraps the unmodified static trainer (section D of the
formulation document).  Every frame is a complete 3DGS run in its own output
subdirectory; frame ``f >= 2`` merely starts from the Gaussians optimized for
frame ``f - 1`` instead of from an SfM point cloud.

The dataset root is expected to contain one ordinary 3DGS scene directory per
time step, for example::

    scene/
      frame_0001/   # COLMAP sparse/0 + images, or transforms_*.json
      frame_0002/
      ...

Usage::

    python scripts/train_4d.py --data scene --config configs/default.yaml \
        --output output/scene_4d
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import asdict, replace
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPO_ROOT / "src", _REPO_ROOT / "extensions" / "4dgs"):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from gaussian_splatting.config import (  # noqa: E402
    Config,
    config_from_mapping,
    load_config,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.data import load_dataset  # noqa: E402
from gaussian_splatting.data.dataset import CameraSplits  # noqa: E402
from gaussian_splatting.io import save_gaussians_ply  # noqa: E402
from gaussian_splatting.model import generate_initial_points  # noqa: E402
from gaussian_splatting.renderer import GaussianRenderer  # noqa: E402
from gaussian_splatting.training.trainer import Trainer  # noqa: E402
from gaussian_splatting.training.snapshot import (  # noqa: E402
    SNAPSHOT_DIRECTORY_NAME,
    GaussianSnapshotWriter,
    parse_iteration_list,
)
from trainer_4d import (  # noqa: E402
    FrameHandoff,
    build_frame_training_state,
    frame_output_directory,
    handoff_from_checkpoint,
)


MANIFEST_NAME = "frames_4d.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="dataset root containing one scene directory per frame",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="complete YAML configuration"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="4D run root directory"
    )
    parser.add_argument(
        "--frame-pattern",
        default="frame_*",
        help="glob matching the per-frame subdirectories (default: frame_*)",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        default=None,
        metavar="NAME",
        help="explicit ordered frame subdirectory names, overriding --frame-pattern",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        metavar="F",
        help="one-based first frame to train (default: 1)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        metavar="F",
        help="one-based last frame to train (default: the final discovered frame)",
    )
    parser.add_argument(
        "--carry-over-checkpoint",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="frame F-1 checkpoint supplying the initial Gaussians when --start-frame > 1",
    )
    parser.add_argument(
        "--carry-over-sh-degree",
        type=int,
        default=None,
        metavar="L",
        help="active SH degree for --carry-over-checkpoint (default: derive from it)",
    )
    parser.add_argument(
        "--subsequent-frame-iterations",
        type=int,
        default=None,
        metavar="N",
        help="iteration budget for carried-over frames (default: training.iterations)",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="allow a pre-existing 4D output root for a fresh run",
    )
    parser.add_argument(
        "--image-directory",
        default="images",
        help="COLMAP image subdirectory within each frame directory",
    )
    parser.add_argument(
        "--disable-training-evaluation",
        action="store_true",
        help="defer all evaluation to the separate native-resolution evaluator",
    )
    parser.add_argument(
        "--render-backend",
        choices=("reference", "cuda"),
        default="reference",
        help="Gaussian rasterization backend (default: reference)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        metavar="N",
        help="record Gaussian centres, opacities, and count every N iterations "
        "into <output>/snapshots for eval/snapshot_viewer.py (0 disables)",
    )
    parser.add_argument(
        "--snapshot-iterations",
        default=None,
        metavar="LIST",
        help="comma-separated extra iterations to snapshot regardless of "
        "--snapshot-interval, e.g. 0,100,200,500",
    )
    parser.add_argument(
        "--snapshot-max-points",
        type=int,
        default=20000,
        metavar="K",
        help="thin each snapshot to at most K Gaussians; 0 stores every "
        "Gaussian (default: 20000)",
    )
    warm_start = parser.add_argument_group(
        "warm start",
        "Overrides for config.warm_start.  These apply to carried-over frames "
        "only, so frame 1 always trains like an ordinary static run.  Whatever "
        "is resolved here is written into <output>/config.yaml.",
    )
    warm_start.add_argument(
        "--position-lr-mode",
        choices=("exponential", "fixed"),
        default=None,
        help="'fixed' holds the position learning rate constant instead of "
        "decaying it over training.iterations, so that changing a frame's "
        "iteration budget does not also change its schedule",
    )
    warm_start.add_argument(
        "--position-lr-fixed",
        type=float,
        default=None,
        metavar="LR",
        help="constant position learning rate for --position-lr-mode fixed, "
        "scaled by the scene extent like training.position_lr_initial",
    )
    warm_start.add_argument(
        "--adam-state",
        choices=("reset", "carry"),
        default=None,
        help="'carry' inherits the previous frame's Adam moments; it requires "
        "features.adaptive_density_control=false so the Gaussian count is fixed",
    )
    storage = parser.add_argument_group("per-frame storage")
    storage.add_argument(
        "--frame-gaussian-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write each finished frame's Gaussians as "
        "DIR/frame_NNNN.ply (official property names).  A PLY holds only the "
        "parameters, so it is roughly a third of a checkpoint",
    )
    storage.add_argument(
        "--keep-frame-checkpoints",
        choices=("all", "last"),
        default="all",
        help="'last' deletes a frame's checkpoints once the next frame has "
        "started from them, holding the run to one checkpoint on disk while "
        "still leaving a resume point (default: all)",
    )
    return parser


def _build_snapshot_writer(
    args: argparse.Namespace, frame_output: Path, *, frame: int
) -> GaussianSnapshotWriter | None:
    """Return this frame's snapshot writer, or None when not requested.

    Each frame keeps its own ``snapshots`` directory, which is what gives the
    viewer its frame-by-iteration grid.
    """

    extra = parse_iteration_list(args.snapshot_iterations)
    if args.snapshot_interval <= 0 and not extra:
        return None
    if args.snapshot_interval < 0:
        raise ValueError("--snapshot-interval must not be negative")
    if args.snapshot_max_points < 0:
        raise ValueError("--snapshot-max-points must not be negative")
    return GaussianSnapshotWriter(
        frame_output / SNAPSHOT_DIRECTORY_NAME,
        interval=args.snapshot_interval,
        frame=frame,
        max_points=args.snapshot_max_points or None,
        extra_iterations=extra,
    )


def _seed_everything(seed: int) -> None:
    """Seed every generator used by initialization and view selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_directory(path: Path, config: Config, *, exist_ok: bool) -> None:
    """Create a run directory and persist its resolved configuration."""
    if path.exists():
        if not exist_ok:
            raise FileExistsError(
                f"output directory already exists and output.exist_policy=error: {path}"
            )
        if not path.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=exist_ok)
    config_path = path / "config.yaml"
    if config_path.exists():
        if load_config(config_path) != config:
            raise ValueError(
                f"existing resolved config does not match this run: {config_path}"
            )
        return
    config_path.write_text(
        yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def discover_frame_directories(
    root: Path,
    *,
    pattern: str,
    names: list[str] | None,
) -> list[Path]:
    """Return the ordered per-frame scene directories of a 4D dataset."""

    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if names is not None:
        directories = []
        for name in names:
            candidate = root / name
            if not candidate.is_dir():
                raise FileNotFoundError(f"frame directory does not exist: {candidate}")
            directories.append(candidate)
    else:
        directories = sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir() and fnmatch(child.name, pattern)
            ),
            key=lambda child: child.name,
        )
    if not directories:
        available = sorted(child.name for child in root.iterdir() if child.is_dir())
        raise FileNotFoundError(
            f"no frame directory in {root} matches {pattern!r}; "
            f"subdirectories are: {available}"
        )
    return directories


def _frame_config(config: Config, *, carried_over: bool, iterations: int | None) -> Config:
    """Return the configuration used for one frame, revalidated after any override."""
    if not carried_over or iterations is None or iterations == config.training.iterations:
        return config
    overridden = replace(config, training=replace(config.training, iterations=iterations))
    return config_from_mapping(asdict(overridden))


def _load_frame_datasets(
    frame_directory: Path,
    config: Config,
    *,
    image_directory: str,
    load_points: bool,
) -> tuple[CameraSplits, list, list, list]:
    """Load one frame's training, warm-up, and evaluation cameras.

    Mirrors the dataset handling of ``scripts/train.py`` for a single scene.
    """

    dataset = load_dataset(
        frame_directory,
        config,
        splits=("train", "val"),
        load_points=load_points,
        image_directory=image_directory,
    )
    train_cameras_quarter: list = []
    train_cameras_half: list = []
    if config.features.resolution_warmup:
        if config.data.resolution_scale != 1.0:
            raise ValueError(
                "resolution warm-up requires data.resolution_scale=1.0 so evaluation is native"
            )
        for scale, destination in ((0.25, "quarter"), (0.5, "half")):
            scaled = load_dataset(
                frame_directory,
                replace(config, data=replace(config.data, resolution_scale=scale)),
                splits=("train",),
                load_points=False,
                image_directory=image_directory,
            ).train
            if destination == "quarter":
                train_cameras_quarter = scaled
            else:
                train_cameras_half = scaled
    if dataset.format == "colmap":
        evaluation_cameras: list = []
    elif dataset.val:
        evaluation_cameras = dataset.val
    else:
        evaluation_cameras = load_dataset(
            frame_directory,
            config,
            splits=("test",),
            load_points=False,
            image_directory=image_directory,
        ).test
    return dataset, train_cameras_quarter, train_cameras_half, evaluation_cameras


def _apply_warm_start_overrides(
    config: Config, args: argparse.Namespace
) -> Config:
    """Fold the warm-start command-line overrides into the resolved config.

    Doing this before the run directory is written is what makes an experiment
    reproducible from ``<output>/config.yaml`` alone, with no need to also
    recover the command line.
    """

    overrides = {
        "position_lr_mode": args.position_lr_mode,
        "position_lr_fixed": args.position_lr_fixed,
        "adam_state": args.adam_state,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}
    if not supplied:
        return config
    updated = replace(config, warm_start=replace(config.warm_start, **supplied))
    # Re-validate through the loader so a command-line value is rejected on the
    # same terms as a value written in the file.
    return config_from_mapping(asdict(updated))


def _git_revision() -> dict[str, object]:
    """Return the working tree's commit and whether it has uncommitted edits."""

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # None means the question could not be answered, which is not the same
        # as a clean tree and should not be recorded as one.
        "dirty": None if status is None else bool(status),
    }


def _write_run_metadata(
    path: Path, config: Config, args: argparse.Namespace, device: torch.device
) -> None:
    """Record what produced this run beside its results."""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_revision(),
        "argv": sys.argv,
        "entry_point": "scripts/train_4d.py",
        "config_source": str(args.config),
        "resolved_config": asdict(config),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "python_version": sys.version,
    }
    destination = path / "run_metadata.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _discard_checkpoints(frame_output: Path) -> None:
    """Delete one finished frame's checkpoints under --keep-frame-checkpoints last."""

    checkpoints = frame_output / "checkpoints"
    if checkpoints.is_dir():
        shutil.rmtree(checkpoints)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish the frame-sequence manifest."""
    temporary = path.with_suffix(".json.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _apply_warm_start_overrides(load_config(args.config), args)
    if args.start_frame < 1:
        raise ValueError("--start-frame must be a positive integer")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("--end-frame must not precede --start-frame")
    if args.start_frame > 1 and args.carry_over_checkpoint is None:
        raise ValueError(
            "--start-frame > 1 requires --carry-over-checkpoint from the preceding frame"
        )
    if args.start_frame == 1 and args.carry_over_checkpoint is not None:
        raise ValueError(
            "frame 1 is initialized from the SfM point cloud; "
            "--carry-over-checkpoint applies only to --start-frame > 1"
        )
    if args.carry_over_sh_degree is not None and args.carry_over_checkpoint is None:
        raise ValueError("--carry-over-sh-degree requires --carry-over-checkpoint")
    if args.subsequent_frame_iterations is not None and args.subsequent_frame_iterations <= 0:
        raise ValueError("--subsequent-frame-iterations must be positive")
    if args.position_lr_fixed is not None and args.position_lr_mode == "exponential":
        raise ValueError(
            "--position-lr-fixed has no effect with --position-lr-mode exponential"
        )
    carry_adam = config.warm_start.adam_state == "carry"

    frame_directories = discover_frame_directories(
        args.data, pattern=args.frame_pattern, names=args.frames
    )
    if args.start_frame > len(frame_directories):
        raise ValueError(
            f"--start-frame {args.start_frame} exceeds the {len(frame_directories)} "
            "discovered frames"
        )
    last_frame = min(args.end_frame or len(frame_directories), len(frame_directories))

    device = resolve_device(config.runtime)
    dtype = resolve_dtype(config.runtime)
    _prepare_directory(
        args.output,
        config,
        exist_ok=args.allow_existing_output or args.start_frame > 1,
    )
    _write_run_metadata(args.output, config, args, device)
    renderer = GaussianRenderer(config.rendering, backend=args.render_backend)
    if args.frame_gaussian_dir is not None:
        args.frame_gaussian_dir.mkdir(parents=True, exist_ok=True)

    handoff: FrameHandoff | None = None
    if args.carry_over_checkpoint is not None:
        handoff = handoff_from_checkpoint(
            args.carry_over_checkpoint,
            config,
            source_frame=args.start_frame - 1,
            dtype=dtype,
            device="cpu",
            active_sh_degree=args.carry_over_sh_degree,
            carry_optimizer_state=carry_adam,
        )

    manifest_path = args.output / MANIFEST_NAME
    records: list[dict[str, object]] = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            record
            for record in previous.get("frames", [])
            if int(record["frame"]) < args.start_frame
        ]

    for frame_number in range(args.start_frame, last_frame + 1):
        frame_directory = frame_directories[frame_number - 1]
        carried_over = handoff is not None
        frame_config = _frame_config(
            config,
            carried_over=carried_over,
            iterations=args.subsequent_frame_iterations,
        )
        # Frame 1 reproduces the static run's seeding exactly; later frames get
        # their own reproducible stream so a restart matches an unbroken run.
        _seed_everything(config.runtime.seed + frame_number - 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        dataset, quarter, half, evaluation_cameras = _load_frame_datasets(
            frame_directory,
            frame_config,
            image_directory=args.image_directory,
            load_points=not carried_over,
        )
        if args.disable_training_evaluation:
            evaluation_cameras = []
        train_cameras = dataset.train

        initial_points = initial_colors = None
        initialization = "carried_over"
        if not carried_over:
            if dataset.initial_points is not None and dataset.initial_colors is not None:
                initial_points = dataset.initial_points.to(device=device, dtype=dtype)
                initial_colors = dataset.initial_colors.to(device=device, dtype=dtype)
                initialization = "sfm_points"
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(config.runtime.seed)
                initial_points, initial_colors = generate_initial_points(
                    dataset.scene_center, frame_config, generator
                )
                initialization = "random_points"

        state = build_frame_training_state(
            config=frame_config,
            train_cameras=train_cameras,
            device=device,
            dtype=dtype,
            handoff=handoff,
            initial_points=initial_points,
            initial_colors=initial_colors,
        )
        del initial_points, initial_colors

        # Density control replaces the model's parameter tensors in place, so
        # the initial counts must be read before training starts.
        initial_num_gaussians = state.num_gaussians
        initial_sh_degree = state.model.active_sh_degree
        frame_output = frame_output_directory(args.output, frame_number)
        _prepare_directory(frame_output, frame_config, exist_ok=False)
        trainer = Trainer(
            model=state.model,
            renderer=renderer,
            train_cameras=train_cameras,
            train_cameras_quarter=quarter or None,
            train_cameras_half=half or None,
            evaluation_cameras=evaluation_cameras,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            config=frame_config,
            output_directory=frame_output,
            start_iteration=0,
            camera_order=None,
            camera_cursor=0,
            best_mean_psnr=None,
            density_statistics=state.density_statistics,
            snapshot_writer=_build_snapshot_writer(
                args, frame_output, frame=frame_number
            ),
        )
        schedule = (
            f"position lr {state.position_learning_rate:.3e} (fixed)"
            if state.position_learning_rate is not None
            else "position lr exponential"
        )
        print(
            f"[frame {frame_number}/{last_frame}] {frame_directory.name}: "
            f"{initialization}, {initial_num_gaussians} Gaussians, "
            f"SH degree {initial_sh_degree}, "
            f"{frame_config.training.iterations} iterations, "
            f"{schedule}, Adam {state.adam_state} -> {frame_output}",
            flush=True,
        )
        try:
            trainer.train()
        except BaseException:
            trainer.write_training_telemetry(status="FAILED")
            records.append(
                {
                    "frame": frame_number,
                    "source": str(frame_directory),
                    "output": str(frame_output),
                    "status": "FAILED",
                    "initialization": initialization,
                }
            )
            _write_manifest(
                manifest_path,
                {"frames": records, "status": "FAILED", "frame_count": len(frame_directories)},
            )
            raise

        # The next frame inherits these parameters (equations 150-151).  Its
        # density statistics are always rebuilt from scratch; the Adam moments
        # travel with them only under warm_start.adam_state="carry".
        handoff = FrameHandoff.from_model(
            trainer.model,
            source_frame=frame_number,
            device="cpu",
            optimizer=state.optimizer if carry_adam else None,
        )
        gaussian_ply = None
        if args.frame_gaussian_dir is not None:
            gaussian_ply = args.frame_gaussian_dir / f"frame_{frame_number:04d}.ply"
            save_gaussians_ply(gaussian_ply, trainer.model)
        records.append(
            {
                "frame": frame_number,
                "source": str(frame_directory),
                "output": str(frame_output),
                "status": "COMPLETED",
                "initialization": initialization,
                "iterations": frame_config.training.iterations,
                "num_gaussians_start": initial_num_gaussians,
                "num_gaussians_end": handoff.num_gaussians,
                "active_sh_degree_start": initial_sh_degree,
                "active_sh_degree_end": handoff.active_sh_degree,
                "scene_extent": state.scene_extent,
                "adam_state": state.adam_state,
                "position_learning_rate": state.position_learning_rate,
                "training_loop_seconds": trainer.training_loop_seconds,
                "best_mean_psnr": trainer.best_mean_psnr,
                "gaussian_ply": None if gaussian_ply is None else str(gaussian_ply),
                "final_checkpoint": str(
                    frame_output
                    / "checkpoints"
                    / f"iteration_{frame_config.training.iterations:08d}.pt"
                ),
            }
        )
        _write_manifest(
            manifest_path,
            {
                "frames": records,
                "status": "COMPLETED" if frame_number == last_frame else "RUNNING",
                "frame_count": len(frame_directories),
            },
        )

        # Drop the previous frame's checkpoints only now: this frame has
        # finished and become the new resume point, so the run never goes
        # through a moment with nothing to restart from.
        if args.keep_frame_checkpoints == "last" and frame_number > args.start_frame:
            _discard_checkpoints(frame_output_directory(args.output, frame_number - 1))

        del trainer, state, dataset, train_cameras, quarter, half, evaluation_cameras
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
