"""Train the 4D extension with the physically based priors of Dynamic 3D Gaussians.

This is ``scripts/4DGS/train_4d.py`` plus the regularization of Luiten et al.,
"Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
(3DV 2024).  Frames are still trained one by one and frame ``f >= 2`` still
starts from the Gaussians of frame ``f - 1``; in addition, every carried-over
frame adds the local-rigidity, rotation-similarity, long-term-isometry, and
colour-consistency priors to its image loss, may start from a
constant-velocity extrapolation, and may hold opacity and scale fixed.  See
``extensions/4dgs/dynamic_regularization.py`` for the terms.

The plain hand-off stays in ``scripts/4DGS/train_4d.py``, unchanged; this script
reuses its argument parser, dataset loading, and run bookkeeping so that the
two produce the same directory layout and differ only in the regularization.

The configuration must contain a ``dynamic_regularization`` section with
``enabled: true`` (for example ``configs/neu3d/dynamic_regularization.yaml``).
Because the neighbour graph fixes the Gaussian set, the usual workflow trains
frame 1 separately with density control and starts here from it::

    python scripts/4DGS/train_4d_regularized.py --data scene \\
        --config configs/neu3d/dynamic_regularization.yaml \\
        --output output/scene_4d_regularized \\
        --start-frame 2 --carry-over-checkpoint frame1/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "extensions" / "4dgs",
    _REPO_ROOT / "scripts" / "4DGS",
):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from gaussian_splatting.config import (  # noqa: E402
    Config,
    load_config,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.io import save_gaussians_ply  # noqa: E402
from gaussian_splatting.model import generate_initial_points  # noqa: E402
from gaussian_splatting.renderer import GaussianRenderer  # noqa: E402
from gaussian_splatting.training.trainer import Trainer  # noqa: E402
from trainer_4d import (  # noqa: E402
    FrameHandoff,
    build_frame_training_state,
    frame_output_directory,
    handoff_from_checkpoint,
)
from dynamic_regularization import (  # noqa: E402
    STATE_DIRECTORY_NAME,
    DynamicRegularizer,
    FrameMotionState,
    NeighborGraph,
    build_neighbor_graph,
    extrapolate_handoff,
    freeze_parameter_groups,
    load_motion_origin,
    load_neighbor_graph,
    save_motion_origin,
    save_neighbor_graph,
    validate_regularization_config,
)
import train_4d  # noqa: E402
from train_4d import (  # noqa: E402
    MANIFEST_NAME,
    _apply_warm_start_overrides,
    _build_snapshot_writer,
    _discard_checkpoints,
    _frame_config,
    _git_revision,
    _load_frame_datasets,
    _prepare_directory,
    _seed_everything,
    _write_manifest,
    discover_frame_directories,
)


def build_parser() -> argparse.ArgumentParser:
    """Return ``train_4d.py``'s parser extended by the regularization options."""

    parser = train_4d.build_parser()
    parser.description = __doc__
    regularization = parser.add_argument_group("dynamic regularization")
    regularization.add_argument(
        "--regularization-state",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory holding the neighbour graph and velocity origin of an "
        "earlier run, needed to restart a regularized sequence at "
        "--start-frame > 2 (default: <output>/" + STATE_DIRECTORY_NAME + ")",
    )
    return parser


def _write_run_metadata(
    path: Path, config: Config, args: argparse.Namespace, device: torch.device
) -> None:
    """Record what produced this run beside its results."""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_revision(),
        "argv": sys.argv,
        "entry_point": "scripts/4DGS/train_4d_regularized.py",
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


def _reference_graph(handoff: FrameHandoff, config: Config) -> NeighborGraph:
    """Build the neighbour graph from the frame a sequence is anchored to."""

    regularization = config.dynamic_regularization
    return build_neighbor_graph(
        handoff.parameters["means_world"],
        num_neighbors=regularization.num_neighbors,
        weight_lambda=regularization.neighbor_weight_lambda,
        reference_frame=handoff.source_frame,
    )


def _restore_regularization_state(
    args: argparse.Namespace,
    config: Config,
    handoff: FrameHandoff,
    output_state: Path,
) -> tuple[NeighborGraph, FrameMotionState | None]:
    """Recover the graph and velocity origin for a run that starts mid-sequence.

    At ``--start-frame 2`` the carried-over frame is the reference frame
    itself, so the graph can be built from it and there is no velocity yet.
    Later start frames need the graph of frame 1 and the state of frame
    ``start - 2``, which only an earlier run of the same sequence recorded.
    """

    source = args.regularization_state or output_state
    graph = load_neighbor_graph(source)
    found, origin = load_motion_origin(source, next_frame=args.start_frame)
    if graph is None:
        if args.start_frame != 2:
            raise FileNotFoundError(
                f"no neighbour graph in {source}: restarting a regularized "
                f"sequence at frame {args.start_frame} needs the graph of the "
                "reference frame; pass --regularization-state"
            )
        graph = _reference_graph(handoff, config)
    if (
        not found
        and args.start_frame > 2
        and config.dynamic_regularization.velocity_initialization
    ):
        raise FileNotFoundError(
            f"no velocity origin in {source} for frame {args.start_frame}; "
            "pass --regularization-state or disable velocity_initialization"
        )
    if graph.num_gaussians != handoff.num_gaussians:
        raise ValueError(
            f"neighbour graph has {graph.num_gaussians} Gaussians but "
            f"{args.carry_over_checkpoint} has {handoff.num_gaussians}"
        )
    # Keep this run restartable on its own, whatever it was restored from.
    save_neighbor_graph(output_state, graph)
    return graph, origin


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
    regularization = config.dynamic_regularization
    if not regularization.enabled:
        raise ValueError(
            f"{args.config} does not enable dynamic_regularization; use "
            "scripts/4DGS/train_4d.py for the plain frame-to-frame hand-off"
        )
    validate_regularization_config(config)

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

    regularization_state = args.output / STATE_DIRECTORY_NAME
    graph: NeighborGraph | None = None
    runtime_graph: NeighborGraph | None = None
    motion_origin: FrameMotionState | None = None
    if handoff is not None:
        graph, motion_origin = _restore_regularization_state(
            args, config, handoff, regularization_state
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

        regularized = carried_over
        previous_motion: FrameMotionState | None = None
        initial_handoff = handoff
        velocity_initialized = False
        if regularized:
            previous_motion = FrameMotionState.from_handoff(
                handoff, epsilon_q=config.model.epsilon_q
            )
            if regularization.velocity_initialization:
                initial_handoff = extrapolate_handoff(
                    handoff, motion_origin, epsilon_q=config.model.epsilon_q
                )
                velocity_initialized = motion_origin is not None

        state = build_frame_training_state(
            config=frame_config,
            train_cameras=train_cameras,
            device=device,
            dtype=dtype,
            handoff=initial_handoff,
            initial_points=initial_points,
            initial_colors=initial_colors,
        )
        del initial_points, initial_colors, initial_handoff

        regularizer: DynamicRegularizer | None = None
        frozen_groups: list[str] = []
        if regularized:
            if runtime_graph is None:
                runtime_graph = graph.to(device=device, dtype=dtype)
            regularizer = DynamicRegularizer(
                runtime_graph,
                previous_motion.to(device=device, dtype=dtype),
                regularization,
            )
            if regularization.freeze_opacity_and_scale:
                frozen_groups = freeze_parameter_groups(state.optimizer)

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
            regularizer=regularizer,
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
            f"{schedule}, Adam {state.adam_state}"
            + (
                f", regularized (k={runtime_graph.num_neighbors}, "
                f"velocity {'on' if velocity_initialized else 'off'}, "
                f"frozen {frozen_groups or 'none'})"
                if regularizer is not None
                else ""
            )
            + f" -> {frame_output}",
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
        if graph is None:
            # Frame 1 of this run is the reference frame.
            graph = _reference_graph(handoff, config)
            save_neighbor_graph(regularization_state, graph)
        # Frame f + 1 extrapolates from frame f - 1 to frame f.
        motion_origin = previous_motion
        save_motion_origin(
            regularization_state, motion_origin, next_frame=frame_number + 1
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
                "dynamic_regularization": regularizer is not None,
                "velocity_initialized": velocity_initialized,
                "frozen_parameter_groups": frozen_groups,
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
        del regularizer, previous_motion
        # torch.save keeps the storages it serialized in a reference cycle
        # (a function-local Pickler class and its persistent_id closure), so
        # the checkpoint written above pins this frame's parameters and Adam
        # state on the GPU until the cyclic collector happens to run.  Collect
        # now so every frame starts from the same residual memory.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
