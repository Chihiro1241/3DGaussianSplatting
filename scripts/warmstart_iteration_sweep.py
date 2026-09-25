"""Sweep the per-frame iteration budget of warm-started 4D training.

A dynamic scene is optimized frame by frame, every frame after the first
starting from the previous frame's Gaussians.  The question this driver answers
is how many iterations one such frame deserves: too few and the reconstruction
lags the motion, too many and the sequence costs more than it is worth.

Every condition shares one frame-1 checkpoint, so the only thing that differs
between them is what happens from frame 2 onward.  Three kinds of condition are
supported:

``warmstart``
    ``scripts/train_4d.py`` with ``configs/neu3d/warmstart_sweep.yaml``:
    densification, pruning, opacity reset, progressive SH, and the resolution
    warm-up are all off, the Gaussian count is therefore fixed, and the
    position learning rate is a constant rather than a decay over the budget.
    The budget itself is the swept variable.
``iter0``
    No training at all -- the frame-1 model is evaluated against every later
    frame.  This is the floor: any budget that fails to beat it is buying
    nothing.
``scratch``
    Each frame trained independently with ordinary adaptive density control,
    as if it were a static scene.  This is the ceiling, and the expensive one,
    so it may be evaluated on a subset of frames.

Metrics always come from held-out cameras.  ``load_dataset`` splits a COLMAP
scene by image name and keeps every eighth camera out of training, and this
driver evaluates exactly that split -- the training views are never scored.

Results accumulate in ``<output>/results.csv`` with one row per
``(condition, frame)``.  Both training and evaluation skip work that the CSV
and the per-frame metric JSON files already record, so an interrupted sweep
resumes by being re-run with the same arguments.

Usage::

    python scripts/warmstart_iteration_sweep.py \\
        --data data/neu3d/cook_spinach/converted_4d \\
        --frame1-checkpoint output/4DGS/neu3d/cook_spinach/\\
cook_spinach_baseline_30k/frame_0001/checkpoints/iteration_00030000.pt \\
        --output output/4DGS/neu3d/cook_spinach/warmstart_sweep_stage1 \\
        --iters 0 100 250 500 1000 2000 5000 \\
        --end-frame 4 --render-backend cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPO_ROOT / "src", _REPO_ROOT / "extensions" / "4dgs"):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from gaussian_splatting.config import (  # noqa: E402
    Config,
    load_config,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.data import load_dataset  # noqa: E402
from gaussian_splatting.data.camera import Camera  # noqa: E402
from gaussian_splatting.evaluation.metrics import LPIPSMetric  # noqa: E402
from gaussian_splatting.evaluation.runner import evaluate_camera_set  # noqa: E402
from gaussian_splatting.io.checkpoint import (  # noqa: E402
    model_from_checkpoint_state,
    read_checkpoint,
)
from gaussian_splatting.renderer import GaussianRenderer  # noqa: E402

RESULTS_NAME = "results.csv"
RESULTS_COLUMNS = (
    "condition",
    "iters",
    "frame",
    "psnr",
    "ssim",
    "lpips",
    "train_sec",
    "n_gaussians",
    "loss",
)
DEFAULT_WARM_START_CONFIG = "configs/neu3d/warmstart_sweep.yaml"
DEFAULT_SCRATCH_CONFIG = "configs/neu3d/baseline_7k.yaml"


@dataclass(frozen=True)
class Condition:
    """One arm of the sweep."""

    name: str
    kind: str  # "warmstart" | "iter0" | "scratch"
    iterations: int
    #: Real dataset frame numbers this arm is scored on.  ``scratch`` may thin
    #: the sequence out because a full independent run per frame is the
    #: costliest arm by far.
    frames: tuple[int, ...]
    #: Where each of those frames lands inside the warm-start run, aligned with
    #: ``frames``.  With ``--frame-stride 1`` the two are identical.  With a
    #: larger stride ``scripts/train_4d.py`` is handed an explicit frame list
    #: and numbers its output directories 1..N over *that* list, so real frame
    #: 7 may live in ``frame_0004``.  The CSV always records the real number,
    #: which keeps every plot on a true time axis.
    positions: tuple[int, ...]

    @property
    def trains(self) -> bool:
        return self.kind != "iter0"

    def position_of(self, frame: int) -> int:
        return self.positions[self.frames.index(frame)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="dataset root holding one scene directory per frame",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="sweep root directory"
    )
    parser.add_argument(
        "--frame1-checkpoint",
        type=Path,
        required=True,
        metavar="CHECKPOINT",
        help="the frame-1 model every condition starts from; train it once "
        "with configs/neu3d/base.yaml (30,000 iterations, seed 0)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / DEFAULT_WARM_START_CONFIG,
        help=f"warm-start configuration (default: {DEFAULT_WARM_START_CONFIG})",
    )
    parser.add_argument(
        "--scratch-config",
        type=Path,
        default=_REPO_ROOT / DEFAULT_SCRATCH_CONFIG,
        help=f"per-frame independent-training configuration for the scratch "
        f"baseline (default: {DEFAULT_SCRATCH_CONFIG})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="warm-start iteration budgets to compare; 0 means the frame-1 "
        "model is evaluated untouched",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        metavar="F",
        help="first frame trained and scored; it must lie on the stride "
        "sequence, which always begins at dataset frame 1 because that is "
        "what --frame1-checkpoint supplies (default: 1 + --frame-stride)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        metavar="F",
        help="last frame trained and scored (default: the final frame)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        metavar="N",
        help="advance the sequence N dataset frames at a time, which scales "
        "the motion each warm-started frame has to absorb; the robustness "
        "check of the whole sweep (default: 1)",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="also run the independent-per-frame baseline",
    )
    parser.add_argument(
        "--scratch-every",
        type=int,
        default=1,
        metavar="N",
        help="score the scratch baseline every N frames instead of all of "
        "them; independent training is the costliest arm (default: 1)",
    )
    parser.add_argument(
        "--adam-state",
        choices=("reset", "carry"),
        default=None,
        help="override warm_start.adam_state for every warm-start condition",
    )
    parser.add_argument(
        "--position-lr-mode",
        choices=("exponential", "fixed"),
        default=None,
        help="override warm_start.position_lr_mode",
    )
    parser.add_argument(
        "--position-lr-fixed",
        type=float,
        default=None,
        metavar="LR",
        help="override warm_start.position_lr_fixed",
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
        help="rasterization backend for training and evaluation (default: cuda)",
    )
    parser.add_argument(
        "--keep-frame-checkpoints",
        choices=("all", "last"),
        default="all",
        help="passed through to scripts/train_4d.py (default: all)",
    )
    parser.add_argument(
        "--frame-gaussians",
        action="store_true",
        help="also write each frame's Gaussians as a PLY under "
        "<output>/<condition>/gaussians",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the commands without running anything",
    )
    return parser


# --------------------------------------------------------------------------
# results table
# --------------------------------------------------------------------------


def read_results(path: Path) -> list[dict[str, str]]:
    """Return the rows already recorded, or an empty list."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def append_result(path: Path, row: dict[str, object]) -> None:
    """Append one ``(condition, frame)`` row, writing the header if needed."""

    is_new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULTS_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def recorded_pairs(rows: list[dict[str, str]]) -> set[tuple[str, int]]:
    return {(row["condition"], int(row["frame"])) for row in rows}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def run(command: list[str], log_path: Path, *, dry_run: bool) -> None:
    """Run a child process, teeing its output to a log."""

    printable = " ".join(command)
    if dry_run:
        print(f"  [dry-run] {printable}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"failed (exit {code}): {printable}")


def train_warmstart(
    condition: Condition, args: argparse.Namespace, condition_dir: Path
) -> None:
    """Drive one warm-start arm through scripts/train_4d.py.

    The first frame of the sequence is never retrained: the run starts one
    position later and takes its initial Gaussians from the shared checkpoint,
    which is what makes the arms comparable.
    """

    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "train_4d.py"),
        "--data", str(args.data),
        "--config", str(args.config),
        "--output", str(condition_dir),
        "--start-frame", str(condition.positions[0]),
        "--end-frame", str(condition.positions[-1]),
        "--carry-over-checkpoint", str(args.frame1_checkpoint),
        "--subsequent-frame-iterations", str(condition.iterations),
        "--image-directory", args.image_directory,
        "--render-backend", args.render_backend,
        "--keep-frame-checkpoints", args.keep_frame_checkpoints,
        "--disable-training-evaluation",
        "--allow-existing-output",
    ]
    for flag, value in (
        ("--adam-state", args.adam_state),
        ("--position-lr-mode", args.position_lr_mode),
        ("--position-lr-fixed", args.position_lr_fixed),
    ):
        if value is not None:
            command += [flag, str(value)]
    if args.frame_gaussians:
        command += ["--frame-gaussian-dir", str(condition_dir / "gaussians")]
    if args.frame_stride != 1:
        # --start-frame/--end-frame index into whatever sequence train_4d.py
        # discovered, so the stride has to be expressed as the sequence itself.
        command += ["--frames", *args.sequence_names]
    run(command, condition_dir / "driver.log", dry_run=args.dry_run)


def train_scratch_frame(
    args: argparse.Namespace, frame_directory: Path, frame_output: Path
) -> None:
    """Train one frame independently, exactly like a static scene."""

    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "train.py"),
        "--data", str(frame_directory),
        "--config", str(args.scratch_config),
        "--output", str(frame_output),
        "--image-directory", args.image_directory,
        "--render-backend", args.render_backend,
        "--disable-training-evaluation",
    ]
    run(command, frame_output.parent / "driver.log", dry_run=args.dry_run)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


class HeldOutEvaluator:
    """Score checkpoints on the held-out cameras of each frame.

    The LPIPS network and each frame's held-out images are loaded once and
    reused across conditions, which is the difference between a sweep that
    spends its time training and one that spends it re-reading VGG weights.
    """

    def __init__(
        self,
        *,
        data_root: Path,
        config: Config,
        image_directory: str,
        render_backend: str,
    ) -> None:
        self.data_root = data_root
        self.config = config
        self.image_directory = image_directory
        self.device = resolve_device(config.runtime)
        self.dtype = resolve_dtype(config.runtime)
        self.renderer = GaussianRenderer(config.rendering, backend=render_backend)
        self._lpips: LPIPSMetric | None = None
        self._cameras: dict[int, list[Camera]] = {}

    @property
    def lpips(self) -> LPIPSMetric:
        if self._lpips is None:
            self._lpips = LPIPSMetric(device=self.device)
        return self._lpips

    def cameras(self, frame_directory: Path, frame: int) -> list[Camera]:
        """Return one frame's held-out cameras, caching them across conditions."""

        cached = self._cameras.get(frame)
        if cached is not None:
            return cached
        dataset = load_dataset(
            frame_directory,
            self.config,
            splits=("test",),
            load_points=False,
            image_directory=self.image_directory,
        )
        cameras = dataset.test
        if not cameras:
            raise RuntimeError(
                f"frame {frame} has no held-out cameras; evaluation must not "
                f"fall back to training views: {frame_directory}"
            )
        self._cameras[frame] = cameras
        return cameras

    def forget_frames(self) -> None:
        """Release cached images once a frame range is finished."""

        self._cameras.clear()

    def score(
        self, checkpoint: Path, frame_directory: Path, frame: int, output_json: Path
    ) -> tuple[dict[str, float], int]:
        """Evaluate one checkpoint, reusing a previous result when present."""

        cameras = self.cameras(frame_directory, frame)
        if output_json.is_file():
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            return (
                {
                    "psnr": float(payload["mean_psnr"]),
                    "ssim": float(payload["mean_ssim"]),
                    "lpips": float(payload["mean_lpips"]),
                },
                int(payload.get("num_gaussians", 0)),
            )
        state = read_checkpoint(checkpoint, map_location="cpu")
        # Every arm is scored through the same configuration so that rendering,
        # SSIM, and the active SH degree are identical across them.  The
        # warm-start config has progressive SH off, which pins the degree at
        # the full 3 -- the degree each arm's checkpoint actually reached,
        # whether it trained for 30,000 iterations or 100.
        model = model_from_checkpoint_state(
            state, self.config, device=self.device, dtype=self.dtype
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate_camera_set(
            model,
            self.renderer,
            cameras,
            output_json,
            ssim_window_size=self.config.loss.ssim_window_size,
            ssim_sigma=self.config.loss.ssim_sigma,
            ssim_k1=self.config.loss.ssim_k1,
            ssim_k2=self.config.loss.ssim_k2,
            lpips_metric=self.lpips,
        )
        count = model.num_gaussians
        # The Gaussian count belongs with the metrics: a reader of the JSON
        # should not have to go back to the checkpoint to learn it.
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        payload["num_gaussians"] = count
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return (
            {
                "psnr": result.mean_psnr,
                "ssim": result.mean_ssim,
                "lpips": result.mean_lpips,
            },
            count,
        )


# --------------------------------------------------------------------------
# per-frame training facts
# --------------------------------------------------------------------------


def frame_training_facts(frame_output: Path) -> tuple[float, float | None]:
    """Return ``(train_sec, final_loss)`` for one finished frame.

    ``train_sec`` is the trainer's own in-loop measurement, which excludes
    evaluation, checkpointing, and dataset loading, so budgets stay comparable.
    """

    telemetry_path = frame_output / "training_telemetry.json"
    train_sec = 0.0
    if telemetry_path.is_file():
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        train_sec = float(telemetry.get("training_loop_seconds", 0.0))
    loss: float | None = None
    log_path = frame_output / "train_log.jsonl"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "loss_total" in record:
                loss = float(record["loss_total"])
    return train_sec, loss


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def discover_frames(root: Path, pattern: str = "frame_*") -> list[Path]:
    found = sorted(path for path in root.glob(pattern) if path.is_dir())
    if not found:
        raise FileNotFoundError(f"no {pattern} directory below {root}")
    return found


def plan_conditions(
    args: argparse.Namespace,
    frames: tuple[int, ...],
    positions: tuple[int, ...],
) -> list[Condition]:
    """Turn the requested budgets into the sweep's arms."""

    conditions: list[Condition] = []
    for iterations in sorted(set(args.iters)):
        if iterations < 0:
            raise ValueError("--iters values must not be negative")
        name = "iter0" if iterations == 0 else f"warmstart_{iterations}"
        conditions.append(
            Condition(
                name=name,
                kind="iter0" if iterations == 0 else "warmstart",
                iterations=iterations,
                frames=frames,
                positions=positions,
            )
        )
    if args.scratch:
        if args.scratch_every < 1:
            raise ValueError("--scratch-every must be at least 1")
        scratch_iterations = load_config(args.scratch_config).training.iterations
        conditions.append(
            Condition(
                name=f"scratch_{scratch_iterations}",
                kind="scratch",
                iterations=scratch_iterations,
                frames=frames[:: args.scratch_every],
                # Independent per-frame training files its output by the real
                # frame number, so position and frame coincide for this arm.
                positions=frames[:: args.scratch_every],
            )
        )
    return conditions


def write_sweep_metadata(args: argparse.Namespace, conditions: list[Condition]) -> None:
    """Record what produced this sweep next to its results."""

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
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
        },
        "argv": sys.argv,
        "entry_point": "scripts/warmstart_iteration_sweep.py",
        "warm_start_config": str(args.config),
        "scratch_config": str(args.scratch_config),
        "frame1_checkpoint": str(args.frame1_checkpoint),
        "frame_stride": args.frame_stride,
        "sequence": getattr(args, "sequence_names", None),
        "conditions": [
            {
                "name": c.name,
                "kind": c.kind,
                "iterations": c.iterations,
                "frames": [c.frames[0], c.frames[-1]],
                "frame_count": len(c.frames),
            }
            for c in conditions
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweep_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.start_frame is None:
        args.start_frame = 1 + args.frame_stride
    if args.start_frame < 2:
        raise ValueError(
            "--start-frame must be at least 2; frame 1 comes from "
            "--frame1-checkpoint and is shared by every condition"
        )
    if not args.frame1_checkpoint.is_file():
        raise FileNotFoundError(
            f"frame-1 checkpoint does not exist: {args.frame1_checkpoint}"
        )

    frame_directories = discover_frames(args.data)
    last_frame = min(args.end_frame or len(frame_directories), len(frame_directories))
    # The sequence begins at dataset frame 1 whatever the stride: that is the
    # frame --frame1-checkpoint was trained on, and every arm inherits it.
    sequence = list(range(1, last_frame + 1, args.frame_stride))
    if len(sequence) < 2:
        raise ValueError(
            f"--frame-stride {args.frame_stride} leaves nothing to train "
            f"below frame {last_frame}"
        )
    if args.start_frame not in sequence:
        nearby = ", ".join(str(n) for n in sequence[1:5])
        raise ValueError(
            f"--start-frame {args.start_frame} is not on the stride-"
            f"{args.frame_stride} sequence; its trainable members begin "
            f"{nearby}, ..."
        )
    start_index = sequence.index(args.start_frame)
    frames = tuple(sequence[start_index:])
    positions = tuple(range(start_index + 1, len(sequence) + 1))
    # train_4d.py indexes --start-frame/--end-frame into the sequence it is
    # given, so a stride has to be passed as the sequence itself.
    args.sequence_names = [f"frame_{number:04d}" for number in sequence]

    config = load_config(args.config)
    conditions = plan_conditions(args, frames, positions)
    write_sweep_metadata(args, conditions)

    results_path = args.output / RESULTS_NAME
    done = recorded_pairs(read_results(results_path))

    stride_note = (
        "" if args.frame_stride == 1 else f", stride {args.frame_stride}"
    )
    print(
        f"=== warm-start iteration sweep: frames {frames[0]}..{frames[-1]} "
        f"({len(frames)} frames{stride_note}), {len(conditions)} conditions ===",
        flush=True,
    )
    for condition in conditions:
        pending = [f for f in condition.frames if (condition.name, f) not in done]
        print(
            f"  {condition.name:<18} {condition.kind:<10} "
            f"iters={condition.iterations:<6} frames={len(condition.frames):<4} "
            f"pending={len(pending)}",
            flush=True,
        )
    if args.dry_run:
        for condition in conditions:
            print(f"--- {condition.name} ---")
            if condition.kind == "warmstart":
                train_warmstart(condition, args, args.output / condition.name)
            elif condition.kind == "scratch":
                train_scratch_frame(
                    args,
                    frame_directories[condition.frames[0] - 1],
                    args.output / condition.name / f"frame_{condition.frames[0]:04d}",
                )
                print("  [dry-run] ... and so on for each scratch frame")
        return 0

    evaluator = HeldOutEvaluator(
        data_root=args.data,
        config=config,
        image_directory=args.image_directory,
        render_backend=args.render_backend,
    )

    for condition in conditions:
        condition_dir = args.output / condition.name
        pending = [f for f in condition.frames if (condition.name, f) not in done]
        if not pending:
            print(f"[{condition.name}] already complete", flush=True)
            continue
        started = time.monotonic()
        print(f"\n[{condition.name}] {len(pending)} frames to do", flush=True)

        if condition.kind == "warmstart":
            train_warmstart(condition, args, condition_dir)
        elif condition.kind == "scratch":
            for frame in pending:
                frame_output = (
                    condition_dir / f"frame_{condition.position_of(frame):04d}"
                )
                if not (frame_output / "training_telemetry.json").is_file():
                    train_scratch_frame(
                        args, frame_directories[frame - 1], frame_output
                    )

        for frame in condition.frames:
            if (condition.name, frame) in done:
                continue
            frame_directory = frame_directories[frame - 1]
            if condition.kind == "iter0":
                checkpoint = args.frame1_checkpoint
                train_sec, loss = 0.0, None
            else:
                frame_output = (
                    condition_dir / f"frame_{condition.position_of(frame):04d}"
                )
                checkpoint = (
                    frame_output
                    / "checkpoints"
                    / f"iteration_{condition.iterations:08d}.pt"
                )
                if not checkpoint.is_file():
                    # --keep-frame-checkpoints last removes all but the most
                    # recent, so a re-run can find the training done and the
                    # checkpoint gone.  Say so instead of failing obscurely.
                    raise FileNotFoundError(
                        f"{condition.name} frame {frame}: checkpoint is missing, "
                        f"so it can no longer be scored: {checkpoint}"
                    )
                train_sec, loss = frame_training_facts(frame_output)
            metrics, count = evaluator.score(
                checkpoint,
                frame_directory,
                frame,
                condition_dir / "metrics" / f"frame_{frame:04d}.json",
            )
            append_result(
                results_path,
                {
                    "condition": condition.name,
                    "iters": condition.iterations,
                    "frame": frame,
                    "psnr": f"{metrics['psnr']:.4f}",
                    "ssim": f"{metrics['ssim']:.5f}",
                    "lpips": f"{metrics['lpips']:.5f}",
                    "train_sec": f"{train_sec:.3f}",
                    "n_gaussians": count,
                    "loss": "" if loss is None else f"{loss:.6f}",
                },
            )
            done.add((condition.name, frame))
            print(
                f"  frame {frame:4d}  PSNR {metrics['psnr']:6.3f}  "
                f"SSIM {metrics['ssim']:.4f}  LPIPS {metrics['lpips']:.4f}  "
                f"{train_sec:7.1f}s  {count} Gaussians",
                flush=True,
            )
        print(
            f"[{condition.name}] done in {time.monotonic() - started:.0f}s",
            flush=True,
        )

    evaluator.forget_frames()
    print(f"\nresults: {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
