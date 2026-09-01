"""Run or resume the guarded 21-scene CUDA paper benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("output/paper_benchmark/manifest.json"))
    parser.add_argument("--scene", action="append", help="optional exact scene filter")
    parser.add_argument("--continue-on-oom", action="store_true", help="continue after OOM instead of stopping the batch")
    return parser


def _descendant_in_group(pid: int, process_group: int) -> bool:
    try:
        return os.getpgid(pid) == process_group
    except (ProcessLookupError, PermissionError):
        return False


def _query_gpu_processes() -> list[tuple[int, float]]:
    command = [
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    values: list[tuple[int, float]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit():
            try:
                values.append((int(fields[0]), float(fields[1])))
            except ValueError:
                pass
    return values


class ProcessVramMonitor:
    def __init__(self, process_group: int) -> None:
        self.process_group = process_group
        self.peak_mib: float | None = None
        self.samples: list[dict[str, float]] = []
        self._started_at: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            total = sum(
                memory for pid, memory in _query_gpu_processes()
                if _descendant_in_group(pid, self.process_group)
            )
            if total > 0:
                self.peak_mib = max(self.peak_mib or 0.0, total)
                if self._started_at is not None:
                    self.samples.append(
                        {
                            "elapsed_seconds": time.monotonic() - self._started_at,
                            "process_vram_MiB": total,
                        }
                    )

    def __enter__(self) -> "ProcessVramMonitor":
        self._started_at = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


def _run_logged(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    vram_samples_path: Path | None = None,
) -> tuple[int, float | None]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n[{_now()}] $ {' '.join(command)}\n")
        stdout.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        group = os.getpgid(process.pid)
        try:
            with ProcessVramMonitor(group) as monitor:
                return_code = process.wait()
            if vram_samples_path is not None:
                _write_json(
                    vram_samples_path,
                    {
                        "sampling_interval_seconds": 0.5,
                        "peak_process_vram_MiB": monitor.peak_mib,
                        "samples": monitor.samples,
                    },
                )
            return return_code, monitor.peak_mib
        except KeyboardInterrupt:
            try:
                os.killpg(group, signal.SIGINT)
                process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            raise


def _image_directory(dataset: str, scene: str) -> str:
    if dataset != "Mip-NeRF360":
        return "images"
    return "images_4" if scene in {"bicycle", "flowers", "garden", "stump", "treehill"} else "images_2"


def _config(dataset: str) -> Path:
    name = "paper_benchmark_synthetic.yaml" if dataset == "Synthetic NeRF" else "paper_benchmark_real.yaml"
    return ROOT / "configs" / name


def _checkpoint_iteration(path: Path) -> int:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state["iteration"])


def _checkpoint_metrics(path: Path) -> dict[str, int]:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    tensors = state["model_state_dict"]
    parameter_names = ("means_world", "raw_quaternions", "raw_scales", "raw_opacities", "sh_dc", "sh_rest")
    return {
        "iteration": int(state["iteration"]),
        "gaussian_count": int(tensors["means_world"].shape[0]),
        "model_parameter_bytes": int(sum(tensors[name].numel() * tensors[name].element_size() for name in parameter_names)),
        "checkpoint_file_bytes": path.stat().st_size,
    }


def _tail_contains_oom(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")[-200_000:].lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _wait_for_gpu_idle(timeout_seconds: float = 120.0) -> bool:
    """Wait until no compute process remains before starting another scene."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        if not _query_gpu_processes():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1.0)


def _training_snapshot(run_dir: Path) -> dict[str, Any]:
    telemetry = _read_json(run_dir / "training_telemetry.json", {})
    snapshot: dict[str, Any] = {
        "last_iteration": telemetry.get("iteration"),
        "gaussian_count": telemetry.get("gaussian_count"),
        "peak_cuda_memory_allocated_MiB": telemetry.get(
            "peak_cuda_memory_allocated_MiB"
        ),
        "peak_cuda_memory_reserved_MiB": telemetry.get(
            "peak_cuda_memory_reserved_MiB"
        ),
    }
    log_path = run_dir / "train_log.jsonl"
    maximum: int | None = None
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            count = record.get("gaussian_count")
            if isinstance(count, int):
                maximum = count if maximum is None else max(maximum, count)
        snapshot["maximum_gaussian_count"] = maximum
    return snapshot


def _run_scene(row: dict[str, Any], output_root: Path) -> str:
    scene = row["scene"]
    run_dir = Path(row["estimated_output_path"])
    state_path = run_dir / "run_state.json"
    state = _read_json(state_path, {"dataset": row["dataset"], "scene": scene, "attempts": []})
    if state.get("status") == "COMPLETED":
        print(f"SKIP completed: {scene}", flush=True)
        return "COMPLETED"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    latest = run_dir / "checkpoints" / "latest.pt"
    command = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--data", row["dataset_path"], "--config", str(_config(row["dataset"])),
        "--output", str(run_dir), "--render-backend", "cuda",
        "--allow-existing-output",
        "--image-directory", _image_directory(row["dataset"], scene),
        "--milestone-iterations", "7000", "30000",
        "--disable-training-evaluation",
    ]
    if latest.is_file():
        command.extend(["--resume", str(latest)])
    attempt = {"started_at": _now(), "resume_iteration": _checkpoint_iteration(latest) if latest.is_file() else 0}
    state.update({"status": "RUNNING", "started_at": state.get("started_at", attempt["started_at"]), "last_attempt_started_at": attempt["started_at"]})
    _write_json(state_path, state)
    started = time.monotonic()
    try:
        return_code, peak_process = _run_logged(
            command,
            stdout_path,
            stderr_path,
            vram_samples_path=(
                run_dir / f"vram_samples_attempt_{len(state['attempts']) + 1:02d}.json"
            ),
        )
    except KeyboardInterrupt:
        attempt.update({"ended_at": _now(), "duration_seconds": time.monotonic() - started, "status": "INTERRUPTED"})
        state["attempts"].append(attempt)
        state.update({"status": "INTERRUPTED", "ended_at": attempt["ended_at"]})
        _write_json(state_path, state)
        raise
    attempt.update({"ended_at": _now(), "duration_seconds": time.monotonic() - started, "return_code": return_code, "peak_process_vram_MiB": peak_process})
    if return_code != 0:
        status = "OOM" if _tail_contains_oom(stderr_path) else "FAILED"
        attempt["status"] = status
        snapshot = _training_snapshot(run_dir)
        attempt.update(snapshot)
        state["attempts"].append(attempt)
        state.update(
            {
                "status": status,
                "ended_at": attempt["ended_at"],
                "peak_process_vram_MiB": max(
                    [a.get("peak_process_vram_MiB") or 0 for a in state["attempts"]]
                )
                or None,
                **snapshot,
            }
        )
        _write_json(state_path, state)
        return status

    checkpoint_records: dict[str, Any] = {}
    evaluation_wall_seconds = 0.0
    evaluation_peak_process: float | None = None
    for iteration in (7000, 30000):
        checkpoint = run_dir / "checkpoints" / f"iteration_{iteration:08d}.pt"
        if not checkpoint.is_file():
            status = "FAILED"
            attempt.update({"status": status, "error": f"missing milestone checkpoint {checkpoint}"})
            state["attempts"].append(attempt)
            state.update({"status": status, "ended_at": attempt["ended_at"]})
            _write_json(state_path, state)
            return status
        try:
            checkpoint_records[str(iteration)] = _checkpoint_metrics(checkpoint)
        except BaseException as error:
            status = "FAILED"
            attempt.update(
                {
                    "status": status,
                    "error": f"corrupted checkpoint {checkpoint}: {error}",
                }
            )
            state["attempts"].append(attempt)
            state.update(
                {
                    "status": status,
                    "ended_at": _now(),
                    **_training_snapshot(run_dir),
                }
            )
            _write_json(state_path, state)
            return status
        evaluation = run_dir / "metrics" / f"test_{iteration:08d}.json"
        if not evaluation.is_file():
            evaluation_command = [
                sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                "--data", row["dataset_path"], "--checkpoint", str(checkpoint),
                "--split", "test", "--output", str(evaluation),
                "--render-backend", "cuda", "--image-directory", _image_directory(row["dataset"], scene),
            ]
            evaluation_started = time.monotonic()
            evaluation_code, evaluation_peak = _run_logged(evaluation_command, stdout_path, stderr_path)
            evaluation_wall_seconds += time.monotonic() - evaluation_started
            if evaluation_peak is not None:
                evaluation_peak_process = max(evaluation_peak_process or 0.0, evaluation_peak)
            if evaluation_code != 0:
                status = "OOM" if _tail_contains_oom(stderr_path) else "EVALUATION_FAILED"
                attempt.update({
                    "status": status,
                    "evaluation_iteration": iteration,
                    "evaluation_wall_time_seconds": evaluation_wall_seconds,
                    "duration_seconds": time.monotonic() - started,
                    "ended_at": _now(),
                })
                state["attempts"].append(attempt)
                state.update({"status": status, "ended_at": _now(), "peak_process_vram_MiB": peak_process})
                _write_json(state_path, state)
                return status

    attempt.update({
        "status": "COMPLETED",
        "training_process_wall_time_seconds": attempt["duration_seconds"],
        "evaluation_wall_time_seconds": evaluation_wall_seconds,
        "peak_evaluation_process_vram_MiB": evaluation_peak_process,
        "duration_seconds": time.monotonic() - started,
        "ended_at": _now(),
    })
    state["attempts"].append(attempt)
    snapshot = _training_snapshot(run_dir)
    state.update({
        "status": "COMPLETED", "ended_at": _now(), "checkpoints": checkpoint_records,
        "peak_process_vram_MiB": max([a.get("peak_process_vram_MiB") or 0 for a in state["attempts"]] + [peak_process or 0]) or None,
        "total_run_wall_time_seconds": sum(float(a["duration_seconds"]) for a in state["attempts"]),
        **snapshot,
    })
    _write_json(state_path, state)
    return "COMPLETED"


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_started") not in {False, True}:
        raise RuntimeError("manifest training_started flag is invalid")
    blockers = manifest.get("blocking_conditions", [])
    unavailable = [row["scene"] for row in manifest["scenes"] if row["status"] != "AVAILABLE"]
    if blockers or unavailable:
        raise SystemExit(
            "benchmark preflight BLOCKED; rerun dry-run after resolving: "
            f"conditions={[item['condition'] for item in blockers]}, scenes={unavailable}"
        )
    if manifest.get("training_started") is False:
        manifest["training_started"] = True
        manifest["benchmark_status"] = "RUNNING"
        manifest["training_started_at"] = _now()
        _write_json(manifest_path, manifest)
    selected = set(args.scene or [])
    unknown = selected - {row["scene"] for row in manifest["scenes"]}
    if unknown:
        raise SystemExit(f"unknown scenes: {sorted(unknown)}")
    rows = [row for row in manifest["scenes"] if not selected or row["scene"] in selected]
    output_root = Path(manifest["output_root"])
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if not _wait_for_gpu_idle():
            raise RuntimeError("GPU did not become idle before the next scene")
        try:
            status = _run_scene(row, output_root)
        except KeyboardInterrupt:
            raise
        except BaseException as error:
            status = "FAILED"
            run_dir = Path(row["estimated_output_path"])
            state_path = run_dir / "run_state.json"
            state = _read_json(
                state_path,
                {"dataset": row["dataset"], "scene": row["scene"], "attempts": []},
            )
            state.update(
                {
                    "status": status,
                    "ended_at": _now(),
                    "runner_error": repr(error),
                    **_training_snapshot(run_dir),
                }
            )
            _write_json(state_path, state)
        if not _wait_for_gpu_idle():
            raise RuntimeError("GPU did not become idle after scene subprocess exit")
        run_dir = Path(row["estimated_output_path"])
        seven = _read_json(run_dir / "metrics" / "test_00007000.json", {})
        thirty = _read_json(run_dir / "metrics" / "test_00030000.json", {})
        state = _read_json(run_dir / "run_state.json", {})
        paper_30k = row.get("paper_psnr_30000")
        ours_30k = thirty.get("mean_psnr")
        delta_30k = (
            None if paper_30k is None or ours_30k is None else ours_30k - paper_30k
        )
        next_scene = rows[index]["scene"] if index < len(rows) else "none"
        print(f"[{index}/{len(rows)}] {row['scene']} {status}", flush=True)
        print(f"7K PSNR: {seven.get('mean_psnr', 'N/A')}", flush=True)
        print(f"30K PSNR: {ours_30k if ours_30k is not None else 'N/A'}", flush=True)
        print(f"Δ30K: {delta_30k if delta_30k is not None else 'N/A'}", flush=True)
        print(f"runtime: {state.get('total_run_wall_time_seconds', 'N/A')}", flush=True)
        print(f"peak VRAM: {state.get('peak_process_vram_MiB', 'N/A')}", flush=True)
        print(f"next: {next_scene}", flush=True)
        if status != "COMPLETED":
            failures.append({"dataset": row["dataset"], "scene": row["scene"], "status": status})
            _write_json(output_root / "results" / "failures.json", failures)
            if status == "OOM" and not args.continue_on_oom:
                break
    _write_json(output_root / "results" / "failures.json", failures)
    generator = [sys.executable, str(ROOT / "scripts" / "generate_paper_benchmark_report.py"), "--manifest", str(manifest_path)]
    subprocess.run(generator, cwd=ROOT, check=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_status"] = "COMPLETED_WITH_FAILURES" if failures else "COMPLETED"
    manifest["training_ended_at"] = _now()
    _write_json(manifest_path, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
