"""Synchronized wall-clock profiling for training iterations."""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


CORE_STAGES = (
    "parameter_transform",
    "projection",
    "rasterization",
    "loss",
    "backward",
    "optimizer",
)
SUMMARY_STAGES = (*CORE_STAGES, "other", "total")


class WallClockIterationTimer:
    """Measure one iteration with synchronization at every timed boundary."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.stage_times_ms: dict[str, float] = {}
        self._synchronize()
        self._started_at = time.perf_counter()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Measure a stage after all earlier CUDA work has completed."""

        if stage in self.stage_times_ms:
            raise ValueError(f"stage has already been measured: {stage}")
        self._synchronize()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.stage_times_ms[stage] = (
                time.perf_counter() - started_at
            ) * 1000.0

    def finish(self) -> tuple[float, dict[str, float]]:
        """Finish the synchronized iteration and derive unclassified time."""

        self._synchronize()
        total_ms = (time.perf_counter() - self._started_at) * 1000.0
        missing = [stage for stage in CORE_STAGES if stage not in self.stage_times_ms]
        if missing:
            raise RuntimeError(f"profiled iteration is missing stages: {missing}")
        core_ms = sum(self.stage_times_ms[stage] for stage in CORE_STAGES)
        times = dict(self.stage_times_ms)
        # Tiny negative values are possible only through timer resolution/overhead.
        times["other"] = max(0.0, total_ms - core_ms)
        times["total"] = total_ms
        return total_ms, times


def _statistics(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class TrainingProfiler:
    """Collect a bounded warm-up and profiling window and write its report."""

    def __init__(
        self,
        *,
        output_directory: str | Path,
        device: torch.device,
        dtype: torch.dtype,
        render_backend: str = "reference",
        warmup_steps: int = 5,
        profile_steps: int = 10,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("profile warm-up steps must be non-negative")
        if profile_steps <= 0:
            raise ValueError("profile steps must be positive")
        self.output_path = (
            Path(output_directory) / "profiling" / "training_profile.json"
        )
        self.device = torch.device(device)
        self.dtype = dtype
        self.render_backend = render_backend
        self.warmup_steps = int(warmup_steps)
        self.profile_steps = int(profile_steps)
        self.iterations: list[dict[str, object]] = []
        self._warmup_count = 0
        self._normal_profile_count = 0
        self._finalized = False

    @property
    def active(self) -> bool:
        return not self._finalized and (
            self._warmup_count < self.warmup_steps
            or self._normal_profile_count < self.profile_steps
        )

    def begin_iteration(self) -> WallClockIterationTimer | None:
        if not self.active:
            return None
        return WallClockIterationTimer(self.device)

    def record_iteration(
        self,
        *,
        iteration: int,
        gaussian_count: int,
        visible_gaussian_count: int,
        width: int,
        height: int,
        density_control_event: bool,
        opacity_reset_event: bool,
        stage_times: dict[str, float],
    ) -> None:
        phase = "warmup" if self._warmup_count < self.warmup_steps else "profile"
        if density_control_event:
            category = "density_control_event"
        elif opacity_reset_event:
            category = "opacity_reset_event"
        else:
            category = "normal"
        self.iterations.append(
            {
                "iteration": int(iteration),
                "phase": phase,
                "category": category,
                "gaussian_count": int(gaussian_count),
                "visible_gaussian_count": int(visible_gaussian_count),
                "image_width": int(width),
                "image_height": int(height),
                "density_control_event": bool(density_control_event),
                "opacity_reset_event": bool(opacity_reset_event),
                "stage_times_ms": stage_times,
            }
        )
        if phase == "warmup":
            self._warmup_count += 1
        elif category == "normal":
            self._normal_profile_count += 1
        if not self.active:
            self.finalize()

    def _summarize(self, records: list[dict[str, object]]) -> dict[str, object]:
        stage_summaries: dict[str, object] = {}
        for stage in SUMMARY_STAGES:
            values = [
                float(record["stage_times_ms"][stage])  # type: ignore[index]
                for record in records
            ]
            stage_summaries[stage] = _statistics(values)

        total_stats = stage_summaries["total"]
        mean_total = (
            float(total_stats["mean_ms"])  # type: ignore[index]
            if total_stats is not None
            else 0.0
        )
        shares: dict[str, float | None] = {}
        for stage in (*CORE_STAGES, "other"):
            stats = stage_summaries[stage]
            shares[stage] = (
                100.0 * float(stats["mean_ms"]) / mean_total  # type: ignore[index]
                if stats is not None and mean_total > 0.0
                else None
            )

        combined_projection = [
            float(record["stage_times_ms"]["parameter_transform"])  # type: ignore[index]
            + float(record["stage_times_ms"]["projection"])  # type: ignore[index]
            for record in records
        ]
        combined_stats = _statistics(combined_projection)
        combined_share = (
            100.0 * float(combined_stats["mean_ms"]) / mean_total
            if combined_stats is not None and mean_total > 0.0
            else None
        )
        return {
            "iteration_count": len(records),
            "stages": stage_summaries,
            "stage_share_percent": shares,
            "projection_combined": {
                "statistics": combined_stats,
                "share_percent": combined_share,
            },
            "rasterization_share_percent": shares["rasterization"],
        }

    def _profile_records(self, category: str) -> list[dict[str, object]]:
        return [
            record
            for record in self.iterations
            if record["phase"] == "profile" and record["category"] == category
        ]

    def _payload(self) -> dict[str, object]:
        gpu_name = None
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
        normal = self._profile_records("normal")
        density = self._profile_records("density_control_event")
        opacity = self._profile_records("opacity_reset_event")
        profiled = [record for record in self.iterations if record["phase"] == "profile"]
        gaussian_counts = sorted({int(record["gaussian_count"]) for record in profiled})
        resolutions = sorted(
            {
                (int(record["image_width"]), int(record["image_height"]))
                for record in profiled
            }
        )
        normal_summary = self._summarize(normal)
        return {
            "environment": {
                "device": str(self.device),
                "gpu_name": gpu_name,
                "dtype": str(self.dtype).removeprefix("torch."),
            },
            "configuration": {
                "warmup_steps": self.warmup_steps,
                "profile_steps": self.profile_steps,
                "profile_steps_definition": "normal_iterations",
                "render_backend": self.render_backend,
                "timing_method": "synchronized_wall_clock_perf_counter",
            },
            "gaussian_counts": gaussian_counts,
            "resolutions": [
                {"width": width, "height": height} for width, height in resolutions
            ],
            "iterations": self.iterations,
            "summaries": {
                "normal_iterations": normal_summary,
                "density_control_event_iterations": self._summarize(density),
                "opacity_reset_event_iterations": self._summarize(opacity),
            },
            "rasterization_share_percent": normal_summary[
                "rasterization_share_percent"
            ],
        }

    @staticmethod
    def _format_value(value: float | None) -> str:
        return "n/a" if value is None else f"{value:12.3f}"

    def _print_report(self, payload: dict[str, object]) -> None:
        summary = payload["summaries"]["normal_iterations"]  # type: ignore[index]
        stages = summary["stages"]  # type: ignore[index]
        shares = summary["stage_share_percent"]  # type: ignore[index]
        labels = (
            ("parameter transform", "parameter_transform"),
            ("projection", "projection"),
            ("rasterization", "rasterization"),
            ("loss", "loss"),
            ("backward", "backward"),
            ("optimizer", "optimizer"),
            ("other", "other"),
        )
        print("\nTraining performance profile")
        print(f"normal iterations: {summary['iteration_count']}")  # type: ignore[index]
        print(f"{'Stage':24s} {'Mean [ms]':>12s} {'Share [%]':>12s}")
        print("-" * 50)
        for label, key in labels:
            stats = stages[key]
            mean = None if stats is None else float(stats["mean_ms"])
            share = shares[key]
            mean_text = self._format_value(mean)
            share_text = "n/a" if share is None else f"{float(share):12.1f}"
            print(f"{label:24s} {mean_text:>12s} {share_text:>12s}")
        print("-" * 50)
        total_stats = stages["total"]
        total = None if total_stats is None else float(total_stats["mean_ms"])
        total_share = "100.0" if total is not None else "n/a"
        print(
            f"{'total':24s} {self._format_value(total):>12s} "
            f"{total_share:>12s}"
        )
        raster_share = payload["rasterization_share_percent"]
        raster_text = "n/a" if raster_share is None else f"{float(raster_share):.1f} %"
        print(f"\nrasterization share: {raster_text}")
        print(f"profile JSON: {self.output_path}")

    def finalize(self) -> None:
        if self._finalized:
            return
        payload = self._payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._print_report(payload)
        self._finalized = True


__all__ = ["TrainingProfiler", "WallClockIterationTimer"]
