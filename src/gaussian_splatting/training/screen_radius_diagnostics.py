"""Read-only sidecar diagnostics for one screen-radius statistics window."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import RenderResult
from gaussian_splatting.training.density_control import ScreenSpaceDensityStatistics


def _distribution(values: torch.Tensor) -> dict[str, int | float | None]:
    values = values.detach().to(device="cpu", dtype=torch.float64)
    if values.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    quantiles = torch.quantile(
        values, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64)
    )
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "median": float(quantiles[0].item()),
        "mean": float(values.mean().item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "max": float(values.max().item()),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().to(device="cpu", dtype=torch.float64)
    right = right.detach().to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.numel() < 2 or left.var(unbiased=False) == 0 or right.var(unbiased=False) == 0:
        return None
    value = torch.corrcoef(torch.stack((left, right)))[0, 1].item()
    return float(value) if math.isfinite(value) else None


class ScreenRadiusDiagnostic:
    """Track radius argmax context without mutating training state or RNG."""

    def __init__(self, output_directory: str | Path, *, start: int, end: int) -> None:
        if start <= 0 or end < start:
            raise ValueError("diagnostic iterations must satisfy 0 < start <= end")
        self.output_directory = Path(output_directory)
        self.start = int(start)
        self.end = int(end)
        self._max_radius: torch.Tensor | None = None
        self._argmax_iteration: torch.Tensor | None = None
        self._depth_at_argmax: torch.Tensor | None = None
        self._argmax_view: list[str | None] | None = None

    def observe(
        self,
        render: RenderResult,
        *,
        iteration: int,
        view_name: str | None,
        model: GaussianModel,
        statistics: ScreenSpaceDensityStatistics,
    ) -> None:
        """Observe one completed accumulation and save immediately at ``end``."""

        if iteration < self.start or iteration > self.end:
            return
        count = model.num_gaussians
        if self._max_radius is None:
            self._max_radius = torch.zeros(count, dtype=torch.int64)
            self._argmax_iteration = torch.full((count,), -1, dtype=torch.int64)
            self._depth_at_argmax = torch.full((count,), torch.nan, dtype=torch.float64)
            self._argmax_view = [None] * count
        if self._max_radius.shape != (count,):
            raise RuntimeError("Gaussian count changed inside diagnostic window")

        projected = render.projected
        indices = projected.original_indices.detach().cpu()
        radii = projected.radii.detach().cpu()
        depths = projected.depths.detach().to(device="cpu", dtype=torch.float64)
        assert self._argmax_iteration is not None
        assert self._depth_at_argmax is not None
        assert self._argmax_view is not None
        updates = radii > self._max_radius[indices]
        update_indices = indices[updates]
        self._max_radius[update_indices] = radii[updates]
        self._argmax_iteration[update_indices] = int(iteration)
        self._depth_at_argmax[update_indices] = depths[updates]
        for index in update_indices.tolist():
            self._argmax_view[index] = view_name

        if iteration == self.end:
            self._save(model, statistics, iteration)

    def _save(
        self,
        model: GaussianModel,
        statistics: ScreenSpaceDensityStatistics,
        iteration: int,
    ) -> None:
        assert self._max_radius is not None
        assert self._argmax_iteration is not None
        assert self._depth_at_argmax is not None
        assert self._argmax_view is not None
        statistics_radius = statistics.max_screen_radius.detach().cpu()
        if not torch.equal(self._max_radius, statistics_radius):
            raise RuntimeError("diagnostic radius does not match density statistics")

        with torch.no_grad():
            transformed = model.transformed_parameters()
            world_scale = transformed.scales.amax(dim=-1).detach().cpu().clone()
            opacity = transformed.opacities.squeeze(-1).detach().cpu().clone()
        denominator = (
            statistics.position_gradient_denominator.detach().cpu().clone()
        )
        gradient_accumulator = (
            statistics.position_gradient_accumulator.detach().cpu().clone()
        )
        mean_gradient = statistics.mean_position_gradient().detach().cpu().clone()
        radius = self._max_radius.clone()
        observed = denominator > 0
        large_screen = radius > 20
        raw = {
            "iteration": int(iteration),
            "window_start": self.start,
            "window_end": self.end,
            "gaussian_index": torch.arange(radius.numel(), dtype=torch.int64),
            "max_screen_radius": radius,
            "argmax_iteration": self._argmax_iteration.clone(),
            "argmax_view": list(self._argmax_view),
            "depth_at_argmax": self._depth_at_argmax.clone(),
            "maximum_world_scale": world_scale,
            "opacity": opacity,
            "position_gradient_denominator": denominator,
            "position_gradient_accumulator": gradient_accumulator,
            "mean_position_gradient": mean_gradient,
        }
        summary = {
            "iteration": int(iteration),
            "window": {"start": self.start, "end": self.end},
            "gaussian_count": int(radius.numel()),
            "max_screen_radius": _distribution(radius),
            "radius_counts": {
                "greater_than_20": int((radius > 20).sum().item()),
                "greater_than_50": int((radius > 50).sum().item()),
                "greater_than_100": int((radius > 100).sum().item()),
            },
            "observed": int(observed.sum().item()),
            "unobserved": int((~observed).sum().item()),
            "mean_position_gradient": _distribution(mean_gradient[observed]),
            "correlations": {
                "radius_world_scale_all": _pearson(radius, world_scale),
                "radius_opacity_all": _pearson(radius, opacity),
                "radius_world_scale_observed": _pearson(
                    radius[observed], world_scale[observed]
                ),
                "radius_opacity_observed": _pearson(
                    radius[observed], opacity[observed]
                ),
            },
            "large_screen": {
                "count": int(large_screen.sum().item()),
                "maximum_world_scale": _distribution(world_scale[large_screen]),
                "opacity": _distribution(opacity[large_screen]),
            },
        }
        self.output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"density_pre_event_{iteration:08d}"
        torch.save(raw, self.output_directory / f"{stem}.pt")
        (self.output_directory / f"{stem}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


__all__ = ["ScreenRadiusDiagnostic"]
