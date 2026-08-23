from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.projection import ProjectedGaussians
from gaussian_splatting.renderer.renderer import RenderResult
from gaussian_splatting.training.density_control import ScreenSpaceDensityStatistics
from gaussian_splatting.training.screen_radius_diagnostics import (
    ScreenRadiusDiagnostic,
)


def _model() -> GaussianModel:
    return GaussianModel(
        means_world=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        raw_quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        raw_scales=torch.log(torch.tensor([[0.1, 0.2, 0.3]] * 3)),
        raw_opacities=torch.logit(torch.tensor([[0.1], [0.2], [0.3]])),
        sh_dc=torch.zeros((3, 1, 3)),
        sh_rest=torch.zeros((3, 15, 3)),
    )


def _render(indices: list[int], radii: list[int], depths: list[float]) -> RenderResult:
    count = len(indices)
    projected = ProjectedGaussians(
        original_indices=torch.tensor(indices, dtype=torch.int64),
        means_camera=torch.zeros((count, 3)),
        means_screen=torch.zeros((count, 2)),
        depths=torch.tensor(depths),
        covariances_3d=torch.zeros((count, 3, 3)),
        covariances_2d=torch.zeros((count, 2, 2)),
        inverse_covariances_2d=torch.zeros((count, 2, 2)),
        colors=torch.zeros((count, 3)),
        opacities=torch.zeros((count, 1)),
        radii=torch.tensor(radii, dtype=torch.int64),
        rectangles=torch.zeros((count, 4), dtype=torch.int64),
    )
    visible = torch.zeros(3, dtype=torch.bool)
    visible[projected.original_indices] = True
    return RenderResult(
        image=torch.zeros((3, 1, 1)),
        final_transmittance=torch.ones((1, 1)),
        projected=projected,
        visible_mask=visible,
    )


def test_tracks_argmax_by_original_index_and_does_not_mutate_training_state(
    tmp_path: Path,
) -> None:
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    diagnostic = ScreenRadiusDiagnostic(tmp_path, start=1, end=2)
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    first = _render([2, 0], [9, 5], [2.9, 2.5])
    statistics.max_screen_radius[:] = torch.tensor([5, 0, 9])
    statistics.position_gradient_denominator[:] = torch.tensor([1, 0, 1])
    diagnostic.observe(
        first, iteration=1, view_name="first.png", model=model, statistics=statistics
    )
    second = _render([0, 1, 2], [4, 7, 12], [4.0, 3.7, 3.2])
    statistics.max_screen_radius[:] = torch.tensor([5, 7, 12])
    statistics.position_gradient_denominator[:] = torch.tensor([2, 1, 2])
    statistics_before = statistics.state_dict()
    diagnostic.observe(
        second, iteration=2, view_name="second.png", model=model, statistics=statistics
    )

    raw = torch.load(
        tmp_path / "density_pre_event_00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["max_screen_radius"].tolist() == [5, 7, 12]
    assert raw["argmax_iteration"].tolist() == [1, 2, 2]
    assert raw["argmax_view"] == ["first.png", "second.png", "second.png"]
    assert "position_gradient_accumulator" in raw
    assert "mean_position_gradient" in raw
    assert torch.allclose(
        raw["depth_at_argmax"],
        torch.tensor([2.5, 3.7, 3.2], dtype=torch.float64),
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, model_before[name])
    for name, value in statistics.state_dict().items():
        assert torch.equal(value, statistics_before[name])
    assert random.getstate() == python_rng
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_rng[0]
    assert np.array_equal(current_numpy[1], numpy_rng[1])
    assert current_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_equal_radius_does_not_replace_argmax_context(tmp_path: Path) -> None:
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    diagnostic = ScreenRadiusDiagnostic(tmp_path, start=1, end=2)
    statistics.max_screen_radius[0] = 8
    statistics.position_gradient_denominator[0] = 1
    diagnostic.observe(
        _render([0], [8], [2.0]),
        iteration=1,
        view_name="first.png",
        model=model,
        statistics=statistics,
    )
    statistics.position_gradient_denominator[0] = 2
    diagnostic.observe(
        _render([0], [8], [1.0]),
        iteration=2,
        view_name="second.png",
        model=model,
        statistics=statistics,
    )
    raw = torch.load(
        tmp_path / "density_pre_event_00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["argmax_iteration"][0].item() == 1
    assert raw["argmax_view"][0] == "first.png"
    assert raw["depth_at_argmax"][0].item() == 2.0
