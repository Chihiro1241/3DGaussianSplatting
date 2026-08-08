from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from gaussian_splatting.config import load_config
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.model.initialization import (
    SH_DC_BASIS,
    generate_initial_points,
    initial_gaussian_scale,
    initial_neighbor_distance,
    initial_raw_opacity,
    initial_raw_quaternion,
    initial_raw_scale,
    initial_sh_dc,
    initialize_gaussian_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _raw_model(dtype: torch.dtype = torch.float64) -> GaussianModel:
    return GaussianModel(
        means_world=torch.zeros(2, 3, dtype=dtype),
        raw_quaternions=initial_raw_quaternion(2, dtype=dtype),
        raw_scales=torch.zeros(2, 3, dtype=dtype),
        raw_opacities=torch.zeros(2, 1, dtype=dtype),
        sh_dc=torch.zeros(2, 1, 3, dtype=dtype),
        sh_rest=torch.zeros(2, 15, 3, dtype=dtype),
        sh_degree=3,
    )


def test_gaussian_model_keeps_dc_and_rest_as_distinct_parameters() -> None:
    model = _raw_model()

    assert isinstance(model.sh_dc, nn.Parameter)
    assert isinstance(model.sh_rest, nn.Parameter)
    assert model.sh_dc is not model.sh_rest
    assert model.sh_coefficients.shape == (2, 16, 3)
    assert model.sh_coefficients.dtype == torch.float64


def test_gaussian_model_transformed_parameters() -> None:
    model = _raw_model()

    parameters = model.transformed_parameters()

    torch.testing.assert_close(
        parameters.quaternions,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64),
    )
    torch.testing.assert_close(parameters.scales, torch.ones(2, 3, dtype=torch.float64))
    torch.testing.assert_close(
        parameters.opacities, torch.full((2, 1), 0.5, dtype=torch.float64)
    )


def test_gaussian_model_rejects_non_degree_three() -> None:
    with pytest.raises(ValueError, match="sh_degree=3"):
        GaussianModel(
            means_world=torch.zeros(1, 3),
            raw_quaternions=torch.zeros(1, 4),
            raw_scales=torch.zeros(1, 3),
            raw_opacities=torch.zeros(1, 1),
            sh_dc=torch.zeros(1, 1, 3),
            sh_rest=torch.zeros(1, 15, 3),
            sh_degree=2,
        )


def test_initial_neighbor_distance__eq_initial_neighbor_distance() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    distances = initial_neighbor_distance(points)

    torch.testing.assert_close(
        distances, torch.tensor([1.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0], dtype=torch.float64)
    )


def test_scale_and_opacity_initialization_round_trip() -> None:
    squared_distances = torch.tensor([0.0, 4.0], dtype=torch.float64)
    scales = initial_gaussian_scale(squared_distances, epsilon_scale=1e-6)
    raw_scales = initial_raw_scale(scales)
    opacities = torch.full((2, 1), 0.1, dtype=torch.float64)
    raw_opacities = initial_raw_opacity(opacities)

    torch.testing.assert_close(torch.exp(raw_scales), scales)
    torch.testing.assert_close(torch.sigmoid(raw_opacities), opacities)
    torch.testing.assert_close(scales[0], torch.full((3,), 1e-3, dtype=torch.float64))


def test_initial_sh_dc__eq_initial_sh_dc() -> None:
    colors = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float64)

    coefficients = initial_sh_dc(colors)
    reconstructed = 0.5 + SH_DC_BASIS * coefficients[:, 0]

    torch.testing.assert_close(reconstructed, colors)


def test_generate_initial_points_is_seeded_and_inside_aabb() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config = replace(
        config,
        runtime=replace(config.runtime, device="cpu", dtype="float64", seed=7),
        initialization=replace(config.initialization, num_gaussians=4),
    )
    target = [1.0, 2.0, 3.0]

    first = generate_initial_points(target, config, torch.Generator().manual_seed(7))
    second = generate_initial_points(target, config, torch.Generator().manual_seed(7))

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert first[0].dtype == torch.float64
    assert torch.all(first[0] >= torch.tensor(target, dtype=torch.float64) - 1.0)
    assert torch.all(first[0] <= torch.tensor(target, dtype=torch.float64) + 1.0)


def test_initialize_gaussian_model_builds_documented_shapes() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config = replace(config, initialization=replace(config.initialization, num_gaussians=4))
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    colors = torch.full((4, 3), 0.5)

    model = initialize_gaussian_model(points, colors, config)

    assert model.means_world.shape == (4, 3)
    assert model.raw_quaternions.shape == (4, 4)
    assert model.raw_scales.shape == (4, 3)
    assert model.raw_opacities.shape == (4, 1)
    assert model.sh_dc.shape == (4, 1, 3)
    assert model.sh_rest.shape == (4, 15, 3)
