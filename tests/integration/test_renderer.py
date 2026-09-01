from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel, GaussianParameters
from gaussian_splatting.training.density_control import ScreenSpaceDensityStatistics
from gaussian_splatting.renderer.renderer import GaussianRenderer, RenderResult


class _OneGaussianModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.means = nn.Parameter(torch.tensor([[0.1, 0.0, 2.0]]))
        self.opacity = nn.Parameter(torch.tensor([[0.5]]))
        self.color_coefficients = nn.Parameter(torch.zeros(1, 16, 3))
        self.register_buffer("quaternions", torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        self.register_buffer("scales", torch.full((1, 3), 0.1))
        self.call_count = 0

    def transformed_parameters(self) -> GaussianParameters:
        self.call_count += 1
        return GaussianParameters(
            means_world=self.means,
            quaternions=self.quaternions,
            scales=self.scales,
            opacities=self.opacity,
            sh_coefficients=self.color_coefficients,
        )


class _StaticGaussianModel(nn.Module):
    def __init__(self, parameters: GaussianParameters) -> None:
        super().__init__()
        self.parameters_for_render = parameters

    def transformed_parameters(self) -> GaussianParameters:
        return self.parameters_for_render


def _config() -> RenderingConfig:
    return RenderingConfig(
        background=(0.0, 0.0, 0.0),
        sigma_extent=3.0,
        epsilon_covariance=0.3,
        epsilon_determinant=1e-8,
        alpha_max=0.99,
        alpha_min=1.0 / 255.0,
        transmittance_min=1e-4,
        tile_based=False,
    )


def _camera() -> Camera:
    return Camera(
        rotation_cw=torch.eye(3),
        translation_cw=torch.zeros(3),
        camera_center_world=torch.zeros(3),
        fx=10.0,
        fy=10.0,
        cx=2.0,
        cy=2.0,
        width=5,
        height=5,
    )


def test_gaussian_renderer_returns_compact_result_and_retains_screen_gradient() -> None:
    config = _config()
    camera = _camera()
    model = _OneGaussianModel()
    result = GaussianRenderer(config)(model, camera, retain_screen_grad=True)

    assert isinstance(result, RenderResult)
    assert model.call_count == 1
    assert result.image.shape == (3, 5, 5)
    assert result.final_transmittance.shape == (5, 5)
    assert torch.equal(result.visible_mask, torch.tensor([True]))
    assert result.projected.original_indices.shape == (1,)

    result.image.sum().backward()
    assert result.projected.means_screen.grad is not None
    assert model.means.grad is not None


def test_centered_anisotropic_gaussian_produces_elliptical_distribution() -> None:
    sh = torch.zeros((1, 16, 3))
    parameters = GaussianParameters(
        means_world=torch.tensor([[0.0, 0.0, 2.0]]),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.tensor([[0.25, 0.08, 0.10]]),
        opacities=torch.tensor([[0.8]]),
        sh_coefficients=sh,
    )

    result = GaussianRenderer(_config())(_StaticGaussianModel(parameters), _camera())
    intensity = result.image[0]

    assert intensity[2, 2] > intensity[2, 1] > intensity[0, 0]
    assert intensity[2, 1] > intensity[1, 2]
    torch.testing.assert_close(intensity[2, 1], intensity[2, 3])
    torch.testing.assert_close(intensity[1, 2], intensity[3, 2])


def test_two_gaussians_project_sort_and_composite_front_to_back() -> None:
    y00 = 0.28209479177387814
    colors = torch.tensor([[0.1, 0.2, 0.8], [0.8, 0.2, 0.1]])
    sh = torch.zeros((2, 16, 3))
    sh[:, 0, :] = (colors - 0.5) / y00
    parameters = GaussianParameters(
        # Back Gaussian is intentionally stored first.
        means_world=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        scales=torch.full((2, 3), 0.12),
        opacities=torch.tensor([[0.5], [0.5]]),
        sh_coefficients=sh,
    )

    result = GaussianRenderer(_config())(_StaticGaussianModel(parameters), _camera())

    assert result.projected.original_indices.tolist() == [1, 0]
    expected_center = 0.5 * colors[1] + (1.0 - 0.5) * 0.5 * colors[0]
    torch.testing.assert_close(result.image[:, 2, 2], expected_center)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("diff_gaussian_rasterization") is None,
    reason="official CUDA rasterizer is unavailable",
)
def test_cuda_renderer_propagates_gradients_and_preserves_adc_inputs() -> None:
    device = torch.device("cuda")
    model = GaussianModel(
        means_world=torch.tensor([[0.0, 0.0, 2.0]], device=device),
        raw_quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        raw_scales=torch.log(torch.full((1, 3), 0.1, device=device)),
        raw_opacities=torch.logit(torch.tensor([[0.8]], device=device)),
        sh_dc=torch.tensor([[[1.0, 0.0, 0.0]]], device=device),
        sh_rest=torch.zeros((1, 15, 3), device=device),
    )
    camera = Camera(
        rotation_cw=torch.eye(3, device=device),
        translation_cw=torch.zeros(3, device=device),
        camera_center_world=torch.zeros(3, device=device),
        fx=10.0,
        fy=10.0,
        cx=2.0,
        cy=2.0,
        width=5,
        height=5,
    )

    result = GaussianRenderer(_config(), backend="cuda")(
        model,
        camera,
        retain_screen_grad=True,
    )
    assert result.image.shape == (3, 5, 5)
    assert result.final_transmittance is None
    assert torch.equal(result.visible_mask, torch.tensor([True], device=device))
    assert result.projected.radii.dtype == torch.int64
    assert result.screen_space_points is not None

    result.image.sum().backward()
    assert result.screen_space_points.grad is not None
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.accumulate(result)
    assert statistics.position_gradient_denominator.item() == 1
    assert statistics.max_screen_radius.item() > 0
