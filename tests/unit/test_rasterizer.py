from __future__ import annotations

import math

import torch

from gaussian_splatting.renderer.projection import ProjectedGaussians
from gaussian_splatting.renderer.rasterizer import (
    clamped_projected_opacity,
    implementation_pixel_color_with_background,
    implementation_projected_opacity,
    opacity_contribution_threshold,
    rasterize_gaussians,
    stable_color_accumulation,
    stable_transmittance_accumulation,
    transmittance_termination,
)


def _projected(
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    means_screen: torch.Tensor | None = None,
) -> ProjectedGaussians:
    count = colors.shape[0]
    dtype = colors.dtype
    device = colors.device
    if means_screen is None:
        means_screen = torch.tensor([[1.0, 1.0]], dtype=dtype, device=device).repeat(count, 1)
    return ProjectedGaussians(
        original_indices=torch.arange(count, device=device),
        means_camera=torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device).repeat(count, 1),
        means_screen=means_screen,
        depths=torch.arange(1, count + 1, dtype=dtype, device=device),
        covariances_3d=torch.eye(3, dtype=dtype, device=device).repeat(count, 1, 1),
        covariances_2d=torch.eye(2, dtype=dtype, device=device).repeat(count, 1, 1),
        inverse_covariances_2d=torch.eye(2, dtype=dtype, device=device).repeat(count, 1, 1),
        colors=colors,
        opacities=opacities,
        radii=torch.full((count,), 1, dtype=torch.int64, device=device),
        rectangles=torch.tensor([[0, 2, 0, 2]], device=device).repeat(count, 1),
    )


def test_implementation_projected_opacity__eq_implementation_projected_opacity() -> None:
    pixels = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    actual = implementation_projected_opacity(
        torch.tensor([0.8], dtype=torch.float64),
        pixels,
        torch.zeros(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
    )
    expected = torch.tensor([0.8, 0.8 * math.exp(-0.5)], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_clamped_projected_opacity__eq_clamped_projected_opacity() -> None:
    alpha = torch.tensor([0.2, 0.999])
    torch.testing.assert_close(
        clamped_projected_opacity(alpha),
        torch.tensor([0.2, 0.99]),
    )


def test_opacity_contribution_threshold__eq_opacity_contribution_threshold() -> None:
    threshold = 1.0 / 255.0
    alpha = torch.tensor([threshold / 2.0, threshold, threshold * 2.0])
    torch.testing.assert_close(
        opacity_contribution_threshold(alpha),
        torch.tensor([0.0, threshold, threshold * 2.0]),
    )


def test_stable_accumulation_equations() -> None:
    transmittance = torch.tensor([0.5])
    alpha = torch.tensor([0.25])
    test_transmittance = stable_transmittance_accumulation(transmittance, alpha)
    torch.testing.assert_close(test_transmittance, torch.tensor([0.375]))
    assert not bool(transmittance_termination(test_transmittance).item())
    color = stable_color_accumulation(
        torch.zeros(1, 3),
        transmittance,
        alpha,
        torch.tensor([1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(color, torch.tensor([[0.125, 0.0, 0.0]]))
    with_background = implementation_pixel_color_with_background(
        color,
        test_transmittance,
        torch.tensor([0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(with_background, torch.tensor([[0.125, 0.0, 0.375]]))


def test_rasterize_gaussians_composites_front_to_back() -> None:
    projected = _projected(
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        opacities=torch.tensor([[0.5], [0.5]]),
    )
    image, transmittance = rasterize_gaussians(
        projected,
        height=3,
        width=3,
        background=torch.zeros(3),
    )
    torch.testing.assert_close(image[:, 1, 1], torch.tensor([0.5, 0.25, 0.0]))
    torch.testing.assert_close(transmittance[1, 1], torch.tensor(0.25))


def test_rasterize_gaussians_rejects_terminating_candidate() -> None:
    projected = _projected(
        colors=torch.tensor([[1.0, 0.0, 0.0]]),
        opacities=torch.tensor([[1.0]]),
    )
    image, transmittance = rasterize_gaussians(
        projected,
        height=3,
        width=3,
        background=torch.tensor([0.0, 0.0, 1.0]),
        alpha_max=0.99995,
        transmittance_min=1e-4,
    )
    torch.testing.assert_close(image[:, 1, 1], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(transmittance[1, 1], torch.tensor(1.0))


def test_rasterize_gaussians_preserves_autograd_path() -> None:
    means_screen = torch.tensor([[1.2, 1.1]], dtype=torch.float64, requires_grad=True)
    colors = torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float64, requires_grad=True)
    opacities = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    projected = _projected(colors, opacities, means_screen=means_screen)
    image, _ = rasterize_gaussians(
        projected,
        height=3,
        width=3,
        background=torch.zeros(3, dtype=torch.float64),
    )
    image.sum().backward()
    assert means_screen.grad is not None
    assert colors.grad is not None
    assert opacities.grad is not None
    assert bool(torch.isfinite(means_screen.grad).all())

