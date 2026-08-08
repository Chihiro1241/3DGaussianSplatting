from __future__ import annotations

import torch

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianParameters
from gaussian_splatting.renderer.projection import (
    gaussian_rendering_radius,
    gaussian_rendering_rectangle,
    implementation_depth_order,
    maximum_2d_covariance_eigenvalue,
    pixel_coordinate_convention,
    project_gaussians,
    visible_depth_condition,
)


def _rendering_config() -> RenderingConfig:
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


def test_visible_depth_condition__eq_visible_depth_condition() -> None:
    points = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.equal(
        visible_depth_condition(points),
        torch.tensor([False, False, True]),
    )


def test_maximum_2d_covariance_eigenvalue__eq_maximum_2d_covariance_eigenvalue() -> None:
    covariance = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], dtype=torch.float64)
    actual = maximum_2d_covariance_eigenvalue(covariance)
    torch.testing.assert_close(actual, torch.tensor([3.0], dtype=torch.float64))


def test_gaussian_rendering_radius__eq_gaussian_rendering_radius() -> None:
    eigenvalues = torch.tensor([0.0, 1.0, 1.01])
    assert torch.equal(
        gaussian_rendering_radius(eigenvalues),
        torch.tensor([0, 3, 4]),
    )


def test_gaussian_rendering_rectangle__eq_gaussian_rendering_rectangle() -> None:
    means = torch.tensor([[2.25, 1.75], [-5.0, 1.0], [9.0, 1.0]])
    radii = torch.tensor([1, 1, 1])
    rectangles = gaussian_rendering_rectangle(means, radii, height=4, width=5)
    assert torch.equal(
        rectangles,
        torch.tensor(
            [
                [1, 4, 0, 3],
                [0, -4, 0, 2],
                [8, 4, 0, 2],
            ]
        ),
    )


def test_pixel_coordinate_convention__eq_pixel_coordinate_convention() -> None:
    grid = pixel_coordinate_convention(2, 3, dtype=torch.float64)
    assert grid.shape == (2, 3, 2)
    assert grid.dtype == torch.float64
    torch.testing.assert_close(
        grid,
        torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            ],
            dtype=torch.float64,
        ),
    )


def test_implementation_depth_order__eq_implementation_depth_order() -> None:
    depths = torch.tensor([2.0, 1.0, 2.0, 2.0])
    original_indices = torch.tensor([7, 4, 2, 5])
    order = implementation_depth_order(depths, original_indices)
    assert torch.equal(order, torch.tensor([1, 2, 3, 0]))


def test_project_gaussians_filters_and_orders_visible_rows() -> None:
    means = torch.tensor(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.0, -1.0], [100.0, 0.0, 2.0]],
        requires_grad=True,
    )
    parameters = GaussianParameters(
        means_world=means,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
        scales=torch.full((4, 3), 0.01),
        opacities=torch.full((4, 1), 0.5),
        sh_coefficients=torch.zeros(4, 16, 3),
    )
    projected, visible_mask = project_gaussians(parameters, _camera(), _rendering_config())

    assert torch.equal(visible_mask, torch.tensor([True, True, False, False]))
    assert torch.equal(projected.original_indices, torch.tensor([0, 1]))
    assert projected.means_screen.shape == (2, 2)
    assert projected.means_screen.requires_grad
    assert torch.equal(
        torch.sort(projected.original_indices).values,
        visible_mask.nonzero().squeeze(1),
    )

