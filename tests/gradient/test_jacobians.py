from __future__ import annotations

import pytest
import torch

from gaussian_splatting.math import jacobians as analytic
from gaussian_splatting.math.covariance import (
    covariance_2d,
    covariance_3d,
    pack_symmetric_2d,
    pack_symmetric_3d,
    quaternion_rotation_matrix,
    unpack_symmetric_2d,
    unpack_symmetric_3d,
)
from gaussian_splatting.math.parameterization import raw_parameter_transformations
from gaussian_splatting.math.spherical_harmonics import (
    real_sh_degree_3,
    sh_color_implementation,
)
from gaussian_splatting.math.transform import (
    perspective_projection,
    projection_jacobian,
    view_direction,
    world_to_camera,
)
from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianParameters
from gaussian_splatting.renderer.projection import project_gaussians
from gaussian_splatting.renderer.rasterizer import (
    implementation_projected_opacity,
    rasterize_gaussians,
)


DTYPE = torch.float64
RTOL = 1e-4
ATOL = 1e-6


def _central_jacobian(function, value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    columns = []
    flat = value.reshape(-1)
    for index in range(flat.numel()):
        perturbation = torch.zeros_like(flat)
        perturbation[index] = epsilon
        positive = function((flat + perturbation).reshape_as(value)).reshape(-1)
        negative = function((flat - perturbation).reshape_as(value)).reshape(-1)
        columns.append((positive - negative) / (2.0 * epsilon))
    return torch.stack(columns, dim=-1)


def _covariance_from_quaternion(quaternion: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rotation = quaternion_rotation_matrix(quaternion[None])[0]
    factor = rotation * scales.unsqueeze(0)
    return factor @ factor.transpose(0, 1)


def test_coordinate_jacobians_match_autograd_and_finite_difference():
    rotation_cw = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    translation = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE)
    point_world = torch.tensor([0.4, 0.5, 2.0], dtype=DTYPE)
    camera_function = lambda point: rotation_cw @ point + translation
    torch.testing.assert_close(
        analytic.backward_world_to_camera_jacobian(rotation_cw),
        torch.autograd.functional.jacobian(camera_function, point_world),
        rtol=RTOL,
        atol=ATOL,
    )

    point_camera = camera_function(point_world)
    projection_function = lambda point: torch.stack(
        (300.0 * point[0] / point[2], 320.0 * point[1] / point[2])
    )
    expected = analytic.backward_projection_jacobian(point_camera, 300.0, 320.0)
    torch.testing.assert_close(
        expected,
        torch.autograd.functional.jacobian(projection_function, point_camera),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        expected,
        _central_jacobian(projection_function, point_camera),
        rtol=RTOL,
        atol=ATOL,
    )


def test_covariance_quaternion_jacobians_match_autograd_and_finite_difference():
    quaternion = torch.tensor([0.7, 0.2, -0.3, 0.6], dtype=DTYPE)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    scales = torch.tensor([0.8, 1.2, 1.7], dtype=DTYPE)
    function = lambda q: pack_symmetric_3d(_covariance_from_quaternion(q, scales))
    expected = analytic.covariance_quaternion_jacobian(quaternion, scales)
    torch.testing.assert_close(
        expected,
        torch.autograd.functional.jacobian(function, quaternion),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        expected,
        _central_jacobian(function, quaternion),
        rtol=RTOL,
        atol=ATOL,
    )
    derivative_functions = (
        analytic.rotation_qw_jacobian,
        analytic.rotation_qx_jacobian,
        analytic.rotation_qy_jacobian,
        analytic.rotation_qz_jacobian,
    )
    rotation_function = lambda q: quaternion_rotation_matrix(q[None])[0]
    full_rotation_jacobian = torch.autograd.functional.jacobian(rotation_function, quaternion)
    for component, derivative_function in enumerate(derivative_functions):
        torch.testing.assert_close(
            derivative_function(quaternion),
            full_rotation_jacobian[..., component],
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            analytic.covariance_quaternion_matrix_derivative(
                quaternion, scales, component
            ),
            torch.autograd.functional.jacobian(
                lambda q: _covariance_from_quaternion(q, scales), quaternion
            )[..., component],
            rtol=RTOL,
            atol=ATOL,
        )


def test_covariance_scale_jacobians_match_autograd():
    quaternion = torch.tensor([0.8, -0.1, 0.3, 0.4], dtype=DTYPE)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    scales = torch.tensor([0.9, 1.1, 1.5], dtype=DTYPE)
    function = lambda s: pack_symmetric_3d(_covariance_from_quaternion(quaternion, s))
    torch.testing.assert_close(
        analytic.covariance_scale_jacobian(quaternion, scales),
        torch.autograd.functional.jacobian(function, scales),
        rtol=RTOL,
        atol=ATOL,
    )
    for component in range(3):
        expected_squared = torch.zeros((3, 3), dtype=DTYPE)
        expected_squared[component, component] = 2.0 * scales[component]
        torch.testing.assert_close(
            analytic.scale_squared_matrix_derivative(scales, component),
            expected_squared,
        )
        matrix_jacobian = torch.autograd.functional.jacobian(
            lambda s: _covariance_from_quaternion(quaternion, s), scales
        )
        torch.testing.assert_close(
            analytic.covariance_scale_matrix_derivative(
                quaternion, scales, component
            ),
            matrix_jacobian[..., component],
            rtol=RTOL,
            atol=ATOL,
        )


def test_projected_covariance_jacobians_match_autograd():
    point = torch.tensor([0.2, -0.3, 2.4], dtype=DTYPE)
    rotation_cw = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    covariance_vector = torch.tensor([1.2, 0.1, -0.2, 0.9, 0.05, 1.4], dtype=DTYPE)

    def project_covariance(covariance, camera_point):
        jacobian = projection_jacobian(camera_point[None], 300.0, 320.0)[0]
        camera_covariance = rotation_cw @ covariance @ rotation_cw.transpose(0, 1)
        return jacobian @ camera_covariance @ jacobian.transpose(0, 1)

    covariance_function = lambda vector: pack_symmetric_2d(
        project_covariance(unpack_symmetric_3d(vector), point)
    )
    expected_covariance = analytic.projected_covariance_3d_jacobian(
        point, rotation_cw, 300.0, 320.0
    )
    torch.testing.assert_close(
        expected_covariance,
        torch.autograd.functional.jacobian(covariance_function, covariance_vector),
        rtol=RTOL,
        atol=ATOL,
    )
    for component in range(6):
        torch.testing.assert_close(
            pack_symmetric_2d(
                analytic.projected_covariance_3d_component_derivative(
                    point, rotation_cw, 300.0, 320.0, component
                )
            ),
            expected_covariance[:, component],
            rtol=RTOL,
            atol=ATOL,
        )

    covariance = unpack_symmetric_3d(covariance_vector)
    position_function = lambda camera_point: pack_symmetric_2d(
        project_covariance(covariance, camera_point)
    )
    expected_position = analytic.projected_covariance_position_jacobian(
        covariance, point, rotation_cw, 300.0, 320.0
    )
    torch.testing.assert_close(
        expected_position,
        torch.autograd.functional.jacobian(position_function, point),
        rtol=RTOL,
        atol=ATOL,
    )
    projection_function = lambda camera_point: projection_jacobian(
        camera_point[None], 300.0, 320.0
    )[0]
    projection_derivatives = torch.autograd.functional.jacobian(projection_function, point)
    for component, function in enumerate(
        (
            analytic.projection_jacobian_x_derivative,
            analytic.projection_jacobian_y_derivative,
            analytic.projection_jacobian_z_derivative,
        )
    ):
        torch.testing.assert_close(
            function(point, 300.0, 320.0),
            projection_derivatives[..., component],
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            pack_symmetric_2d(
                analytic.projected_covariance_position_matrix_derivative(
                    covariance,
                    point,
                    rotation_cw,
                    300.0,
                    320.0,
                    component,
                )
            ),
            expected_position[:, component],
            rtol=RTOL,
            atol=ATOL,
        )


def test_sh_jacobians_match_autograd_and_coefficient_layout():
    direction = torch.tensor([0.2, -0.5, 0.7], dtype=DTYPE)
    coefficients = torch.arange(48, dtype=DTYPE).reshape(16, 3) / 20.0
    color_function = lambda value: (
        real_sh_degree_3(value).unsqueeze(-1) * coefficients
    ).sum(dim=0)
    torch.testing.assert_close(
        analytic.color_view_direction_jacobian(direction, coefficients),
        torch.autograd.functional.jacobian(color_function, direction),
        rtol=RTOL,
        atol=ATOL,
    )
    coefficient_function = lambda value: (
        real_sh_degree_3(direction).unsqueeze(-1) * value.reshape(16, 3)
    ).sum(dim=0)
    flattened = coefficients.reshape(-1)
    torch.testing.assert_close(
        analytic.color_sh_coefficient_jacobian(direction, 3),
        torch.autograd.functional.jacobian(coefficient_function, flattened),
        rtol=RTOL,
        atol=ATOL,
    )
    for basis_index in (0, 5, 15):
        expected_block = analytic.color_sh_coefficient_jacobian(direction, 3)[
            :, 3 * basis_index : 3 * (basis_index + 1)
        ]
        torch.testing.assert_close(
            analytic.color_sh_component_jacobian(direction, basis_index), expected_block
        )

    camera_center = torch.tensor([-0.1, 0.3, 0.2], dtype=DTYPE)
    mean_world = torch.tensor([0.4, -0.2, 1.2], dtype=DTYPE)

    def world_color(mean):
        offset = mean - camera_center
        unit_direction = offset / torch.linalg.vector_norm(offset)
        return color_function(unit_direction)

    torch.testing.assert_close(
        analytic.color_world_position_jacobian(
            mean_world, camera_center, coefficients
        ),
        torch.autograd.functional.jacobian(world_color, mean_world),
        rtol=RTOL,
        atol=ATOL,
    )
    direction_function = lambda mean: (mean - camera_center) / torch.linalg.vector_norm(
        mean - camera_center
    )
    torch.testing.assert_close(
        analytic.view_direction_position_jacobian(mean_world, camera_center),
        torch.autograd.functional.jacobian(direction_function, mean_world),
        rtol=RTOL,
        atol=ATOL,
    )


def test_opacity_and_pixel_jacobians_match_autograd():
    displacement = torch.tensor([0.4, -0.2], dtype=DTYPE)
    covariance_vector = torch.tensor([1.4, 0.2, 0.9], dtype=DTYPE)
    inverse_covariance = torch.linalg.inv(unpack_symmetric_2d(covariance_vector))
    opacity = torch.tensor(0.7, dtype=DTYPE)
    transmittance = torch.tensor(0.6, dtype=DTYPE)
    color = torch.tensor([0.3, 0.8, 0.1], dtype=DTYPE)
    next_color = torch.tensor([0.2, 0.1, 0.4], dtype=DTYPE)

    weight_function = lambda position: torch.exp(
        -0.5 * position @ inverse_covariance @ position
    )
    gaussian_weight = weight_function(displacement)
    torch.testing.assert_close(
        analytic.screen_opacity_parameter_derivative(
            displacement, inverse_covariance
        ),
        gaussian_weight,
    )
    torch.testing.assert_close(
        analytic.gaussian_weight_position_jacobian(
            displacement, inverse_covariance
        ),
        torch.autograd.functional.jacobian(weight_function, displacement),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        analytic.screen_opacity_position_jacobian(
            opacity, displacement, inverse_covariance
        ),
        opacity * torch.autograd.functional.jacobian(weight_function, displacement),
        rtol=RTOL,
        atol=ATOL,
    )

    inverse_function = lambda vector: pack_symmetric_2d(
        torch.linalg.inv(unpack_symmetric_2d(vector))
    )
    torch.testing.assert_close(
        analytic.inverse_covariance_jacobian(covariance_vector),
        torch.autograd.functional.jacobian(inverse_function, covariance_vector),
        rtol=RTOL,
        atol=ATOL,
    )
    inverse_vector = pack_symmetric_2d(inverse_covariance)
    inverse_weight_function = lambda vector: torch.exp(
        -0.5 * displacement @ unpack_symmetric_2d(vector) @ displacement
    )
    torch.testing.assert_close(
        analytic.gaussian_weight_inverse_covariance_jacobian(
            displacement, inverse_covariance
        ),
        torch.autograd.functional.jacobian(
            inverse_weight_function, inverse_vector
        ),
        rtol=RTOL,
        atol=ATOL,
    )

    opacity_function = lambda value: value * gaussian_weight
    torch.testing.assert_close(
        analytic.screen_opacity_parameter_derivative(
            displacement, inverse_covariance
        ),
        torch.autograd.functional.jacobian(opacity_function, opacity),
    )
    screen_covariance_function = lambda vector: opacity * torch.exp(
        -0.5
        * displacement
        @ torch.linalg.inv(unpack_symmetric_2d(vector))
        @ displacement
    )
    torch.testing.assert_close(
        analytic.screen_opacity_covariance_jacobian(
            opacity, displacement, covariance_vector
        ),
        torch.autograd.functional.jacobian(
            screen_covariance_function, covariance_vector
        ),
        rtol=RTOL,
        atol=ATOL,
    )

    recursive = lambda screen_opacity: screen_opacity * color + (
        1.0 - screen_opacity
    ) * next_color
    screen_opacity = opacity * gaussian_weight
    torch.testing.assert_close(
        analytic.recursive_color_opacity_jacobian(color, next_color),
        torch.autograd.functional.jacobian(recursive, screen_opacity),
    )
    torch.testing.assert_close(
        analytic.pixel_color_recursive_jacobian(transmittance),
        transmittance * torch.eye(3, dtype=DTYPE),
    )
    torch.testing.assert_close(
        analytic.pixel_color_screen_opacity_jacobian(
            transmittance, color, next_color
        ),
        transmittance * torch.autograd.functional.jacobian(recursive, screen_opacity),
    )
    torch.testing.assert_close(
        analytic.pixel_color_color_jacobian(transmittance, screen_opacity),
        transmittance * screen_opacity * torch.eye(3, dtype=DTYPE),
    )

    opacity_pixel_function = lambda value: transmittance * recursive(
        value * gaussian_weight
    )
    torch.testing.assert_close(
        analytic.pixel_color_opacity_parameter_jacobian(
            transmittance, color, next_color, gaussian_weight
        ),
        torch.autograd.functional.jacobian(opacity_pixel_function, opacity),
        rtol=RTOL,
        atol=ATOL,
    )
    position_pixel_function = lambda position: transmittance * recursive(
        opacity * weight_function(position)
    )
    torch.testing.assert_close(
        analytic.pixel_color_position_jacobian(
            transmittance,
            color,
            next_color,
            opacity,
            displacement,
            inverse_covariance,
        ),
        torch.autograd.functional.jacobian(position_pixel_function, displacement),
        rtol=RTOL,
        atol=ATOL,
    )
    covariance_pixel_function = lambda vector: transmittance * recursive(
        screen_covariance_function(vector)
    )
    torch.testing.assert_close(
        analytic.pixel_color_covariance_jacobian(
            transmittance,
            color,
            next_color,
            opacity,
            displacement,
            covariance_vector,
        ),
        torch.autograd.functional.jacobian(
            covariance_pixel_function, covariance_vector
        ),
        rtol=RTOL,
        atol=ATOL,
    )


def test_jacobian_api_rejects_non_float64_batch_and_bad_components():
    with pytest.raises(TypeError, match="float64"):
        analytic.backward_world_to_camera_jacobian(torch.eye(3, dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        analytic.backward_projection_jacobian(
            torch.ones((1, 3), dtype=DTYPE), 1.0, 1.0
        )
    with pytest.raises(ValueError, match="component"):
        analytic.covariance_quaternion_matrix_derivative(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
            torch.ones(3, dtype=DTYPE),
            4,
        )


def test_gradcheck_position_projection_and_raw_covariance_paths() -> None:
    rotation_cw = torch.eye(3, dtype=DTYPE)
    translation_cw = torch.zeros(3, dtype=DTYPE)
    point = torch.tensor([[0.2, -0.1, 2.0]], dtype=DTYPE, requires_grad=True)

    def project(value: torch.Tensor) -> torch.Tensor:
        camera_point = world_to_camera(value, rotation_cw, translation_cw)
        return perspective_projection(camera_point, 10.0, 11.0, 2.0, 2.0)

    assert torch.autograd.gradcheck(
        project, (point,), eps=1e-6, atol=1e-6, rtol=1e-4
    )

    raw_quaternion = torch.tensor(
        [[1.0, 0.1, -0.05, 0.02]], dtype=DTYPE, requires_grad=True
    )
    raw_scale = torch.log(
        torch.tensor([[0.2, 0.3, 0.4]], dtype=DTYPE)
    ).requires_grad_()
    raw_opacity = torch.zeros((1, 1), dtype=DTYPE)

    def raw_covariance(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        quaternion, scales, _ = raw_parameter_transformations(
            q, scale, raw_opacity
        )
        rotation = quaternion_rotation_matrix(quaternion)
        return covariance_3d(rotation, scales)

    assert torch.autograd.gradcheck(
        raw_covariance,
        (raw_quaternion, raw_scale),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_gradcheck_projected_covariance_and_sh_position_paths() -> None:
    point_camera = torch.tensor([[0.2, -0.1, 2.0]], dtype=DTYPE)
    jacobian = projection_jacobian(point_camera, 10.0, 11.0)
    rotation_cw = torch.eye(3, dtype=DTYPE)
    covariance_vector = torch.tensor(
        [0.4, 0.02, 0.01, 0.3, 0.01, 0.2],
        dtype=DTYPE,
        requires_grad=True,
    )

    def project_covariance(value: torch.Tensor) -> torch.Tensor:
        covariance = unpack_symmetric_3d(value).unsqueeze(0)
        return covariance_2d(jacobian, rotation_cw, covariance)

    assert torch.autograd.gradcheck(
        project_covariance,
        (covariance_vector,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )

    mean_world = torch.tensor(
        [[0.2, -0.1, 2.0]], dtype=DTYPE, requires_grad=True
    )
    camera_center = torch.zeros(3, dtype=DTYPE)
    coefficients = torch.full((1, 16, 3), 0.01, dtype=DTYPE)

    def position_color(value: torch.Tensor) -> torch.Tensor:
        direction = view_direction(value, camera_center)
        return sh_color_implementation(direction, coefficients)

    assert torch.autograd.gradcheck(
        position_color,
        (mean_world,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_gradcheck_screen_opacity_center_and_covariance() -> None:
    opacity = torch.tensor([0.4], dtype=DTYPE, requires_grad=True)
    mean_screen = torch.tensor([1.1, 0.9], dtype=DTYPE, requires_grad=True)
    covariance_vector = torch.tensor(
        [1.2, 0.1, 0.8], dtype=DTYPE, requires_grad=True
    )
    pixel = torch.tensor([1.0, 1.0], dtype=DTYPE)

    def screen_opacity(
        alpha: torch.Tensor,
        mean: torch.Tensor,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        inverse = torch.linalg.inv(unpack_symmetric_2d(covariance))
        return implementation_projected_opacity(alpha, pixel, mean, inverse)

    assert torch.autograd.gradcheck(
        screen_opacity,
        (opacity, mean_screen, covariance_vector),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_gradcheck_full_raw_parameter_rendering_path() -> None:
    rendering = RenderingConfig(
        background=(0.0, 0.0, 0.0),
        sigma_extent=3.0,
        epsilon_covariance=0.3,
        epsilon_determinant=1e-8,
        alpha_max=0.99,
        alpha_min=1.0 / 255.0,
        transmittance_min=1e-4,
        tile_based=False,
    )
    camera = Camera(
        rotation_cw=torch.eye(3, dtype=DTYPE),
        translation_cw=torch.zeros(3, dtype=DTYPE),
        camera_center_world=torch.zeros(3, dtype=DTYPE),
        fx=8.0,
        fy=8.0,
        cx=2.0,
        cy=2.0,
        width=5,
        height=5,
    )
    mean = torch.tensor([[0.1, 0.0, 2.0]], dtype=DTYPE, requires_grad=True)
    raw_quaternion = torch.tensor(
        [[1.0, 0.05, -0.03, 0.02]], dtype=DTYPE, requires_grad=True
    )
    raw_scale = torch.log(
        torch.tensor([[0.18, 0.22, 0.20]], dtype=DTYPE)
    ).requires_grad_()
    raw_opacity = torch.logit(
        torch.tensor([[0.35]], dtype=DTYPE)
    ).requires_grad_()
    coefficients = torch.full(
        (1, 16, 3), 0.01, dtype=DTYPE, requires_grad=True
    )

    def render_from_raw(
        means: torch.Tensor,
        quaternions_raw: torch.Tensor,
        scales_raw: torch.Tensor,
        opacities_raw: torch.Tensor,
        sh_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        quaternions, scales, opacities = raw_parameter_transformations(
            quaternions_raw, scales_raw, opacities_raw
        )
        parameters = GaussianParameters(
            means_world=means,
            quaternions=quaternions,
            scales=scales,
            opacities=opacities,
            sh_coefficients=sh_coefficients,
        )
        projected, _ = project_gaussians(parameters, camera, rendering)
        background = means.new_zeros(3)
        image, _ = rasterize_gaussians(
            projected,
            camera.height,
            camera.width,
            background,
            alpha_max=rendering.alpha_max,
            alpha_min=rendering.alpha_min,
            transmittance_min=rendering.transmittance_min,
        )
        return image

    assert torch.autograd.gradcheck(
        render_from_raw,
        (mean, raw_quaternion, raw_scale, raw_opacity, coefficients),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )
