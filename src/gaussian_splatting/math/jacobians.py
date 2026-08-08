"""Analytic single-Gaussian Jacobians used only for gradient verification.

Every public function in this module requires CPU ``torch.float64`` tensor
inputs and has no batch dimension. Jacobian rows index output degrees of
freedom and columns index input degrees of freedom.
"""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor

from ._validation import (
    jacobian_scalar,
    validate_active_degree,
    validate_component,
    validate_jacobian_tensor,
    validate_symmetric,
)
from .covariance import (
    pack_symmetric_2d,
    pack_symmetric_3d,
    quaternion_rotation_matrix,
    unpack_symmetric_3d,
)
from .spherical_harmonics import real_sh_degree_3
from .transform import projection_jacobian


_VALID_BASIS_COUNTS = (1, 4, 9, 16)


def backward_world_to_camera_jacobian(rotation_cw: Tensor) -> Tensor:
    """Return ``d point_camera / d point_world`` as a ``(3, 3)`` matrix.

    Args:
        rotation_cw: CPU float64 world-to-camera rotation, shape ``(3, 3)``.

    Returns:
        CPU float64 Jacobian with shape ``(3, 3)``; Cartesian component order.

    TeX: eq:backward_world_to_camera_jacobian
    """

    validate_jacobian_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    return rotation_cw.clone()


def backward_projection_jacobian(
    point_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Return ``d image_position / d point_camera`` with shape ``(2, 3)``.

    Args:
        point_camera: CPU float64 ``(x, y, z)`` vector, shape ``(3,)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar.

    Returns:
        CPU float64 ``(2, 3)`` Jacobian; output order ``(x, y)`` and input
        order ``(x, y, z)``.

    TeX: eq:backward_projection_jacobian
    """

    validate_jacobian_tensor(point_camera, "point_camera", shape=(3,))
    fx_t = _positive_jacobian_scalar(fx, "fx")
    fy_t = _positive_jacobian_scalar(fy, "fy")
    return projection_jacobian(point_camera.unsqueeze(0), fx_t, fy_t)[0]


def covariance_quaternion_matrix_derivative(
    quaternion: Tensor,
    scales: Tensor,
    component: int,
) -> Tensor:
    """Return ``d Sigma / d q_component`` as a ``(3, 3)`` matrix.

    Args:
        quaternion: CPU float64 quaternion ``(qw,qx,qy,qz)``, shape ``(4,)``.
        scales: CPU float64 axis scales, shape ``(3,)``.
        component: ``0=qw, 1=qx, 2=qy, 3=qz``.

    Returns:
        CPU float64 symmetric derivative matrix, shape ``(3, 3)``.

    TeX: eq:covariance_quaternion_matrix_derivative
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    validate_jacobian_tensor(scales, "scales", shape=(3,))
    index = validate_component(component, 4)
    rotation = _rotation_matrix(quaternion)
    rotation_derivative = _rotation_derivative(quaternion, index)
    scale_squared = torch.diag(scales.square())
    return (
        rotation_derivative @ scale_squared @ rotation.transpose(0, 1)
        + rotation @ scale_squared @ rotation_derivative.transpose(0, 1)
    )


def rotation_qw_jacobian(quaternion: Tensor) -> Tensor:
    """Return ``d R / d qw`` with shape ``(3, 3)``.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.

    Returns:
        CPU float64 matrix with shape ``(3, 3)``.

    TeX: eq:rotation_qw_jacobian
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    return _rotation_derivative(quaternion, 0)


def rotation_qx_jacobian(quaternion: Tensor) -> Tensor:
    """Return ``d R / d qx`` with shape ``(3, 3)``.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.

    Returns:
        CPU float64 matrix with shape ``(3, 3)``.

    TeX: eq:rotation_qx_jacobian
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    return _rotation_derivative(quaternion, 1)


def rotation_qy_jacobian(quaternion: Tensor) -> Tensor:
    """Return ``d R / d qy`` with shape ``(3, 3)``.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.

    Returns:
        CPU float64 matrix with shape ``(3, 3)``.

    TeX: eq:rotation_qy_jacobian
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    return _rotation_derivative(quaternion, 2)


def rotation_qz_jacobian(quaternion: Tensor) -> Tensor:
    """Return ``d R / d qz`` with shape ``(3, 3)``.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.

    Returns:
        CPU float64 matrix with shape ``(3, 3)``.

    TeX: eq:rotation_qz_jacobian
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    return _rotation_derivative(quaternion, 3)


def covariance_quaternion_jacobian(quaternion: Tensor, scales: Tensor) -> Tensor:
    """Return ``d packed(Sigma) / d quaternion`` with shape ``(6, 4)``.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.
        scales: CPU float64 axis scales, shape ``(3,)``.

    Returns:
        CPU float64 Jacobian ``(6, 4)``. Covariance row order is
        ``(00,01,02,11,12,22)`` and quaternion column order is
        ``(qw,qx,qy,qz)``.

    TeX: eq:covariance_quaternion_jacobian, eq:rotation_gradient_chain
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    validate_jacobian_tensor(scales, "scales", shape=(3,))
    columns = [
        pack_symmetric_3d(
            covariance_quaternion_matrix_derivative(quaternion, scales, component)
        )
        for component in range(4)
    ]
    return torch.stack(columns, dim=-1)


def scale_squared_matrix_derivative(scales: Tensor, component: int) -> Tensor:
    """Return ``d diag(scales**2) / d scale_component``, shape ``(3, 3)``.

    Args:
        scales: CPU float64 axis scales, shape ``(3,)``.
        component: Scale component ``0=x, 1=y, 2=z``.

    Returns:
        CPU float64 derivative matrix with shape ``(3, 3)``.

    TeX: eq:scale_squared_matrix_derivative
    """

    validate_jacobian_tensor(scales, "scales", shape=(3,))
    index = validate_component(component, 3)
    diagonal = torch.zeros_like(scales)
    diagonal[index] = 2.0 * scales[index]
    return torch.diag(diagonal)


def covariance_scale_matrix_derivative(
    quaternion: Tensor,
    scales: Tensor,
    component: int,
) -> Tensor:
    """Return ``d Sigma / d scale_component`` as a ``(3, 3)`` matrix.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.
        scales: CPU float64 axis scales, shape ``(3,)``.
        component: Scale component ``0=x, 1=y, 2=z``.

    Returns:
        CPU float64 symmetric derivative matrix, shape ``(3, 3)``.

    TeX: eq:covariance_scale_matrix_derivative
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    validate_jacobian_tensor(scales, "scales", shape=(3,))
    index = validate_component(component, 3)
    rotation = _rotation_matrix(quaternion)
    scale_derivative = scale_squared_matrix_derivative(scales, index)
    return rotation @ scale_derivative @ rotation.transpose(0, 1)


def covariance_scale_jacobian(quaternion: Tensor, scales: Tensor) -> Tensor:
    """Return ``d packed(Sigma) / d scales`` with shape ``(6, 3)``.

    Args:
        quaternion: CPU float64 ``(qw,qx,qy,qz)`` vector, shape ``(4,)``.
        scales: CPU float64 axis scales, shape ``(3,)``.

    Returns:
        CPU float64 Jacobian ``(6, 3)``. Covariance row order is
        ``(00,01,02,11,12,22)``; scale columns are ``(x,y,z)``.

    TeX: eq:covariance_scale_jacobian, eq:scale_gradient_chain
    """

    validate_jacobian_tensor(quaternion, "quaternion", shape=(4,))
    validate_jacobian_tensor(scales, "scales", shape=(3,))
    columns = [
        pack_symmetric_3d(covariance_scale_matrix_derivative(quaternion, scales, component))
        for component in range(3)
    ]
    return torch.stack(columns, dim=-1)


def projected_covariance_3d_component_derivative(
    point_camera: Tensor,
    rotation_cw: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
    component: int,
) -> Tensor:
    """Return the projected derivative for one packed 3D covariance component.

    Args:
        point_camera: CPU float64 camera point ``(x,y,z)``, shape ``(3,)``.
        rotation_cw: CPU float64 world-to-camera rotation, shape ``(3,3)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar.
        component: Packed 3D covariance component in
            ``(00,01,02,11,12,22)`` order.

    Returns:
        CPU float64 matrix with shape ``(2, 2)``.

    TeX: eq:projected_covariance_3d_component_derivative
    """

    validate_jacobian_tensor(point_camera, "point_camera", shape=(3,))
    validate_jacobian_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    index = validate_component(component, 6)
    jacobian = backward_projection_jacobian(point_camera, fx, fy)
    packed_basis = point_camera.new_zeros(6)
    packed_basis[index] = 1.0
    covariance_basis = unpack_symmetric_3d(packed_basis)
    return (
        jacobian
        @ rotation_cw
        @ covariance_basis
        @ rotation_cw.transpose(0, 1)
        @ jacobian.transpose(0, 1)
    )


def projected_covariance_3d_jacobian(
    point_camera: Tensor,
    rotation_cw: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Return ``d packed(Sigma_2d) / d packed(Sigma_3d)``, shape ``(3,6)``.

    Args:
        point_camera: CPU float64 camera point, shape ``(3,)``.
        rotation_cw: CPU float64 world-to-camera rotation, shape ``(3,3)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar.

    Returns:
        CPU float64 Jacobian ``(3,6)``. Rows use ``(00,01,11)`` and columns
        use ``(00,01,02,11,12,22)``.

    TeX: eq:projected_covariance_3d_jacobian
    """

    validate_jacobian_tensor(point_camera, "point_camera", shape=(3,))
    validate_jacobian_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    columns = [
        pack_symmetric_2d(
            projected_covariance_3d_component_derivative(
                point_camera, rotation_cw, fx, fy, component
            )
        )
        for component in range(6)
    ]
    return torch.stack(columns, dim=-1)


def projection_jacobian_x_derivative(
    point_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Return ``d J / d x`` with shape ``(2, 3)``.

    Args:
        point_camera: CPU float64 camera point ``(x,y,z)``, shape ``(3,)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar (validated for interface consistency).

    Returns:
        CPU float64 matrix with shape ``(2, 3)``.

    TeX: eq:projection_jacobian_x_derivative
    """

    point, fx_t, _ = _validated_projection_inputs(point_camera, fx, fy)
    z = point[2]
    zero = point.new_zeros(())
    return torch.stack(
        (
            torch.stack((zero, zero, -fx_t / z.square())),
            torch.stack((zero, zero, zero)),
        )
    )


def projection_jacobian_y_derivative(
    point_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Return ``d J / d y`` with shape ``(2, 3)``.

    Args:
        point_camera: CPU float64 camera point ``(x,y,z)``, shape ``(3,)``.
        fx: Positive focal length scalar (validated for interface consistency).
        fy: Positive focal length scalar.

    Returns:
        CPU float64 matrix with shape ``(2, 3)``.

    TeX: eq:projection_jacobian_y_derivative
    """

    point, _, fy_t = _validated_projection_inputs(point_camera, fx, fy)
    z = point[2]
    zero = point.new_zeros(())
    return torch.stack(
        (
            torch.stack((zero, zero, zero)),
            torch.stack((zero, zero, -fy_t / z.square())),
        )
    )


def projection_jacobian_z_derivative(
    point_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Return ``d J / d z`` with shape ``(2, 3)``.

    Args:
        point_camera: CPU float64 camera point ``(x,y,z)``, shape ``(3,)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar.

    Returns:
        CPU float64 matrix with shape ``(2, 3)``.

    TeX: eq:projection_jacobian_z_derivative
    """

    point, fx_t, fy_t = _validated_projection_inputs(point_camera, fx, fy)
    x, y, z = point.unbind()
    zero = point.new_zeros(())
    return torch.stack(
        (
            torch.stack((-fx_t / z.square(), zero, 2.0 * fx_t * x / z.pow(3))),
            torch.stack((zero, -fy_t / z.square(), 2.0 * fy_t * y / z.pow(3))),
        )
    )


def projected_covariance_position_matrix_derivative(
    covariance_3d: Tensor,
    point_camera: Tensor,
    rotation_cw: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
    component: int,
) -> Tensor:
    """Return ``d Sigma_2d / d point_component`` with shape ``(2, 2)``.

    Args:
        covariance_3d: CPU float64 symmetric covariance, shape ``(3,3)``.
        point_camera: CPU float64 camera point, shape ``(3,)``.
        rotation_cw: CPU float64 world-to-camera rotation, shape ``(3,3)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar.
        component: Point component ``0=x, 1=y, 2=z``.

    Returns:
        CPU float64 symmetric matrix with shape ``(2,2)``.

    TeX: eq:projected_covariance_position_matrix_derivative
    """

    validate_jacobian_tensor(covariance_3d, "covariance_3d", shape=(3, 3))
    validate_symmetric(covariance_3d, "covariance_3d")
    validate_jacobian_tensor(point_camera, "point_camera", shape=(3,))
    validate_jacobian_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    index = validate_component(component, 3)
    jacobian = backward_projection_jacobian(point_camera, fx, fy)
    derivatives = (
        projection_jacobian_x_derivative,
        projection_jacobian_y_derivative,
        projection_jacobian_z_derivative,
    )
    jacobian_derivative = derivatives[index](point_camera, fx, fy)
    covariance_camera = rotation_cw @ covariance_3d @ rotation_cw.transpose(0, 1)
    return (
        jacobian_derivative @ covariance_camera @ jacobian.transpose(0, 1)
        + jacobian @ covariance_camera @ jacobian_derivative.transpose(0, 1)
    )


def projected_covariance_position_jacobian(
    covariance_3d: Tensor,
    point_camera: Tensor,
    rotation_cw: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Return ``d packed(Sigma_2d) / d point_camera``, shape ``(3,3)``.

    Args:
        covariance_3d: CPU float64 symmetric covariance, shape ``(3,3)``.
        point_camera: CPU float64 camera point, shape ``(3,)``.
        rotation_cw: CPU float64 world-to-camera rotation, shape ``(3,3)``.
        fx: Positive focal length scalar.
        fy: Positive focal length scalar.

    Returns:
        CPU float64 Jacobian ``(3,3)``. Rows use packed order
        ``(00,01,11)``; columns use point order ``(x,y,z)``.

    TeX: eq:projected_covariance_position_jacobian
    """

    validate_jacobian_tensor(covariance_3d, "covariance_3d", shape=(3, 3))
    validate_jacobian_tensor(point_camera, "point_camera", shape=(3,))
    validate_jacobian_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    columns = [
        pack_symmetric_2d(
            projected_covariance_position_matrix_derivative(
                covariance_3d,
                point_camera,
                rotation_cw,
                fx,
                fy,
                component,
            )
        )
        for component in range(3)
    ]
    return torch.stack(columns, dim=-1)


def color_sh_component_jacobian(direction: Tensor, basis_index: int) -> Tensor:
    """Return ``d color / d one RGB SH coefficient``, shape ``(3,3)``.

    Args:
        direction: CPU float64 view direction, shape ``(3,)``.
        basis_index: Basis-major index in ``[0,15]``.

    Returns:
        CPU float64 Jacobian ``Y_basis(direction) * I3``, shape ``(3,3)``;
        RGB component order is used on both axes.

    TeX: eq:color_sh_component_jacobian
    """

    validate_jacobian_tensor(direction, "direction", shape=(3,))
    index = validate_component(basis_index, 16, "basis_index")
    basis_value = real_sh_degree_3(direction)[index]
    return basis_value * torch.eye(3, dtype=direction.dtype, device=direction.device)


def color_sh_coefficient_jacobian(direction: Tensor, active_degree: int) -> Tensor:
    """Return ``d color / d flattened_SH``, shape ``(3, 3B)``.

    Args:
        direction: CPU float64 view direction, shape ``(3,)``.
        active_degree: Highest active degree in ``[0,3]``; ``B=(degree+1)^2``.

    Returns:
        CPU float64 Jacobian ``(3,3B)``. Columns flatten coefficients in
        ``(basis, rgb)`` order.

    TeX: eq:color_sh_coefficient_jacobian, eq:sh_gradient_chain
    """

    validate_jacobian_tensor(direction, "direction", shape=(3,))
    degree = validate_active_degree(active_degree)
    basis_count = (degree + 1) ** 2
    basis = real_sh_degree_3(direction)[:basis_count]
    identity = torch.eye(3, dtype=direction.dtype, device=direction.device)
    blocks = identity[:, None, :] * basis[None, :, None]
    return blocks.reshape(3, 3 * basis_count)


def view_direction_position_jacobian(
    mean_world: Tensor,
    camera_center_world: Tensor,
) -> Tensor:
    """Return ``d view_direction / d mean_world`` with shape ``(3,3)``.

    Args:
        mean_world: CPU float64 Gaussian center, shape ``(3,)``.
        camera_center_world: CPU float64 camera center, shape ``(3,)``.

    Returns:
        CPU float64 Cartesian Jacobian with shape ``(3,3)``.

    TeX: eq:view_direction_position_jacobian
    """

    validate_jacobian_tensor(mean_world, "mean_world", shape=(3,))
    validate_jacobian_tensor(camera_center_world, "camera_center_world", shape=(3,))
    offset = mean_world - camera_center_world
    norm = torch.linalg.vector_norm(offset)
    if not bool(norm > 0):
        raise ValueError("mean_world must not equal camera_center_world")
    direction = offset / norm
    identity = torch.eye(3, dtype=mean_world.dtype, device=mean_world.device)
    return (identity - torch.outer(direction, direction)) / norm


def color_view_direction_jacobian(
    direction: Tensor,
    sh_coefficients: Tensor,
) -> Tensor:
    """Return ``d theoretical_SH_color / d direction``, shape ``(3,3)``.

    Args:
        direction: CPU float64 view direction, shape ``(3,)``.
        sh_coefficients: CPU float64 coefficients, shape ``(B,3)``, where
            ``B`` is one of ``1,4,9,16`` and storage is basis-major.

    Returns:
        CPU float64 Jacobian ``(3,3)``; RGB rows and Cartesian columns.

    TeX: eq:color_view_direction_jacobian
    """

    validate_jacobian_tensor(direction, "direction", shape=(3,))
    validate_jacobian_tensor(sh_coefficients, "sh_coefficients", shape=(None, 3))
    basis_count = sh_coefficients.shape[0]
    if basis_count not in _VALID_BASIS_COUNTS:
        raise ValueError("sh_coefficients basis count B must be one of 1, 4, 9, or 16")
    basis_gradients = _real_sh_degree_3_gradients(direction)[:basis_count]
    return sh_coefficients.transpose(0, 1) @ basis_gradients


def color_world_position_jacobian(
    mean_world: Tensor,
    camera_center_world: Tensor,
    sh_coefficients: Tensor,
) -> Tensor:
    """Return ``d theoretical_SH_color / d mean_world``, shape ``(3,3)``.

    Args:
        mean_world: CPU float64 Gaussian center, shape ``(3,)``.
        camera_center_world: CPU float64 camera center, shape ``(3,)``.
        sh_coefficients: CPU float64 coefficients, shape ``(B,3)`` for
            ``B`` in ``{1,4,9,16}``, in basis-major order.

    Returns:
        CPU float64 Jacobian ``(3,3)``; RGB rows and world-position columns.

    TeX: eq:color_world_position_jacobian, eq:position_gradient_chain
    """

    validate_jacobian_tensor(mean_world, "mean_world", shape=(3,))
    validate_jacobian_tensor(camera_center_world, "camera_center_world", shape=(3,))
    validate_jacobian_tensor(sh_coefficients, "sh_coefficients", shape=(None, 3))
    offset = mean_world - camera_center_world
    norm = torch.linalg.vector_norm(offset)
    if not bool(norm > 0):
        raise ValueError("mean_world must not equal camera_center_world")
    direction = offset / norm
    return color_view_direction_jacobian(
        direction,
        sh_coefficients,
    ) @ view_direction_position_jacobian(mean_world, camera_center_world)


def recursive_color_opacity_jacobian(
    color: Tensor,
    next_recursive_color: Tensor,
) -> Tensor:
    """Return ``d recursive_color / d screen_opacity``, shape ``(3,)``.

    Args:
        color: CPU float64 Gaussian RGB, shape ``(3,)``.
        next_recursive_color: CPU float64 color behind it, shape ``(3,)``.

    Returns:
        CPU float64 RGB derivative with shape ``(3,)``.

    TeX: eq:recursive_color_opacity_jacobian, eq:pixel_color_backward,
    eq:transmittance_backward
    """

    validate_jacobian_tensor(color, "color", shape=(3,))
    validate_jacobian_tensor(next_recursive_color, "next_recursive_color", shape=(3,))

    def recursive_pixel_color(screen_opacity: Tensor) -> Tensor:
        """Evaluate the local recursive color relation.

        TeX: eq:recursive_pixel_color
        """

        return screen_opacity * color + (1.0 - screen_opacity) * next_recursive_color

    # Keep the reference recursion executable at a non-boundary value so its
    # definition is validated by the same finite checks as the derivative.
    reference = recursive_pixel_color(torch.tensor(0.5, dtype=torch.float64))
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("recursive pixel color contains NaN or Inf")
    return color - next_recursive_color


def pixel_color_recursive_jacobian(transmittance: Tensor | Real) -> Tensor:
    """Return ``d pixel_color / d recursive_color``, shape ``(3,3)``.

    Args:
        transmittance: CPU float64 scalar (or Python real) before this Gaussian.

    Returns:
        CPU float64 Jacobian ``transmittance * I3``, shape ``(3,3)``.

    TeX: eq:pixel_color_recursive_jacobian
    """

    transmittance_t = jacobian_scalar(transmittance, "transmittance")
    return transmittance_t * torch.eye(3, dtype=torch.float64, device="cpu")


def pixel_color_screen_opacity_jacobian(
    transmittance: Tensor | Real,
    color: Tensor,
    next_recursive_color: Tensor,
) -> Tensor:
    """Return ``d pixel_color / d screen_opacity``, shape ``(3,)``.

    Args:
        transmittance: CPU float64 scalar (or Python real) before this Gaussian.
        color: CPU float64 Gaussian RGB, shape ``(3,)``.
        next_recursive_color: CPU float64 color behind it, shape ``(3,)``.

    Returns:
        CPU float64 RGB derivative with shape ``(3,)``.

    TeX: eq:pixel_color_screen_opacity_jacobian
    """

    transmittance_t = jacobian_scalar(transmittance, "transmittance")
    return transmittance_t * recursive_color_opacity_jacobian(color, next_recursive_color)


def pixel_color_color_jacobian(
    transmittance: Tensor | Real,
    screen_opacity: Tensor | Real,
) -> Tensor:
    """Return ``d pixel_color / d Gaussian_color``, shape ``(3,3)``.

    Args:
        transmittance: CPU float64 scalar (or Python real) before this Gaussian.
        screen_opacity: CPU float64 scalar (or Python real) at this pixel.

    Returns:
        CPU float64 RGB Jacobian with shape ``(3,3)``.

    TeX: eq:pixel_color_color_jacobian
    """

    transmittance_t = jacobian_scalar(transmittance, "transmittance")
    screen_opacity_t = jacobian_scalar(screen_opacity, "screen_opacity")
    return (
        transmittance_t
        * screen_opacity_t
        * torch.eye(3, dtype=torch.float64, device="cpu")
    )


def screen_opacity_parameter_derivative(
    displacement: Tensor,
    inverse_covariance: Tensor,
) -> Tensor:
    """Return ``d screen_opacity / d opacity`` as a float64 scalar.

    Args:
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        inverse_covariance: CPU float64 symmetric inverse covariance,
            shape ``(2,2)``.

    Returns:
        CPU float64 scalar Gaussian weight.

    TeX: eq:screen_opacity_parameter_derivative, eq:screen_opacity_backward,
    eq:pixel_displacement_backward, eq:gaussian_weight_backward
    """

    displacement, inverse_covariance = _validate_opacity_geometry(
        displacement, inverse_covariance
    )
    gaussian_weight = torch.exp(
        -0.5 * displacement @ inverse_covariance @ displacement
    )
    if not bool(torch.isfinite(gaussian_weight)):
        raise ValueError("Gaussian weight is NaN or Inf")
    return gaussian_weight


def gaussian_weight_position_jacobian(
    displacement: Tensor,
    inverse_covariance: Tensor,
) -> Tensor:
    """Return ``d Gaussian_weight / d projected_position``, shape ``(2,)``.

    Args:
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        inverse_covariance: CPU float64 symmetric inverse covariance,
            shape ``(2,2)``.

    Returns:
        CPU float64 row-vector values in projected ``(x,y)`` order.

    TeX: eq:gaussian_weight_position_jacobian
    """

    displacement, inverse_covariance = _validate_opacity_geometry(
        displacement, inverse_covariance
    )
    gaussian_weight = screen_opacity_parameter_derivative(
        displacement, inverse_covariance
    )
    return -(inverse_covariance @ displacement) * gaussian_weight


def screen_opacity_position_jacobian(
    opacity: Tensor | Real,
    displacement: Tensor,
    inverse_covariance: Tensor,
) -> Tensor:
    """Return ``d screen_opacity / d projected_position``, shape ``(2,)``.

    Args:
        opacity: CPU float64 scalar (or Python real) Gaussian opacity.
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        inverse_covariance: CPU float64 symmetric inverse covariance,
            shape ``(2,2)``.

    Returns:
        CPU float64 derivative in projected ``(x,y)`` order.

    TeX: eq:screen_opacity_position_jacobian
    """

    opacity_t = jacobian_scalar(opacity, "opacity")
    return opacity_t * gaussian_weight_position_jacobian(
        displacement, inverse_covariance
    )


def gaussian_weight_inverse_covariance_jacobian(
    displacement: Tensor,
    inverse_covariance: Tensor,
) -> Tensor:
    """Return ``d Gaussian_weight / d packed_inverse_covariance``, shape ``(3,)``.

    Args:
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        inverse_covariance: CPU float64 symmetric inverse covariance,
            shape ``(2,2)``.

    Returns:
        CPU float64 derivative in packed ``(00,01,11)`` order.

    TeX: eq:gaussian_weight_inverse_covariance_jacobian
    """

    displacement, inverse_covariance = _validate_opacity_geometry(
        displacement, inverse_covariance
    )
    gaussian_weight = screen_opacity_parameter_derivative(
        displacement, inverse_covariance
    )
    dx, dy = displacement.unbind()
    return torch.stack((-0.5 * dx.square(), -dx * dy, -0.5 * dy.square())) * gaussian_weight


def inverse_covariance_jacobian(covariance_vector: Tensor) -> Tensor:
    """Return ``d packed(Sigma^-1) / d packed(Sigma)``, shape ``(3,3)``.

    Args:
        covariance_vector: CPU float64 covariance in packed
            ``(00,01,11)`` order, shape ``(3,)``.

    Returns:
        CPU float64 Jacobian ``(3,3)`` with both axes in ``(00,01,11)`` order.

    TeX: eq:inverse_covariance_jacobian, eq:inverse_covariance_vector,
    eq:screen_covariance_determinant, eq:screen_covariance_inverse_components
    """

    validate_jacobian_tensor(covariance_vector, "covariance_vector", shape=(3,))
    a, b, c = covariance_vector.unbind()
    determinant = a * c - b.square()
    if bool(determinant == 0):
        raise ValueError("covariance_vector must represent a nonsingular matrix")
    numerator = torch.stack(
        (
            torch.stack((-c.square(), 2.0 * b * c, -b.square())),
            torch.stack((b * c, -(a * c + b.square()), a * b)),
            torch.stack((-b.square(), 2.0 * a * b, -a.square())),
        )
    )
    return numerator / determinant.square()


def screen_opacity_covariance_jacobian(
    opacity: Tensor | Real,
    displacement: Tensor,
    covariance_vector: Tensor,
) -> Tensor:
    """Return ``d screen_opacity / d packed_covariance``, shape ``(3,)``.

    Args:
        opacity: CPU float64 scalar (or Python real) Gaussian opacity.
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        covariance_vector: CPU float64 covariance packed as ``(00,01,11)``,
            shape ``(3,)``.

    Returns:
        CPU float64 derivative in covariance order ``(00,01,11)``.

    TeX: eq:screen_opacity_covariance_jacobian
    """

    opacity_t = jacobian_scalar(opacity, "opacity")
    validate_jacobian_tensor(displacement, "displacement", shape=(2,))
    validate_jacobian_tensor(covariance_vector, "covariance_vector", shape=(3,))
    inverse_covariance = _inverse_covariance_from_vector(covariance_vector)
    weight_inverse_jacobian = gaussian_weight_inverse_covariance_jacobian(
        displacement,
        inverse_covariance,
    )
    return opacity_t * (weight_inverse_jacobian @ inverse_covariance_jacobian(covariance_vector))


def pixel_color_opacity_parameter_jacobian(
    transmittance: Tensor | Real,
    color: Tensor,
    next_recursive_color: Tensor,
    gaussian_weight: Tensor | Real,
) -> Tensor:
    """Return ``d pixel_color / d Gaussian_opacity``, shape ``(3,)``.

    Args:
        transmittance: CPU float64 scalar (or Python real) before this Gaussian.
        color: CPU float64 Gaussian RGB, shape ``(3,)``.
        next_recursive_color: CPU float64 color behind it, shape ``(3,)``.
        gaussian_weight: CPU float64 scalar (or Python real).

    Returns:
        CPU float64 RGB derivative with shape ``(3,)``.

    TeX: eq:pixel_color_opacity_parameter_jacobian, eq:opacity_gradient_chain
    """

    gaussian_weight_t = jacobian_scalar(gaussian_weight, "gaussian_weight")
    return pixel_color_screen_opacity_jacobian(
        transmittance,
        color,
        next_recursive_color,
    ) * gaussian_weight_t


def pixel_color_position_jacobian(
    transmittance: Tensor | Real,
    color: Tensor,
    next_recursive_color: Tensor,
    opacity: Tensor | Real,
    displacement: Tensor,
    inverse_covariance: Tensor,
) -> Tensor:
    """Return ``d pixel_color / d projected_position``, shape ``(3,2)``.

    Args:
        transmittance: CPU float64 scalar (or Python real) before this Gaussian.
        color: CPU float64 Gaussian RGB, shape ``(3,)``.
        next_recursive_color: CPU float64 color behind it, shape ``(3,)``.
        opacity: CPU float64 scalar (or Python real) Gaussian opacity.
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        inverse_covariance: CPU float64 symmetric inverse covariance,
            shape ``(2,2)``.

    Returns:
        CPU float64 Jacobian ``(3,2)``; RGB rows and projected ``(x,y)`` columns.

    TeX: eq:pixel_color_position_jacobian
    """

    color_derivative = pixel_color_screen_opacity_jacobian(
        transmittance,
        color,
        next_recursive_color,
    )
    opacity_derivative = screen_opacity_position_jacobian(
        opacity,
        displacement,
        inverse_covariance,
    )
    return torch.outer(color_derivative, opacity_derivative)


def pixel_color_covariance_jacobian(
    transmittance: Tensor | Real,
    color: Tensor,
    next_recursive_color: Tensor,
    opacity: Tensor | Real,
    displacement: Tensor,
    covariance_vector: Tensor,
) -> Tensor:
    """Return ``d pixel_color / d packed_covariance``, shape ``(3,3)``.

    Args:
        transmittance: CPU float64 scalar (or Python real) before this Gaussian.
        color: CPU float64 Gaussian RGB, shape ``(3,)``.
        next_recursive_color: CPU float64 color behind it, shape ``(3,)``.
        opacity: CPU float64 scalar (or Python real) Gaussian opacity.
        displacement: CPU float64 ``u-x_pixel`` vector, shape ``(2,)``.
        covariance_vector: CPU float64 covariance packed as ``(00,01,11)``,
            shape ``(3,)``.

    Returns:
        CPU float64 Jacobian ``(3,3)``; RGB rows and packed covariance columns.

    TeX: eq:pixel_color_covariance_jacobian
    """

    color_derivative = pixel_color_screen_opacity_jacobian(
        transmittance,
        color,
        next_recursive_color,
    )
    opacity_derivative = screen_opacity_covariance_jacobian(
        opacity,
        displacement,
        covariance_vector,
    )
    return torch.outer(color_derivative, opacity_derivative)


def _positive_jacobian_scalar(value: Tensor | Real, name: str) -> Tensor:
    """Validate a positive CPU float64 scalar."""

    result = jacobian_scalar(value, name)
    if not bool(result > 0):
        raise ValueError(f"{name} must be positive")
    return result


def _rotation_matrix(quaternion: Tensor) -> Tensor:
    """Evaluate the main rotation formula for one validated quaternion."""

    return quaternion_rotation_matrix(quaternion.unsqueeze(0))[0]


def _rotation_derivative(quaternion: Tensor, component: int) -> Tensor:
    """Evaluate one of TeX eq:rotation_q*_jacobian without revalidation."""

    qw, qx, qy, qz = quaternion.unbind()
    zero = quaternion.new_zeros(())
    two = quaternion.new_tensor(2.0)
    if component == 0:
        matrix = torch.stack(
            (
                torch.stack((zero, -qz, qy)),
                torch.stack((qz, zero, -qx)),
                torch.stack((-qy, qx, zero)),
            )
        )
    elif component == 1:
        matrix = torch.stack(
            (
                torch.stack((zero, qy, qz)),
                torch.stack((qy, -two * qx, -qw)),
                torch.stack((qz, qw, -two * qx)),
            )
        )
    elif component == 2:
        matrix = torch.stack(
            (
                torch.stack((-two * qy, qx, qw)),
                torch.stack((qx, zero, qz)),
                torch.stack((-qw, qz, -two * qy)),
            )
        )
    else:
        matrix = torch.stack(
            (
                torch.stack((-two * qz, -qw, qx)),
                torch.stack((qw, -two * qz, qy)),
                torch.stack((qx, qy, zero)),
            )
        )
    return two * matrix


def _validated_projection_inputs(
    point_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate common analytic projection-derivative inputs."""

    validate_jacobian_tensor(point_camera, "point_camera", shape=(3,))
    if not bool(point_camera[2] > 0):
        raise ValueError("point_camera must have strictly positive depth z")
    return (
        point_camera,
        _positive_jacobian_scalar(fx, "fx"),
        _positive_jacobian_scalar(fy, "fy"),
    )


def _real_sh_degree_3_gradients(direction: Tensor) -> Tensor:
    """Return gradients of all 16 TeX real-SH polynomials, shape ``(16,3)``."""

    x, y, z = direction.unbind()
    zero = direction.new_zeros(())
    c1 = direction.new_tensor(0.4886025119029199)
    c20 = direction.new_tensor(1.0925484305920792)
    c22 = direction.new_tensor(0.5462742152960396)
    c2z = direction.new_tensor(0.31539156525252005)
    c30 = direction.new_tensor(0.5900435899266435)
    c31 = direction.new_tensor(2.890611442640554)
    c32 = direction.new_tensor(0.4570457994644658)
    c33 = direction.new_tensor(0.3731763325901154)
    c34 = direction.new_tensor(1.445305721320277)

    return torch.stack(
        (
            torch.stack((zero, zero, zero)),
            torch.stack((zero, -c1, zero)),
            torch.stack((zero, zero, c1)),
            torch.stack((-c1, zero, zero)),
            torch.stack((c20 * y, c20 * x, zero)),
            torch.stack((zero, -c20 * z, -c20 * y)),
            torch.stack((zero, zero, 6.0 * c2z * z)),
            torch.stack((-c20 * z, zero, -c20 * x)),
            torch.stack((2.0 * c22 * x, -2.0 * c22 * y, zero)),
            torch.stack(
                (
                    -6.0 * c30 * x * y,
                    -3.0 * c30 * (x.square() - y.square()),
                    zero,
                )
            ),
            torch.stack((c31 * y * z, c31 * x * z, c31 * x * y)),
            torch.stack(
                (
                    zero,
                    -c32 * (5.0 * z.square() - 1.0),
                    -10.0 * c32 * y * z,
                )
            ),
            torch.stack((zero, zero, c33 * (15.0 * z.square() - 3.0))),
            torch.stack(
                (
                    -c32 * (5.0 * z.square() - 1.0),
                    zero,
                    -10.0 * c32 * x * z,
                )
            ),
            torch.stack(
                (
                    2.0 * c34 * x * z,
                    -2.0 * c34 * y * z,
                    c34 * (x.square() - y.square()),
                )
            ),
            torch.stack(
                (
                    -3.0 * c30 * (x.square() - y.square()),
                    6.0 * c30 * x * y,
                    zero,
                )
            ),
        )
    )


def _validate_opacity_geometry(
    displacement: Tensor,
    inverse_covariance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate single-pixel opacity geometry."""

    validate_jacobian_tensor(displacement, "displacement", shape=(2,))
    validate_jacobian_tensor(inverse_covariance, "inverse_covariance", shape=(2, 2))
    validate_symmetric(inverse_covariance, "inverse_covariance")
    return displacement, inverse_covariance


def _inverse_covariance_from_vector(covariance_vector: Tensor) -> Tensor:
    """Build the theoretical inverse used by the analytic Jacobians."""

    a, b, c = covariance_vector.unbind()
    determinant = a * c - b.square()
    if bool(determinant == 0):
        raise ValueError("covariance_vector must represent a nonsingular matrix")
    return torch.stack(
        (
            torch.stack((c, -b)),
            torch.stack((-b, a)),
        )
    ) / determinant
