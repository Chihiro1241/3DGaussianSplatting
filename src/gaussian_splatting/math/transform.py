"""Coordinate transforms and perspective-projection mathematics."""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor

from ._validation import (
    tensor_scalar,
    validate_float_tensor,
    validate_same_dtype_device,
)


def world_to_camera(
    means_world: Tensor,
    rotation_cw: Tensor,
    translation_cw: Tensor,
) -> Tensor:
    """Transform world-space centers into camera coordinates.

    Args:
        means_world: World-space points with shape ``(N, 3)``.
        rotation_cw: World-to-camera rotation with shape ``(3, 3)``.
        translation_cw: World-to-camera translation with shape ``(3,)``.

    Returns:
        Camera-space points with shape ``(N, 3)``.

    TeX: eq:world_to_camera, eq:backward_world_to_camera
    """

    validate_float_tensor(means_world, "means_world", shape=(None, 3))
    validate_float_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    validate_float_tensor(translation_cw, "translation_cw", shape=(3,))
    validate_same_dtype_device(
        ("means_world", means_world),
        ("rotation_cw", rotation_cw),
        ("translation_cw", translation_cw),
    )

    means_camera = means_world @ rotation_cw.transpose(0, 1) + translation_cw
    if not bool(torch.isfinite(means_camera).all()):
        raise ValueError("world-to-camera output contains NaN or Inf")
    return means_camera


def perspective_projection(
    points_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
    cx: Tensor | Real,
    cy: Tensor | Real,
) -> Tensor:
    """Project positive-depth camera-space points onto the image plane.

    Args:
        points_camera: Camera-space points ``(N, 3)`` in ``(x, y, z)`` order.
        fx: Positive horizontal focal length.
        fy: Positive vertical focal length.
        cx: Horizontal principal point.
        cy: Vertical principal point.

    Returns:
        Image coordinates ``(N, 2)`` in ``(x, y)`` order.

    TeX: eq:perspective_projection, eq:backward_perspective_projection
    """

    validate_float_tensor(points_camera, "points_camera", shape=(None, 3))
    fx_t = tensor_scalar(fx, "fx", points_camera, positive=True)
    fy_t = tensor_scalar(fy, "fy", points_camera, positive=True)
    cx_t = tensor_scalar(cx, "cx", points_camera)
    cy_t = tensor_scalar(cy, "cy", points_camera)
    if not bool((points_camera[:, 2] > 0).all()):
        raise ValueError("points_camera must have strictly positive depth z")

    x, y, z = points_camera.unbind(dim=-1)
    projected = torch.stack((fx_t * x / z + cx_t, fy_t * y / z + cy_t), dim=-1)
    if not bool(torch.isfinite(projected).all()):
        raise ValueError("perspective projection contains NaN or Inf")
    return projected


def projection_jacobian(
    points_camera: Tensor,
    fx: Tensor | Real,
    fy: Tensor | Real,
) -> Tensor:
    """Compute the perspective-projection Jacobian for each point.

    Args:
        points_camera: Positive-depth camera points with shape ``(N, 3)``.
        fx: Positive horizontal focal length.
        fy: Positive vertical focal length.

    Returns:
        Jacobians with shape ``(N, 2, 3)``.

    TeX: eq:projection_jacobian
    """

    validate_float_tensor(points_camera, "points_camera", shape=(None, 3))
    fx_t = tensor_scalar(fx, "fx", points_camera, positive=True)
    fy_t = tensor_scalar(fy, "fy", points_camera, positive=True)
    if not bool((points_camera[:, 2] > 0).all()):
        raise ValueError("points_camera must have strictly positive depth z")

    x, y, z = points_camera.unbind(dim=-1)
    zeros = torch.zeros_like(z)
    first_row = torch.stack((fx_t / z, zeros, -fx_t * x / z.square()), dim=-1)
    second_row = torch.stack((zeros, fy_t / z, -fy_t * y / z.square()), dim=-1)
    jacobian = torch.stack((first_row, second_row), dim=-2)
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError("projection Jacobian contains NaN or Inf")
    return jacobian


def view_direction(means_world: Tensor, camera_center_world: Tensor) -> Tensor:
    """Return camera-to-Gaussian unit directions in world coordinates.

    Args:
        means_world: Gaussian centers with shape ``(N, 3)``.
        camera_center_world: Camera center with shape ``(3,)``.

    Returns:
        Unit directions with shape ``(N, 3)``.

    TeX: eq:view_direction, eq:backward_view_direction
    """

    validate_float_tensor(means_world, "means_world", shape=(None, 3))
    validate_float_tensor(camera_center_world, "camera_center_world", shape=(3,))
    validate_same_dtype_device(
        ("means_world", means_world),
        ("camera_center_world", camera_center_world),
    )
    offsets = means_world - camera_center_world
    norms = torch.linalg.vector_norm(offsets, dim=-1, keepdim=True)
    if not bool((norms > 0).all()):
        raise ValueError("a Gaussian center must not equal the camera center")
    directions = offsets / norms
    if not bool(torch.isfinite(directions).all()):
        raise ValueError("view directions contain NaN or Inf")
    return directions
