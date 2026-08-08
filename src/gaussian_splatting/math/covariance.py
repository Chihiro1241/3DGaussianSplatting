"""Quaternion rotations and 3D/2D covariance calculations."""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor

from ._validation import (
    tensor_scalar,
    validate_float_tensor,
    validate_same_dtype_device,
    validate_symmetric,
)


def quaternion_rotation_matrix(quaternions: Tensor) -> Tensor:
    """Convert ``(qw, qx, qy, qz)`` quaternions to rotation matrices.

    Args:
        quaternions: Unit quaternions with shape ``(N, 4)``.

    Returns:
        Rotation matrices with shape ``(N, 3, 3)``.

    TeX: eq:quaternion_rotation_matrix
    """

    validate_float_tensor(quaternions, "quaternions", shape=(None, 4))
    qw, qx, qy, qz = quaternions.unbind(dim=-1)
    two = quaternions.new_tensor(2.0)
    one = quaternions.new_tensor(1.0)

    row_0 = torch.stack(
        (
            one - two * (qy.square() + qz.square()),
            two * (qx * qy - qz * qw),
            two * (qx * qz + qy * qw),
        ),
        dim=-1,
    )
    row_1 = torch.stack(
        (
            two * (qx * qy + qz * qw),
            one - two * (qx.square() + qz.square()),
            two * (qy * qz - qx * qw),
        ),
        dim=-1,
    )
    row_2 = torch.stack(
        (
            two * (qx * qz - qy * qw),
            two * (qy * qz + qx * qw),
            one - two * (qx.square() + qy.square()),
        ),
        dim=-1,
    )
    rotations = torch.stack((row_0, row_1, row_2), dim=-2)
    if not bool(torch.isfinite(rotations).all()):
        raise ValueError("rotation matrices contain NaN or Inf")
    return rotations


def scale_matrix(scales: Tensor) -> Tensor:
    """Place per-axis Gaussian scales on matrix diagonals.

    Args:
        scales: Axis scales with shape ``(N, 3)``.

    Returns:
        Diagonal scale matrices with shape ``(N, 3, 3)``.

    TeX: eq:scale_matrix
    """

    validate_float_tensor(scales, "scales", shape=(None, 3))
    return torch.diag_embed(scales)


def covariance_3d(rotation_matrices: Tensor, scales: Tensor) -> Tensor:
    """Construct world-space 3D Gaussian covariance matrices.

    Args:
        rotation_matrices: Rotation matrices with shape ``(N, 3, 3)``.
        scales: Axis scales with shape ``(N, 3)``.

    Returns:
        Covariance matrices with shape ``(N, 3, 3)``.
        Their packed order is ``(00, 01, 02, 11, 12, 22)``.

    TeX: eq:covariance_3d, eq:backward_3d_covariance
    """

    validate_float_tensor(rotation_matrices, "rotation_matrices", shape=(None, 3, 3))
    validate_float_tensor(scales, "scales", shape=(None, 3))
    validate_same_dtype_device(
        ("rotation_matrices", rotation_matrices),
        ("scales", scales),
    )
    if rotation_matrices.shape[0] != scales.shape[0]:
        raise ValueError("rotation_matrices and scales must have the same Gaussian count N")

    # A = R S, so scaling the columns of R avoids materializing S.
    factor = rotation_matrices * scales.unsqueeze(-2)
    covariances = factor @ factor.transpose(-1, -2)
    if not bool(torch.isfinite(covariances).all()):
        raise ValueError("3D covariances contain NaN or Inf")
    validate_symmetric(covariances, "3D covariances")
    return covariances


def covariance_2d(
    projection_jacobians: Tensor,
    rotation_cw: Tensor,
    covariances_3d: Tensor,
) -> Tensor:
    """Project 3D covariance matrices onto the image plane.

    Args:
        projection_jacobians: Perspective Jacobians with shape ``(N, 2, 3)``.
        rotation_cw: World-to-camera rotation with shape ``(3, 3)``.
        covariances_3d: World-space covariances with shape ``(N, 3, 3)``.

    Returns:
        Unstabilized image-space covariances with shape ``(N, 2, 2)``.
        Their packed order is ``(00, 01, 11)``.

    TeX: eq:covariance_2d, eq:camera_covariance_definition,
    eq:backward_2d_covariance
    """

    validate_float_tensor(
        projection_jacobians,
        "projection_jacobians",
        shape=(None, 2, 3),
    )
    validate_float_tensor(rotation_cw, "rotation_cw", shape=(3, 3))
    validate_float_tensor(covariances_3d, "covariances_3d", shape=(None, 3, 3))
    validate_same_dtype_device(
        ("projection_jacobians", projection_jacobians),
        ("rotation_cw", rotation_cw),
        ("covariances_3d", covariances_3d),
    )
    if projection_jacobians.shape[0] != covariances_3d.shape[0]:
        raise ValueError("projection_jacobians and covariances_3d must have the same N")
    validate_symmetric(covariances_3d, "covariances_3d")

    # TeX: eq:camera_covariance_definition
    covariance_camera = rotation_cw @ covariances_3d @ rotation_cw.transpose(0, 1)
    projected = projection_jacobians @ covariance_camera @ projection_jacobians.transpose(-1, -2)
    if not bool(torch.isfinite(projected).all()):
        raise ValueError("2D covariances contain NaN or Inf")
    # The expression is symmetric in exact arithmetic, but float32 matrix
    # multiplication can produce different round-off in the two off-diagonal
    # entries.  Do not reject that expected numerical asymmetry here: the
    # rendering pipeline applies ``symmetrized_2d_covariance`` immediately
    # afterwards, before stabilization and inversion validate symmetry.
    return projected


def covariance_2d_components(covariances_2d: Tensor) -> Tensor:
    """Return 2D covariance components in ``(00, 01, 11)`` order.

    Args:
        covariances_2d: Symmetric matrices with shape ``(N, 2, 2)``.

    Returns:
        Packed components with shape ``(N, 3)``.

    TeX: eq:covariance_2d_components
    """

    validate_float_tensor(covariances_2d, "covariances_2d", shape=(None, 2, 2))
    return pack_symmetric_2d(covariances_2d)


def inverse_2d_covariance(covariances_2d: Tensor) -> Tensor:
    """Invert nonsingular symmetric 2D covariance matrices analytically.

    Args:
        covariances_2d: Symmetric matrices with shape ``(N, 2, 2)``.

    Returns:
        Inverse matrices with shape ``(N, 2, 2)``.

    TeX: eq:inverse_2d_covariance
    """

    validate_float_tensor(covariances_2d, "covariances_2d", shape=(None, 2, 2))
    validate_symmetric(covariances_2d, "covariances_2d")
    a = covariances_2d[:, 0, 0]
    b = covariances_2d[:, 0, 1]
    c = covariances_2d[:, 1, 1]
    determinant = a * c - b.square()
    if bool((determinant == 0).any()):
        raise ValueError("covariances_2d must be nonsingular")
    result = _inverse_from_components(a, b, c, determinant)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("inverse 2D covariances contain NaN or Inf")
    return result


def symmetrized_2d_covariance(covariances_2d: Tensor) -> Tensor:
    """Symmetrize image-space covariances.

    Args:
        covariances_2d: Matrices with shape ``(N, 2, 2)``.

    Returns:
        Symmetric matrices with shape ``(N, 2, 2)``.

    TeX: eq:symmetrized_2d_covariance
    """

    validate_float_tensor(covariances_2d, "covariances_2d", shape=(None, 2, 2))
    return 0.5 * (covariances_2d + covariances_2d.transpose(-1, -2))


def stabilized_2d_covariance(
    covariances_2d: Tensor,
    epsilon_cov: Tensor | Real = 0.3,
) -> Tensor:
    """Add the rendering variance floor to symmetric 2D covariances.

    Args:
        covariances_2d: Symmetric matrices with shape ``(N, 2, 2)``.
        epsilon_cov: Positive diagonal variance in pixel squared units.

    Returns:
        Stabilized matrices with shape ``(N, 2, 2)``.

    TeX: eq:stabilized_2d_covariance
    """

    validate_float_tensor(covariances_2d, "covariances_2d", shape=(None, 2, 2))
    validate_symmetric(covariances_2d, "covariances_2d")
    epsilon = tensor_scalar(epsilon_cov, "epsilon_cov", covariances_2d, positive=True)
    identity = torch.eye(2, dtype=covariances_2d.dtype, device=covariances_2d.device)
    result = covariances_2d + epsilon * identity
    if not bool(torch.isfinite(result).all()):
        raise ValueError("stabilized 2D covariances contain NaN or Inf")
    return result


def safe_2d_covariance_determinant(
    covariances_2d: Tensor,
    epsilon_det: Tensor | Real = 1e-8,
) -> Tensor:
    """Clamp 2D covariance determinants away from zero.

    Args:
        covariances_2d: Symmetric matrices with shape ``(N, 2, 2)``.
        epsilon_det: Positive determinant floor.

    Returns:
        Safe determinants with shape ``(N,)``.

    TeX: eq:safe_2d_covariance_determinant,
    eq:screen_covariance_determinant
    """

    validate_float_tensor(covariances_2d, "covariances_2d", shape=(None, 2, 2))
    validate_symmetric(covariances_2d, "covariances_2d")
    epsilon = tensor_scalar(epsilon_det, "epsilon_det", covariances_2d, positive=True)
    determinant = (
        covariances_2d[:, 0, 0] * covariances_2d[:, 1, 1]
        - covariances_2d[:, 0, 1].square()
    )
    if not bool(torch.isfinite(determinant).all()):
        raise ValueError("2D covariance determinants contain NaN or Inf")
    return determinant.clamp_min(epsilon)


def stabilized_inverse_2d_covariance(
    covariances_2d: Tensor,
    epsilon_det: Tensor | Real = 1e-8,
) -> Tensor:
    """Invert stabilized 2D covariances using a safe determinant.

    Args:
        covariances_2d: Stabilized symmetric matrices with shape ``(N, 2, 2)``.
        epsilon_det: Positive determinant floor.

    Returns:
        Inverse covariance matrices with shape ``(N, 2, 2)``.

    TeX: eq:stabilized_inverse_2d_covariance
    """

    validate_float_tensor(covariances_2d, "covariances_2d", shape=(None, 2, 2))
    validate_symmetric(covariances_2d, "covariances_2d")
    determinant = safe_2d_covariance_determinant(covariances_2d, epsilon_det)
    a = covariances_2d[:, 0, 0]
    b = covariances_2d[:, 0, 1]
    c = covariances_2d[:, 1, 1]
    result = _inverse_from_components(a, b, c, determinant)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("stabilized inverse 2D covariances contain NaN or Inf")
    return result


def pack_symmetric_3d(matrices: Tensor) -> Tensor:
    """Pack symmetric 3D matrices as ``(00, 01, 02, 11, 12, 22)``.

    Args:
        matrices: Tensor with trailing shape ``(3, 3)``.

    Returns:
        Tensor with the same leading dimensions and trailing dimension ``6``.
    """

    validate_float_tensor(matrices, "matrices")
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"matrices must have trailing shape (3, 3); got {tuple(matrices.shape)}")
    validate_symmetric(matrices, "matrices")
    return torch.stack(
        (
            matrices[..., 0, 0],
            matrices[..., 0, 1],
            matrices[..., 0, 2],
            matrices[..., 1, 1],
            matrices[..., 1, 2],
            matrices[..., 2, 2],
        ),
        dim=-1,
    )


def unpack_symmetric_3d(vectors: Tensor) -> Tensor:
    """Unpack ``(..., 6)`` vectors into symmetric 3D matrices."""

    validate_float_tensor(vectors, "vectors")
    if vectors.ndim < 1 or vectors.shape[-1] != 6:
        raise ValueError(f"vectors must have trailing dimension 6; got {tuple(vectors.shape)}")
    v00, v01, v02, v11, v12, v22 = vectors.unbind(dim=-1)
    row_0 = torch.stack((v00, v01, v02), dim=-1)
    row_1 = torch.stack((v01, v11, v12), dim=-1)
    row_2 = torch.stack((v02, v12, v22), dim=-1)
    return torch.stack((row_0, row_1, row_2), dim=-2)


def pack_symmetric_2d(matrices: Tensor) -> Tensor:
    """Pack symmetric 2D matrices as ``(00, 01, 11)``.

    TeX: eq:screen_covariance_vectors, eq:inverse_covariance_vector
    """

    validate_float_tensor(matrices, "matrices")
    if matrices.ndim < 2 or matrices.shape[-2:] != (2, 2):
        raise ValueError(f"matrices must have trailing shape (2, 2); got {tuple(matrices.shape)}")
    validate_symmetric(matrices, "matrices")
    return torch.stack(
        (matrices[..., 0, 0], matrices[..., 0, 1], matrices[..., 1, 1]),
        dim=-1,
    )


def unpack_symmetric_2d(vectors: Tensor) -> Tensor:
    """Unpack ``(..., 3)`` vectors into symmetric 2D matrices."""

    validate_float_tensor(vectors, "vectors")
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError(f"vectors must have trailing dimension 3; got {tuple(vectors.shape)}")
    v00, v01, v11 = vectors.unbind(dim=-1)
    row_0 = torch.stack((v00, v01), dim=-1)
    row_1 = torch.stack((v01, v11), dim=-1)
    return torch.stack((row_0, row_1), dim=-2)


def _inverse_from_components(a: Tensor, b: Tensor, c: Tensor, determinant: Tensor) -> Tensor:
    """Build inverse matrices from symmetric components and determinants."""

    row_0 = torch.stack((c, -b), dim=-1)
    row_1 = torch.stack((-b, a), dim=-1)
    return torch.stack((row_0, row_1), dim=-2) / determinant[..., None, None]
