"""Degree-three real spherical harmonics with the 3DGS sign convention."""

from __future__ import annotations

import torch
from torch import Tensor

from ._validation import (
    validate_active_degree,
    validate_float_tensor,
    validate_same_dtype_device,
)


def real_sh_degree_3(directions: Tensor) -> Tensor:
    """Evaluate the 16 real SH bases for degrees zero through three.

    Args:
        directions: Unit directions with trailing dimension ``3``.

    Returns:
        Basis values with the same leading dimensions and trailing dimension
        ``16``. The order is increasing ``l``, then increasing ``m``; the
        index is ``b(l, m) = l**2 + l + m``.

    TeX: eq:real_sh_degree_3
    """

    validate_float_tensor(directions, "directions")
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(
            f"directions must have trailing dimension 3; got {tuple(directions.shape)}"
        )

    x, y, z = directions.unbind(dim=-1)
    basis = torch.stack(
        (
            torch.ones_like(x) * 0.28209479177387814,
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z.square() - 1.0),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x.square() - y.square()),
            -0.5900435899266435 * y * (3.0 * x.square() - y.square()),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (5.0 * z.square() - 1.0),
            0.3731763325901154 * z * (5.0 * z.square() - 3.0),
            -0.4570457994644658 * x * (5.0 * z.square() - 1.0),
            1.445305721320277 * z * (x.square() - y.square()),
            -0.5900435899266435 * x * (x.square() - 3.0 * y.square()),
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("real SH basis values contain NaN or Inf")
    return basis


def sh_color_theory(
    directions: Tensor,
    sh_coefficients: Tensor,
    active_degree: int = 3,
) -> Tensor:
    """Compute the linear real-SH color without implementation bias/clamp.

    Args:
        directions: Unit view directions with shape ``(N, 3)``.
        sh_coefficients: RGB coefficients with shape ``(N, 16, 3)`` in
            basis-major order.
        active_degree: Highest active SH degree in ``[0, 3]``.

    Returns:
        RGB values with shape ``(N, 3)``.

    TeX: eq:sh_color_theory, eq:backward_sh_color
    """

    validate_float_tensor(directions, "directions", shape=(None, 3))
    validate_float_tensor(sh_coefficients, "sh_coefficients", shape=(None, 16, 3))
    validate_same_dtype_device(
        ("directions", directions),
        ("sh_coefficients", sh_coefficients),
    )
    if directions.shape[0] != sh_coefficients.shape[0]:
        raise ValueError("directions and sh_coefficients must have the same N")
    degree = validate_active_degree(active_degree)
    basis_count = (degree + 1) ** 2
    basis = real_sh_degree_3(directions)[:, :basis_count]
    colors = torch.sum(
        sh_coefficients[:, :basis_count, :] * basis.unsqueeze(-1),
        dim=1,
    )
    if not bool(torch.isfinite(colors).all()):
        raise ValueError("SH colors contain NaN or Inf")
    return colors


def sh_color_implementation(
    directions: Tensor,
    sh_coefficients: Tensor,
    active_degree: int = 3,
) -> Tensor:
    """Compute renderer SH color with ``+0.5`` and a nonnegative clamp.

    Args:
        directions: Unit view directions with shape ``(N, 3)``.
        sh_coefficients: RGB coefficients with shape ``(N, 16, 3)`` in
            basis-major order.
        active_degree: Highest active SH degree in ``[0, 3]``.

    Returns:
        Nonnegative RGB values with shape ``(N, 3)``.

    TeX: eq:sh_color_implementation
    """

    return (sh_color_theory(directions, sh_coefficients, active_degree) + 0.5).clamp_min(0.0)
