"""Transform unconstrained Gaussian parameters into rendering parameters."""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor

from ._validation import (
    validate_float_tensor,
    validate_positive_real,
    validate_same_dtype_device,
)


def raw_parameter_transformations(
    raw_quaternions: Tensor,
    raw_scales: Tensor,
    raw_opacities: Tensor,
    epsilon_q: Real = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Transform raw quaternion, scale, and opacity parameters.

    Args:
        raw_quaternions: Raw quaternions with shape ``(N, 4)`` in
            ``(qw, qx, qy, qz)`` order.
        raw_scales: Raw axis scales with shape ``(N, 3)``.
        raw_opacities: Raw opacities with shape ``(N, 1)``.
        epsilon_q: Positive lower bound for the quaternion norm.

    Returns:
        ``(quaternions, scales, opacities)`` with shapes ``(N, 4)``,
        ``(N, 3)``, and ``(N, 1)`` respectively.

    TeX: eq:raw_parameter_transformations
    """

    validate_float_tensor(raw_quaternions, "raw_quaternions", shape=(None, 4))
    validate_float_tensor(raw_scales, "raw_scales", shape=(None, 3))
    validate_float_tensor(raw_opacities, "raw_opacities", shape=(None, 1))
    validate_same_dtype_device(
        ("raw_quaternions", raw_quaternions),
        ("raw_scales", raw_scales),
        ("raw_opacities", raw_opacities),
    )
    if raw_scales.shape[0] != raw_quaternions.shape[0] or raw_opacities.shape[0] != raw_quaternions.shape[0]:
        raise ValueError("raw parameter tensors must have the same Gaussian count N")

    epsilon = validate_positive_real(epsilon_q, "epsilon_q")
    quaternion_norms = torch.linalg.vector_norm(raw_quaternions, dim=-1, keepdim=True)
    quaternions = raw_quaternions / quaternion_norms.clamp_min(epsilon)
    scales = torch.exp(raw_scales)
    opacities = torch.sigmoid(raw_opacities)

    if not bool(torch.isfinite(quaternions).all()):
        raise ValueError("transformed quaternions contain NaN or Inf")
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("transformed scales contain NaN or Inf")
    if not bool(torch.isfinite(opacities).all()):
        raise ValueError("transformed opacities contain NaN or Inf")
    return quaternions, scales, opacities
