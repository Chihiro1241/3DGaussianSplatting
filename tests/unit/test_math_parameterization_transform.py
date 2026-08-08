from __future__ import annotations

import pytest
import torch

from gaussian_splatting.math.parameterization import raw_parameter_transformations
from gaussian_splatting.math.transform import (
    perspective_projection,
    projection_jacobian,
    view_direction,
    world_to_camera,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_raw_parameter_transformations__eq_raw_parameter_transformations(dtype):
    raw_quaternions = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=dtype)
    raw_scales = torch.log(torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype))
    raw_opacities = torch.tensor([[0.0]], dtype=dtype)

    quaternions, scales, opacities = raw_parameter_transformations(
        raw_quaternions, raw_scales, raw_opacities
    )

    assert quaternions.dtype == scales.dtype == opacities.dtype == dtype
    torch.testing.assert_close(
        quaternions, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    )
    torch.testing.assert_close(scales, torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype))
    torch.testing.assert_close(opacities, torch.tensor([[0.5]], dtype=dtype))


def test_raw_parameter_transformations_uses_epsilon_for_zero_quaternion():
    zeros = torch.zeros((1, 4), dtype=torch.float64)
    scales = torch.zeros((1, 3), dtype=torch.float64)
    opacities = torch.zeros((1, 1), dtype=torch.float64)
    quaternions, _, _ = raw_parameter_transformations(zeros, scales, opacities)
    torch.testing.assert_close(quaternions, zeros)


def test_world_to_camera__eq_world_to_camera():
    means = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    result = world_to_camera(means, torch.eye(3), torch.zeros(3))
    torch.testing.assert_close(result, means)


def test_perspective_projection__eq_perspective_projection():
    point = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    projected = perspective_projection(point, 100.0, 120.0, 40.0, 30.0)
    torch.testing.assert_close(projected, torch.tensor([[40.0, 30.0]], dtype=torch.float64))


def test_projection_jacobian__eq_projection_jacobian():
    point = torch.tensor([[2.0, -3.0, 4.0]], dtype=torch.float64)
    actual = projection_jacobian(point, 8.0, 12.0)
    expected = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, 2.25]]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_view_direction__eq_view_direction():
    means = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)
    direction = view_direction(means, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(direction, torch.tensor([[0.6, 0.8, 0.0]], dtype=torch.float64))


def test_math_functions_reject_mixed_dtype_and_nonpositive_depth():
    with pytest.raises(TypeError, match="same dtype"):
        world_to_camera(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="positive depth"):
        perspective_projection(
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            1.0,
            1.0,
            0.0,
            0.0,
        )
