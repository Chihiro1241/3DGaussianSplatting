from __future__ import annotations

import pytest
import torch

from gaussian_splatting.math.spherical_harmonics import (
    real_sh_degree_3,
    sh_color_implementation,
    sh_color_theory,
)


def test_real_sh_degree_3__eq_real_sh_degree_3_official_signs():
    direction = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    expected = torch.tensor(
        [[
            0.28209479177387814,
            0.0,
            0.0,
            -0.4886025119029199,
            0.0,
            0.0,
            -0.31539156525252005,
            0.0,
            0.5462742152960396,
            0.0,
            0.0,
            0.0,
            0.0,
            0.4570457994644658,
            0.0,
            -0.5900435899266435,
        ]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(real_sh_degree_3(direction), expected)


def test_sh_color_theory__eq_sh_color_theory_uses_active_degree():
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    coefficients = torch.zeros((1, 16, 3), dtype=torch.float64)
    coefficients[:, 0] = 1.0
    coefficients[:, 2] = 100.0
    color = sh_color_theory(direction, coefficients, active_degree=0)
    torch.testing.assert_close(color, torch.full((1, 3), 0.28209479177387814, dtype=torch.float64))


def test_sh_color_implementation__eq_sh_color_implementation_zero_coefficients():
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    coefficients = torch.zeros((1, 16, 3), dtype=torch.float32)
    torch.testing.assert_close(
        sh_color_implementation(directions, coefficients),
        torch.full((1, 3), 0.5),
    )


def test_sh_color_implementation_clamps_only_below_zero():
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    coefficients = torch.zeros((1, 16, 3), dtype=torch.float64)
    coefficients[0, 0] = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64)
    color = sh_color_implementation(directions, coefficients)
    assert color[0, 0] == 0
    assert color[0, 1] == 0.5
    assert color[0, 2] > 1.0


def test_sh_validation_rejects_bad_degree_and_mixed_dtype():
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    coefficients = torch.zeros((1, 16, 3), dtype=torch.float64)
    with pytest.raises(TypeError, match="same dtype"):
        sh_color_theory(direction, coefficients)
    with pytest.raises(ValueError, match="active_degree"):
        sh_color_theory(direction.double(), coefficients, active_degree=4)
