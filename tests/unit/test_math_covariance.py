from __future__ import annotations

import pytest
import torch

from gaussian_splatting.math.covariance import (
    covariance_2d,
    covariance_2d_components,
    covariance_3d,
    inverse_2d_covariance,
    pack_symmetric_2d,
    pack_symmetric_3d,
    quaternion_rotation_matrix,
    safe_2d_covariance_determinant,
    scale_matrix,
    stabilized_2d_covariance,
    stabilized_inverse_2d_covariance,
    symmetrized_2d_covariance,
    unpack_symmetric_2d,
    unpack_symmetric_3d,
)


def test_quaternion_rotation_matrix__eq_quaternion_rotation_matrix():
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        quaternion_rotation_matrix(quaternion),
        torch.eye(3, dtype=torch.float64).unsqueeze(0),
    )


def test_scale_matrix__eq_scale_matrix():
    scales = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(scale_matrix(scales), torch.diag_embed(scales))


def test_covariance_3d__eq_covariance_3d_is_symmetric_psd_and_isotropic():
    angle = torch.tensor(0.63, dtype=torch.float64)
    quaternion = torch.stack(
        (
            torch.cos(angle / 2),
            angle.new_tensor(0.0),
            angle.new_tensor(0.0),
            torch.sin(angle / 2),
        )
    )[None]
    rotation = quaternion_rotation_matrix(quaternion)
    scales = torch.full((1, 3), 2.0, dtype=torch.float64)
    covariance = covariance_3d(rotation, scales)
    torch.testing.assert_close(covariance, 4.0 * torch.eye(3, dtype=torch.float64)[None])
    assert bool((torch.linalg.eigvalsh(covariance) >= 0).all())


def test_covariance_2d_and_stabilization_equations():
    jacobian = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]], dtype=torch.float64)
    covariance3d = torch.eye(3, dtype=torch.float64)[None]
    projected = covariance_2d(jacobian, torch.eye(3, dtype=torch.float64), covariance3d)
    torch.testing.assert_close(projected, torch.diag_embed(torch.tensor([[4.0, 9.0]], dtype=torch.float64)))

    asymmetric = projected.clone()
    asymmetric[0, 0, 1] = 2e-7
    symmetric = symmetrized_2d_covariance(asymmetric)
    stabilized = stabilized_2d_covariance(symmetric, 0.3)
    torch.testing.assert_close(stabilized.diagonal(dim1=-2, dim2=-1), torch.tensor([[4.3, 9.3]], dtype=torch.float64))
    assert bool((torch.linalg.eigvalsh(stabilized) > 0).all())


def test_covariance_2d_allows_float32_roundoff_before_explicit_symmetrization():
    # Regression for a real Blender camera/Gaussian combination.  In float32,
    # the mathematically equal off-diagonal products differ by about 2.5e-4,
    # which is expected to be removed by the documented symmetrization step.
    jacobian = torch.tensor(
        [[[412.85965, 0.0, -0.72415996], [0.0, 412.85965, 4.3821344]]],
        dtype=torch.float32,
    )
    rotation_cw = torch.tensor(
        [
            [-0.5877853, -0.8090168, 7.0443505e-8],
            [-0.5200260, 0.3778210, -0.7660445],
            [0.6197429, -0.4502699, -0.6427875],
        ],
        dtype=torch.float32,
    )
    covariance3d = 0.030340895 * torch.eye(3, dtype=torch.float32)[None]

    projected = covariance_2d(jacobian, rotation_cw, covariance3d)
    symmetric = symmetrized_2d_covariance(projected)

    assert torch.equal(symmetric, symmetric.transpose(-1, -2))
    stabilized = stabilized_2d_covariance(symmetric, 0.3)
    assert bool(torch.isfinite(stabilized).all())
    assert bool((torch.linalg.eigvalsh(stabilized) > 0).all())


def test_inverse_2d_covariance_equations():
    covariance = torch.tensor([[[2.0, 0.25], [0.25, 1.0]]], dtype=torch.float64)
    inverse = inverse_2d_covariance(covariance)
    torch.testing.assert_close(covariance @ inverse, torch.eye(2, dtype=torch.float64)[None])
    safe_inverse = stabilized_inverse_2d_covariance(covariance)
    torch.testing.assert_close(safe_inverse, inverse)
    torch.testing.assert_close(
        safe_2d_covariance_determinant(covariance),
        torch.tensor([1.9375], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("pack", "unpack", "matrix"),
    [
        (
            pack_symmetric_2d,
            unpack_symmetric_2d,
            torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        ),
        (
            pack_symmetric_3d,
            unpack_symmetric_3d,
            torch.tensor(
                [[2.0, 0.5, 0.1], [0.5, 1.0, -0.2], [0.1, -0.2, 3.0]],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_symmetric_pack_round_trip(pack, unpack, matrix):
    torch.testing.assert_close(unpack(pack(matrix)), matrix)


def test_covariance_2d_components__eq_covariance_2d_components():
    covariance = torch.tensor([[[2.0, 0.5], [0.5, 1.0]]])
    torch.testing.assert_close(covariance_2d_components(covariance), torch.tensor([[2.0, 0.5, 1.0]]))


def test_covariance_validation_rejects_asymmetry_and_dtype_mixture():
    with pytest.raises(ValueError, match="symmetric"):
        inverse_2d_covariance(torch.tensor([[[1.0, 1.0], [0.0, 1.0]]]))
    with pytest.raises(TypeError, match="same dtype"):
        covariance_3d(torch.eye(3, dtype=torch.float32)[None], torch.ones((1, 3), dtype=torch.float64))
