from __future__ import annotations

import torch
import pytest

from gaussian_splatting.evaluation.metrics import mean_psnr, mse, psnr
from gaussian_splatting.training.losses import (
    dssim_loss,
    l1_loss,
    l1_norm,
    ssim,
    ssim_constants,
    total_loss,
)


def test_l1_norm__eq_l1_norm() -> None:
    rgb = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(l1_norm(rgb), torch.tensor([6.0], dtype=torch.float64))


def test_l1_loss__eq_l1_loss() -> None:
    rendered = torch.zeros((3, 2, 2), dtype=torch.float32)
    target = torch.ones_like(rendered)
    torch.testing.assert_close(l1_loss(rendered, target), torch.tensor(1.0))


def test_ssim_and_dssim_identical_images__eq_ssim__eq_dssim_loss() -> None:
    torch.manual_seed(0)
    image = torch.rand((3, 16, 16), dtype=torch.float64)
    torch.testing.assert_close(ssim(image, image), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(dssim_loss(image, image), torch.tensor(0.0, dtype=torch.float64))


def test_ssim_constants__eq_ssim_constants() -> None:
    assert ssim_constants() == (0.0001, 0.0009)


def test_total_loss__eq_total_loss() -> None:
    rendered = torch.zeros((3, 12, 12), dtype=torch.float32)
    target = torch.ones_like(rendered)
    result = total_loss(rendered, target, lambda_dssim=0.2)
    torch.testing.assert_close(result.total, 0.8 * result.l1 + 0.2 * result.dssim)
    torch.testing.assert_close(result.dssim, 1.0 - result.ssim)


def test_mse_psnr_and_mean__eq_mse__eq_psnr__eq_mean_psnr() -> None:
    image = torch.zeros((3, 2, 2), dtype=torch.float64)
    torch.testing.assert_close(mse(image, image), torch.tensor(0.0, dtype=torch.float64))
    assert torch.isposinf(psnr(image, image))
    values = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    torch.testing.assert_close(mean_psnr(values), torch.tensor(20.0, dtype=torch.float64))


def test_mean_psnr_rejects_mixed_list_dtypes() -> None:
    with pytest.raises(TypeError, match="same dtype"):
        mean_psnr(
            [
                torch.tensor(10.0, dtype=torch.float32),
                torch.tensor(20.0, dtype=torch.float64),
            ]
        )


def test_losses_preserve_autograd() -> None:
    rendered = torch.full((3, 12, 12), 0.25, dtype=torch.float64, requires_grad=True)
    target = torch.full_like(rendered, 0.75)
    result = total_loss(rendered, target)
    result.total.backward()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
