"""Camera data representation and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor


_FLOAT_DTYPES = (torch.float32, torch.float64)


def _validate_tensor(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_real(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Camera:
    """An immutable OpenCV-convention camera with an optional ground-truth image."""

    rotation_cw: Tensor
    translation_cw: Tensor
    camera_center_world: Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image: Tensor | None = None
    image_name: str | None = None

    def __post_init__(self) -> None:
        _validate_tensor("rotation_cw", self.rotation_cw, (3, 3))
        _validate_tensor("translation_cw", self.translation_cw, (3,))
        _validate_tensor("camera_center_world", self.camera_center_world, (3,))
        if not (
            self.rotation_cw.dtype
            == self.translation_cw.dtype
            == self.camera_center_world.dtype
        ):
            raise TypeError("camera pose tensors must have the same dtype")
        if not (
            self.rotation_cw.device
            == self.translation_cw.device
            == self.camera_center_world.device
        ):
            raise ValueError("camera pose tensors must be on the same device")

        for name in ("fx", "fy", "cx", "cy"):
            _validate_real(name, getattr(self, name), positive=True)
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.image is not None:
            _validate_tensor("image", self.image, (3, self.height, self.width))
            if torch.any((self.image < 0.0) | (self.image > 1.0)).item():
                raise ValueError("image values must be in [0, 1]")
        if self.image_name is not None and not isinstance(self.image_name, str):
            raise TypeError("image_name must be a string or None")

        torch.testing.assert_close(
            self.translation_cw,
            -self.rotation_cw @ self.camera_center_world,
            rtol=1e-5,
            atol=1e-6,
        )


__all__ = ["Camera"]
