"""Conversion of a Blender camera/image directory into :class:`Camera` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor

from gaussian_splatting.config import Config, DataConfig
from gaussian_splatting.data.blender_loader import (
    blender_c2w_to_opencv_w2c,
    read_camera_poses_json,
)
from gaussian_splatting.data.camera import Camera


def _data_config(config: Config | DataConfig) -> DataConfig:
    if isinstance(config, Config):
        return config.data
    if isinstance(config, DataConfig):
        return config
    raise TypeError("config must be Config or DataConfig")


def _scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    if scaled_width <= 0 or scaled_height <= 0:
        raise ValueError("resolution_scale produces a zero-sized image")
    return scaled_width, scaled_height


def _resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    # Match the official graphdeco-inria implementation's PILtoTorch helper,
    # which calls Image.resize(resolution) without overriding Pillow's
    # mode-dependent default resampling filter.
    return image.resize(size)


def _image_to_tensor(image: Image.Image, rgba_background: str) -> Tensor:
    if image.mode == "RGB":
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    elif image.mode == "RGBA":
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        alpha = rgba[..., 3:4]
        background_value = 0.0 if rgba_background == "black" else 1.0
        rgb = rgba[..., :3] * alpha + background_value * (1.0 - alpha)
    elif image.mode in {"1", "L", "I", "F"}:
        grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        rgb = np.repeat(grayscale[..., None], 3, axis=2)
    else:
        raise ValueError(
            f"unsupported image mode {image.mode!r}; expected grayscale, RGB, or RGBA"
        )
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def load_blender_dataset(
    data_directory: str | Path,
    config: Config | DataConfig,
) -> list[Camera]:
    """Load validated Blender poses and their PNG images as a camera list."""

    data_config = _data_config(config)
    root = Path(data_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    if data_config.rgba_background not in {"black", "white"}:
        raise ValueError("data.rgba_background must be black or white")
    if not 0.0 < data_config.resolution_scale <= 1.0:
        raise ValueError("data.resolution_scale must satisfy 0 < value <= 1")

    metadata = read_camera_poses_json(root / data_config.camera_file)
    original_width = metadata["resolution_x"]
    original_height = metadata["resolution_y"]
    width, height = _scaled_size(
        original_width, original_height, data_config.resolution_scale
    )

    base_focal = metadata["lens_mm"] / metadata["sensor_width_mm"] * original_width
    scale_x = width / original_width
    scale_y = height / original_height
    fx = float(base_focal * scale_x)
    fy = float(base_focal * scale_y)
    cx = float((original_width / 2.0) * scale_x)
    cy = float((original_height / 2.0) * scale_y)

    cameras: list[Camera] = []
    for frame in metadata["frames"]:
        image_path = root / frame["file_path"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image for frame {frame['index']} does not exist: {image_path}"
            )
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.size != (original_width, original_height):
                raise ValueError(
                    f"image {image_path} has EXIF-corrected size {image.size}, expected "
                    f"{(original_width, original_height)} from the camera JSON"
                )
            image = _resize(image, (width, height))
            image_tensor = _image_to_tensor(image, data_config.rgba_background)

        c2w_blender = torch.tensor(
            frame["transform_matrix_blender_c2w"], dtype=torch.float32
        )
        rotation_cw, translation_cw, camera_center_world = (
            blender_c2w_to_opencv_w2c(c2w_blender)
        )
        cameras.append(
            Camera(
                rotation_cw=rotation_cw,
                translation_cw=translation_cw,
                camera_center_world=camera_center_world,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                image=image_tensor,
                image_name=frame["file_path"],
            )
        )
    return cameras


def split_cameras(
    cameras: Sequence[Camera], test_every: int
) -> tuple[list[Camera], list[Camera]]:
    """Split image-name-sorted cameras into train and evaluation views.

    The zero-based sorted position ``k`` is an evaluation view exactly when
    ``k % test_every == 0``.
    """

    if type(test_every) is not int:
        raise TypeError("test_every must be an integer")
    if test_every <= 1:
        raise ValueError("test_every must be greater than 1")
    if any(camera.image_name is None for camera in cameras):
        raise ValueError("every camera must have image_name before splitting")

    ordered = sorted(cameras, key=lambda camera: camera.image_name or "")
    train = [camera for index, camera in enumerate(ordered) if index % test_every != 0]
    evaluation = [camera for index, camera in enumerate(ordered) if index % test_every == 0]
    if not train or not evaluation:
        raise ValueError("camera split must produce non-empty train and evaluation sets")
    return train, evaluation


__all__ = ["load_blender_dataset", "split_cameras"]
