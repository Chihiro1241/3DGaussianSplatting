"""Conversion of a Blender camera/image directory into :class:`Camera` objects."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import math
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
from gaussian_splatting.data.colmap_loader import qvec_to_rotation_cw, read_colmap_model
from gaussian_splatting.data.nerf_synthetic_loader import (
    SPLITS,
    read_nerf_synthetic_json,
    resolve_nerf_image_path,
)


@dataclass(frozen=True)
class CameraSplits:
    """Dataset cameras preserving their source-defined splits."""

    train: list[Camera]
    val: list[Camera]
    test: list[Camera]
    scene_center: tuple[float, float, float]
    format: str
    initial_points: Tensor | None = None
    initial_colors: Tensor | None = None


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


def load_nerf_synthetic_dataset(
    data_directory: str | Path,
    config: Config | DataConfig,
    splits: Sequence[str] = SPLITS,
) -> CameraSplits:
    """Load all three source-defined splits of a NeRF Synthetic scene."""

    data_config = _data_config(config)
    root = Path(data_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    if data_config.rgba_background not in {"black", "white"}:
        raise ValueError("data.rgba_background must be black or white")
    if not 0.0 < data_config.resolution_scale <= 1.0:
        raise ValueError("data.resolution_scale must satisfy 0 < value <= 1")

    requested = tuple(splits)
    unknown = sorted(set(requested) - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown NeRF Synthetic splits: {', '.join(unknown)}")
    loaded: dict[str, list[Camera]] = {split: [] for split in SPLITS}
    for split in requested:
        metadata = read_nerf_synthetic_json(root / f"transforms_{split}.json")
        cameras: list[Camera] = []
        expected_size: tuple[int, int] | None = None
        for frame in metadata["frames"]:
            image_path = resolve_nerf_image_path(root, frame["file_path"])
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened)
                if expected_size is None:
                    expected_size = image.size
                elif image.size != expected_size:
                    raise ValueError(
                        f"image {image_path} has size {image.size}, expected {expected_size}"
                    )
                original_width, original_height = image.size
                width, height = _scaled_size(
                    original_width, original_height, data_config.resolution_scale
                )
                image_tensor = _image_to_tensor(
                    _resize(image, (width, height)), data_config.rgba_background
                )
            focal = 0.5 * original_width / math.tan(0.5 * metadata["camera_angle_x"])
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            rotation, translation, center = blender_c2w_to_opencv_w2c(c2w)
            cameras.append(
                Camera(
                    rotation_cw=rotation,
                    translation_cw=translation,
                    camera_center_world=center,
                    fx=float(focal * width / original_width),
                    fy=float(focal * height / original_height),
                    cx=float(width / 2.0),
                    cy=float(height / 2.0),
                    width=width,
                    height=height,
                    image=image_tensor,
                    image_name=str(image_path.relative_to(root)),
                )
            )
        loaded[split] = cameras
    return CameraSplits(
        train=loaded["train"], val=loaded["val"], test=loaded["test"],
        scene_center=(0.0, 0.0, 0.0), format="nerf_synthetic"
    )


def _colmap_sparse_directory(root: Path) -> Path | None:
    sparse = root / "sparse" / "0"
    has_binary = (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file()
    has_text = (sparse / "cameras.txt").is_file() and (sparse / "images.txt").is_file()
    return sparse if has_binary or has_text else None


def load_colmap_dataset(
    data_directory: str | Path,
    config: Config | DataConfig,
    splits: Sequence[str] = ("train", "test"),
    *,
    load_points: bool = True,
) -> CameraSplits:
    """Load a PINHOLE COLMAP scene and apply the official every-eighth split."""

    data_config = _data_config(config)
    root = Path(data_directory)
    sparse = _colmap_sparse_directory(root)
    if sparse is None:
        raise FileNotFoundError(f"COLMAP sparse model does not exist below {root}")
    if data_config.rgba_background not in {"black", "white"}:
        raise ValueError("data.rgba_background must be black or white")
    if not 0.0 < data_config.resolution_scale <= 1.0:
        raise ValueError("data.resolution_scale must satisfy 0 < value <= 1")
    requested = tuple(splits)
    unknown = sorted(set(requested) - {"train", "test"})
    if unknown:
        raise ValueError(f"COLMAP datasets have no splits: {', '.join(unknown)}")

    colmap_cameras, colmap_images, points, colors = read_colmap_model(
        sparse, read_points=load_points
    )
    ordered = sorted(colmap_images.values(), key=lambda image: image.name)
    split_records = {
        "train": [image for index, image in enumerate(ordered) if index % 8 != 0],
        "test": [image for index, image in enumerate(ordered) if index % 8 == 0],
    }
    loaded: dict[str, list[Camera]] = {"train": [], "test": []}
    for split in requested:
        for image_record in split_records[split]:
            if image_record.camera_id not in colmap_cameras:
                raise ValueError(
                    f"image {image_record.name} references unknown camera "
                    f"{image_record.camera_id}"
                )
            intrinsics = colmap_cameras[image_record.camera_id]
            if intrinsics.model != "PINHOLE" or len(intrinsics.params) != 4:
                raise ValueError(
                    f"unsupported COLMAP camera model {intrinsics.model!r}; "
                    "the inspected datasets require PINHOLE"
                )
            image_path = root / "images" / image_record.name
            if not image_path.is_file():
                raise FileNotFoundError(f"COLMAP image does not exist: {image_path}")
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened)
                actual_width, actual_height = image.size
                width, height = _scaled_size(
                    actual_width, actual_height, data_config.resolution_scale
                )
                image_tensor = _image_to_tensor(
                    _resize(image, (width, height)), data_config.rgba_background
                )
            metadata_scale_x = actual_width / intrinsics.width
            metadata_scale_y = actual_height / intrinsics.height
            output_scale_x = width / actual_width
            output_scale_y = height / actual_height
            fx, fy, cx, cy = intrinsics.params
            rotation = qvec_to_rotation_cw(image_record.qvec)
            translation = torch.tensor(image_record.tvec, dtype=torch.float32)
            center = -(rotation.transpose(0, 1) @ translation)
            loaded[split].append(
                Camera(
                    rotation_cw=rotation,
                    translation_cw=translation,
                    camera_center_world=center,
                    fx=float(fx * metadata_scale_x * output_scale_x),
                    fy=float(fy * metadata_scale_y * output_scale_y),
                    cx=float(cx * metadata_scale_x * output_scale_x),
                    cy=float(cy * metadata_scale_y * output_scale_y),
                    width=width,
                    height=height,
                    image=image_tensor,
                    image_name=image_record.name,
                )
            )
    if points is not None and points.shape[0] < 4:
        raise ValueError("COLMAP point cloud must contain at least four points")
    center_values = (
        tuple(float(value) for value in points.mean(dim=0))
        if points is not None else (0.0, 0.0, 0.0)
    )
    return CameraSplits(
        train=loaded["train"], val=[], test=loaded["test"],
        scene_center=center_values, format="colmap",
        initial_points=points, initial_colors=colors,
    )


def load_dataset(
    data_directory: str | Path,
    config: Config | DataConfig,
    splits: Sequence[str] = SPLITS,
    *,
    load_points: bool = True,
) -> CameraSplits:
    """Auto-detect and load either supported dataset layout."""

    root = Path(data_directory)
    if (root / "transforms_train.json").is_file():
        return load_nerf_synthetic_dataset(root, config, splits=splits)
    data_config = _data_config(config)
    camera_path = root / data_config.camera_file
    if camera_path.is_file():
        cameras = load_blender_dataset(root, data_config)
        train, test = split_cameras(cameras, data_config.test_every)
        metadata = read_camera_poses_json(camera_path)
        return CameraSplits(
            train=train, val=[], test=test,
            scene_center=tuple(metadata["target"]), format="blender"
        )
    if _colmap_sparse_directory(root) is not None:
        if tuple(splits) == ("val",):
            raise ValueError("COLMAP datasets do not define a val split")
        colmap_splits = tuple(split for split in splits if split != "val")
        return load_colmap_dataset(
            root, data_config, splits=colmap_splits, load_points=load_points
        )
    raise FileNotFoundError(
        f"could not detect dataset format in {root}: expected transforms_train.json "
        f"or {data_config.camera_file}, or sparse/0/{{cameras,images}}.bin/.txt"
    )


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


__all__ = [
    "CameraSplits", "load_blender_dataset", "load_dataset",
    "load_colmap_dataset", "load_nerf_synthetic_dataset", "split_cameras",
]
