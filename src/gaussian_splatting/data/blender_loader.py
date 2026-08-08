"""Blender camera-pose JSON loading and coordinate conversion."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


EXPECTED_COORDINATE_SYSTEM = "Blender: x-y floor plane, z-up"
EXPECTED_CAMERA_TRANSFORM = "camera-to-world"
EXPECTED_FRAME_COUNT = 100
_TOP_LEVEL_KEYS = {
    "coordinate_system",
    "camera_transform",
    "resolution_x",
    "resolution_y",
    "lens_mm",
    "sensor_width_mm",
    "target",
    "camera_radius",
    "frames",
}
_FRAME_KEYS = {
    "index",
    "file_path",
    "azimuth_deg",
    "elevation_deg",
    "transform_matrix_blender_c2w",
}


def _require_keys(value: dict[str, Any], keys: set[str], path: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite")
    if positive and converted <= 0.0:
        raise ValueError(f"{path} must be positive")
    return converted


def _integer(value: object, path: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{path} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def _vector(value: object, length: int, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{path} must be a list of length {length}")
    return [_number(element, f"{path}[{index}]") for index, element in enumerate(value)]


def _matrix4(value: object, path: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path} must have shape (4, 4)")
    matrix = [_vector(row, 4, f"{path}[{index}]") for index, row in enumerate(value)]
    if matrix[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"{path} must be a homogeneous transform with last row [0, 0, 0, 1]")
    tensor = torch.tensor(matrix, dtype=torch.float64)
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{path} contains NaN or Inf")
    try:
        torch.linalg.inv(tensor)
    except RuntimeError as error:
        raise ValueError(f"{path} must be invertible") from error
    return matrix


def read_camera_poses_json(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate ``camera_poses_blender.json``.

    Frames in the returned dictionary are sorted by their contiguous zero-based
    ``index`` values. Image existence and image metadata are checked by
    :func:`gaussian_splatting.data.dataset.load_blender_dataset`.
    """

    json_path = Path(path)
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {json_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("camera pose JSON root must be an object")
    _require_keys(value, _TOP_LEVEL_KEYS, "camera pose JSON")

    if value["coordinate_system"] != EXPECTED_COORDINATE_SYSTEM:
        raise ValueError(
            f"coordinate_system must be exactly {EXPECTED_COORDINATE_SYSTEM!r}"
        )
    if value["camera_transform"] != EXPECTED_CAMERA_TRANSFORM:
        raise ValueError(f"camera_transform must be exactly {EXPECTED_CAMERA_TRANSFORM!r}")
    _integer(value["resolution_x"], "resolution_x", positive=True)
    _integer(value["resolution_y"], "resolution_y", positive=True)
    _number(value["lens_mm"], "lens_mm", positive=True)
    _number(value["sensor_width_mm"], "sensor_width_mm", positive=True)
    value["target"] = _vector(value["target"], 3, "target")
    _number(value["camera_radius"], "camera_radius", positive=True)

    frames = value["frames"]
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    if len(frames) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"frames must contain exactly {EXPECTED_FRAME_COUNT} entries, got {len(frames)}"
        )

    validated_frames: list[dict[str, Any]] = []
    for position, frame in enumerate(frames):
        path_prefix = f"frames[{position}]"
        if not isinstance(frame, dict):
            raise ValueError(f"{path_prefix} must be an object")
        _require_keys(frame, _FRAME_KEYS, path_prefix)
        _integer(frame["index"], f"{path_prefix}.index")
        if not isinstance(frame["file_path"], str) or not frame["file_path"]:
            raise ValueError(f"{path_prefix}.file_path must be a non-empty string")
        _number(frame["azimuth_deg"], f"{path_prefix}.azimuth_deg")
        _number(frame["elevation_deg"], f"{path_prefix}.elevation_deg")
        frame["transform_matrix_blender_c2w"] = _matrix4(
            frame["transform_matrix_blender_c2w"],
            f"{path_prefix}.transform_matrix_blender_c2w",
        )
        validated_frames.append(frame)

    indices = [frame["index"] for frame in validated_frames]
    if len(set(indices)) != len(indices):
        raise ValueError("frame index values must not contain duplicates")
    expected_indices = list(range(len(validated_frames)))
    if sorted(indices) != expected_indices:
        raise ValueError(
            "frame index values must be contiguous and start at zero "
            f"(expected 0 through {len(validated_frames) - 1})"
        )
    file_paths = [frame["file_path"] for frame in validated_frames]
    if len(set(file_paths)) != len(file_paths):
        raise ValueError("frame file_path values must not contain duplicates")

    value["frames"] = sorted(validated_frames, key=lambda frame: frame["index"])
    return value


def blender_c2w_to_opencv_w2c(c2w_blender: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a Blender camera-to-world transform to OpenCV world-to-camera.

    Args:
        c2w_blender: Floating-point tensor of shape ``(4, 4)``.

    Returns:
        ``(rotation_cw, translation_cw, camera_center_world)`` with shapes
        ``(3, 3)``, ``(3,)``, and ``(3,)`` respectively.
    """

    if not isinstance(c2w_blender, Tensor):
        raise TypeError("c2w_blender must be a torch.Tensor")
    if tuple(c2w_blender.shape) != (4, 4):
        raise ValueError(f"c2w_blender must have shape (4, 4), got {tuple(c2w_blender.shape)}")
    if c2w_blender.dtype not in (torch.float32, torch.float64):
        raise TypeError("c2w_blender must have dtype torch.float32 or torch.float64")
    if not torch.isfinite(c2w_blender).all().item():
        raise ValueError("c2w_blender contains NaN or Inf")

    expected_last_row = c2w_blender.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(c2w_blender[3], expected_last_row, rtol=0.0, atol=1e-7):
        raise ValueError("c2w_blender must be a homogeneous transform")
    basis_change = torch.diag(c2w_blender.new_tensor([1.0, -1.0, -1.0, 1.0]))
    c2w_opencv = c2w_blender @ basis_change
    try:
        w2c_opencv = torch.linalg.inv(c2w_opencv)
    except RuntimeError as error:
        raise ValueError("c2w_blender must be invertible") from error

    rotation_cw = w2c_opencv[:3, :3]
    translation_cw = w2c_opencv[:3, 3]
    camera_center_world = c2w_opencv[:3, 3]
    return rotation_cw, translation_cw, camera_center_world


__all__ = [
    "EXPECTED_CAMERA_TRANSFORM",
    "EXPECTED_COORDINATE_SYSTEM",
    "EXPECTED_FRAME_COUNT",
    "blender_c2w_to_opencv_w2c",
    "read_camera_poses_json",
]
