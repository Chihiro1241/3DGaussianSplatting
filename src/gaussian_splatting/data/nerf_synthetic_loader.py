"""Reader for the canonical NeRF Synthetic ``transforms_*.json`` format."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gaussian_splatting.data.blender_loader import _matrix4, _number


SPLITS = ("train", "val", "test")


def read_nerf_synthetic_json(path: str | Path) -> dict[str, Any]:
    """Read and validate one NeRF Synthetic transform file."""

    json_path = Path(path)
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {json_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{json_path} root must be an object")
    if "camera_angle_x" not in value or "frames" not in value:
        raise ValueError(f"{json_path} must contain camera_angle_x and frames")
    angle = _number(value["camera_angle_x"], "camera_angle_x", positive=True)
    if angle >= math.pi:
        raise ValueError("camera_angle_x must be less than pi radians")
    frames = value["frames"]
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    validated: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        prefix = f"frames[{index}]"
        if not isinstance(frame, dict):
            raise ValueError(f"{prefix} must be an object")
        if "file_path" not in frame or "transform_matrix" not in frame:
            raise ValueError(f"{prefix} must contain file_path and transform_matrix")
        if not isinstance(frame["file_path"], str) or not frame["file_path"]:
            raise ValueError(f"{prefix}.file_path must be a non-empty string")
        validated.append(
            {
                **frame,
                "transform_matrix": _matrix4(
                    frame["transform_matrix"], f"{prefix}.transform_matrix"
                ),
            }
        )
    value["camera_angle_x"] = angle
    value["frames"] = validated
    return value


def resolve_nerf_image_path(root: Path, file_path: str) -> Path:
    """Resolve extension-less paths used by the published dataset."""

    relative = file_path[2:] if file_path.startswith("./") else file_path
    path = root / relative
    if path.is_file():
        return path
    png_path = path.with_suffix(".png") if not path.suffix else path
    if png_path.is_file():
        return png_path
    raise FileNotFoundError(f"NeRF Synthetic image does not exist: {png_path}")


__all__ = ["SPLITS", "read_nerf_synthetic_json", "resolve_nerf_image_path"]
