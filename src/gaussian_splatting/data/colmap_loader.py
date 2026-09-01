"""Minimal COLMAP sparse-model reader for the real datasets used by 3DGS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import torch
from torch import Tensor


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_exact(stream: object, size: int, path: Path) -> bytes:
    value = stream.read(size)  # type: ignore[attr-defined]
    if len(value) != size:
        raise ValueError(f"unexpected end of COLMAP file: {path}")
    return value


def _unpack(stream: object, fmt: str, path: Path) -> tuple[object, ...]:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, _read_exact(stream, size, path))


def _read_c_string(stream: object, path: Path) -> str:
    value = bytearray()
    while True:
        byte = _read_exact(stream, 1, path)
        if byte == b"\0":
            break
        value.extend(byte)
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 image name in {path}") from error


def read_cameras_binary(path: str | Path) -> dict[int, ColmapCamera]:
    binary_path = Path(path)
    cameras: dict[int, ColmapCamera] = {}
    with binary_path.open("rb") as stream:
        count = int(_unpack(stream, "Q", binary_path)[0])
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(stream, "iiQQ", binary_path)
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            model, parameter_count = _CAMERA_MODELS[int(model_id)]
            params = tuple(
                float(value)
                for value in _unpack(stream, "d" * parameter_count, binary_path)
            )
            camera = ColmapCamera(int(camera_id), model, int(width), int(height), params)
            if camera.camera_id in cameras:
                raise ValueError(f"duplicate COLMAP camera id {camera.camera_id}")
            cameras[camera.camera_id] = camera
    return cameras


def read_images_binary(path: str | Path) -> dict[int, ColmapImage]:
    binary_path = Path(path)
    images: dict[int, ColmapImage] = {}
    with binary_path.open("rb") as stream:
        count = int(_unpack(stream, "Q", binary_path)[0])
        for _ in range(count):
            values = _unpack(stream, "idddddddi", binary_path)
            image_id = int(values[0])
            image = ColmapImage(
                image_id=image_id,
                qvec=tuple(float(value) for value in values[1:5]),  # type: ignore[arg-type]
                tvec=tuple(float(value) for value in values[5:8]),  # type: ignore[arg-type]
                camera_id=int(values[8]),
                name=_read_c_string(stream, binary_path),
            )
            point_count = int(_unpack(stream, "Q", binary_path)[0])
            _read_exact(stream, point_count * 24, binary_path)
            if image_id in images:
                raise ValueError(f"duplicate COLMAP image id {image_id}")
            images[image_id] = image
    return images


def read_points3d_binary(path: str | Path) -> tuple[Tensor, Tensor]:
    binary_path = Path(path)
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    with binary_path.open("rb") as stream:
        count = int(_unpack(stream, "Q", binary_path)[0])
        for _ in range(count):
            values = _unpack(stream, "QdddBBBd", binary_path)
            xyz.append((float(values[1]), float(values[2]), float(values[3])))
            rgb.append((int(values[4]), int(values[5]), int(values[6])))
            track_length = int(_unpack(stream, "Q", binary_path)[0])
            _read_exact(stream, track_length * 8, binary_path)
    return torch.tensor(xyz, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32) / 255.0


def _data_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def read_cameras_text(path: str | Path) -> dict[int, ColmapCamera]:
    text_path = Path(path)
    cameras: dict[int, ColmapCamera] = {}
    for line in _data_lines(text_path):
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = ColmapCamera(
            camera_id, fields[1], int(fields[2]), int(fields[3]),
            tuple(float(value) for value in fields[4:]),
        )
    return cameras


def read_images_text(path: str | Path) -> dict[int, ColmapImage]:
    text_path = Path(path)
    lines = _data_lines(text_path)
    if len(lines) % 2:
        raise ValueError(f"COLMAP images text must contain two lines per image: {text_path}")
    images: dict[int, ColmapImage] = {}
    for line in lines[::2]:
        fields = line.split()
        image_id = int(fields[0])
        images[image_id] = ColmapImage(
            image_id,
            tuple(float(value) for value in fields[1:5]),  # type: ignore[arg-type]
            tuple(float(value) for value in fields[5:8]),  # type: ignore[arg-type]
            int(fields[8]),
            " ".join(fields[9:]),
        )
    return images


def read_points3d_text(path: str | Path) -> tuple[Tensor, Tensor]:
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    for line in _data_lines(Path(path)):
        fields = line.split()
        xyz.append(tuple(float(value) for value in fields[1:4]))  # type: ignore[arg-type]
        rgb.append(tuple(int(value) for value in fields[4:7]))  # type: ignore[arg-type]
    return torch.tensor(xyz, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32) / 255.0


def read_colmap_model(
    sparse_directory: str | Path, *, read_points: bool = True
) -> tuple[dict[int, ColmapCamera], dict[int, ColmapImage], Tensor | None, Tensor | None]:
    """Read a sparse model, preferring binary files when both forms exist."""

    root = Path(sparse_directory)
    binary = (root / "cameras.bin").is_file() and (root / "images.bin").is_file()
    if binary:
        cameras = read_cameras_binary(root / "cameras.bin")
        images = read_images_binary(root / "images.bin")
        points_path = root / "points3D.bin"
        points, colors = read_points3d_binary(points_path) if read_points else (None, None)
    else:
        cameras = read_cameras_text(root / "cameras.txt")
        images = read_images_text(root / "images.txt")
        points_path = root / "points3D.txt"
        points, colors = read_points3d_text(points_path) if read_points else (None, None)
    return cameras, images, points, colors


def qvec_to_rotation_cw(qvec: tuple[float, float, float, float]) -> Tensor:
    """Convert COLMAP's Hamilton world-to-camera quaternion to a matrix."""

    quaternion = torch.tensor(qvec, dtype=torch.float64)
    norm = torch.linalg.vector_norm(quaternion)
    if not torch.isfinite(quaternion).all().item() or norm <= 0.0:
        raise ValueError("COLMAP quaternion must be finite and non-zero")
    qw, qx, qy, qz = quaternion / norm
    return torch.stack((
        torch.stack((1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy)),
        torch.stack((2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx)),
        torch.stack((2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy)),
    )).to(torch.float32)


__all__ = [
    "ColmapCamera", "ColmapImage", "qvec_to_rotation_cw", "read_cameras_binary",
    "read_cameras_text", "read_colmap_model", "read_images_binary",
    "read_images_text", "read_points3d_binary", "read_points3d_text",
]
