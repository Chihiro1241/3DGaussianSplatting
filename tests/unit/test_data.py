from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from gaussian_splatting.config import DataConfig
from gaussian_splatting.data import blender_loader
from gaussian_splatting.data.blender_loader import (
    blender_c2w_to_opencv_w2c,
    read_camera_poses_json,
)
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.data.dataset import load_blender_dataset, split_cameras


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _camera(name: str) -> Camera:
    rotation = torch.eye(3)
    center = torch.zeros(3)
    return Camera(
        rotation_cw=rotation,
        translation_cw=-rotation @ center,
        camera_center_world=center,
        fx=10.0,
        fy=10.0,
        cx=1.0,
        cy=1.0,
        width=2,
        height=2,
        image=torch.zeros(3, 2, 2),
        image_name=name,
    )


def test_camera_validates_pose_consistency() -> None:
    with pytest.raises(AssertionError):
        Camera(
            rotation_cw=torch.eye(3),
            translation_cw=torch.ones(3),
            camera_center_world=torch.zeros(3),
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
            width=2,
            height=2,
        )


def test_read_camera_poses_json_validates_and_sorts_real_frames() -> None:
    metadata = read_camera_poses_json(PROJECT_ROOT / "data" / "camera_poses_blender.json")

    assert len(metadata["frames"]) == 100
    assert [frame["index"] for frame in metadata["frames"]] == list(range(100))
    assert [frame["file_path"] for frame in metadata["frames"]] == [
        f"view_{index:03d}.png" for index in range(100)
    ]


@pytest.mark.data
@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "camera_poses_blender.json").is_file(),
    reason="ignored local Blender dataset is not present",
)
def test_real_blender_dataset_loads_all_one_hundred_800px_images() -> None:
    config = DataConfig(
        camera_file="camera_poses_blender.json",
        resolution_scale=1.0,
        rgba_background="black",
        test_every=8,
    )

    cameras = load_blender_dataset(PROJECT_ROOT / "data", config)

    assert len(cameras) == 100
    assert [camera.image_name for camera in cameras] == [
        f"view_{index:03d}.png" for index in range(100)
    ]
    assert all((camera.width, camera.height) == (800, 800) for camera in cameras)
    assert all(
        camera.image is not None and camera.image.shape == (3, 800, 800)
        for camera in cameras
    )


def test_blender_conversion_preserves_first_camera_center() -> None:
    metadata = read_camera_poses_json(PROJECT_ROOT / "data" / "camera_poses_blender.json")
    frame = metadata["frames"][0]
    c2w = torch.tensor(frame["transform_matrix_blender_c2w"], dtype=torch.float64)

    rotation, translation, center = blender_c2w_to_opencv_w2c(c2w)

    torch.testing.assert_close(center, c2w[:3, 3])
    torch.testing.assert_close(translation, -rotation @ center, rtol=1e-5, atol=1e-6)


def test_load_blender_dataset_converts_rgba_to_chw_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blender_loader, "EXPECTED_FRAME_COUNT", 1)
    metadata = {
        "coordinate_system": "Blender: x-y floor plane, z-up",
        "camera_transform": "camera-to-world",
        "resolution_x": 2,
        "resolution_y": 2,
        "lens_mm": 35.0,
        "sensor_width_mm": 36.0,
        "target": [0.0, 0.0, 0.0],
        "camera_radius": 1.0,
        "frames": [
            {
                "index": 0,
                "file_path": "view_000.png",
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "transform_matrix_blender_c2w": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (tmp_path / "camera.json").write_text(json.dumps(metadata), encoding="utf-8")
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(tmp_path / "view_000.png")
    config = DataConfig(
        camera_file="camera.json",
        resolution_scale=1.0,
        rgba_background="black",
        test_every=8,
    )

    cameras = load_blender_dataset(tmp_path, config)

    assert len(cameras) == 1
    assert cameras[0].image is not None
    assert cameras[0].image.shape == (3, 2, 2)
    torch.testing.assert_close(
        cameras[0].image[:, 0, 0], torch.tensor([128.0 / 255.0, 0.0, 0.0])
    )


def test_load_blender_dataset_scales_image_and_intrinsics_like_official_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blender_loader, "EXPECTED_FRAME_COUNT", 1)
    metadata = {
        "coordinate_system": "Blender: x-y floor plane, z-up",
        "camera_transform": "camera-to-world",
        "resolution_x": 4,
        "resolution_y": 4,
        "lens_mm": 35.0,
        "sensor_width_mm": 36.0,
        "target": [0.0, 0.0, 0.0],
        "camera_radius": 1.0,
        "frames": [
            {
                "index": 0,
                "file_path": "view_000.png",
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "transform_matrix_blender_c2w": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (tmp_path / "camera.json").write_text(json.dumps(metadata), encoding="utf-8")
    Image.new("RGB", (4, 4), (64, 128, 255)).save(tmp_path / "view_000.png")
    config = DataConfig(
        camera_file="camera.json",
        resolution_scale=0.5,
        rgba_background="black",
        test_every=8,
    )

    camera = load_blender_dataset(tmp_path, config)[0]

    assert (camera.width, camera.height) == (2, 2)
    assert camera.image is not None and camera.image.shape == (3, 2, 2)
    assert camera.fx == pytest.approx((35.0 / 36.0 * 4) * 0.5)
    assert camera.fy == pytest.approx((35.0 / 36.0 * 4) * 0.5)
    assert camera.cx == pytest.approx(1.0)
    assert camera.cy == pytest.approx(1.0)


def test_split_cameras_sorts_by_image_name_before_periodic_split() -> None:
    train, evaluation = split_cameras(
        [_camera("view_002.png"), _camera("view_000.png"), _camera("view_001.png")],
        test_every=2,
    )

    assert [camera.image_name for camera in train] == ["view_001.png"]
    assert [camera.image_name for camera in evaluation] == [
        "view_000.png",
        "view_002.png",
    ]
