from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image
import pytest
import torch

from gaussian_splatting.config import DataConfig
from gaussian_splatting.data import load_dataset


def _write_scene(root: Path) -> None:
    poses = {
        "train": ([1.0, 2.0, 3.0], (255, 0, 0, 128)),
        "val": ([4.0, 5.0, 6.0], (0, 255, 0, 128)),
        "test": ([7.0, 8.0, 9.0], (0, 0, 255, 128)),
    }
    for split, (center, color) in poses.items():
        image_dir = root / split
        image_dir.mkdir()
        Image.new("RGBA", (4, 2), color).save(image_dir / "r_0.png")
        matrix = [
            [1.0, 0.0, 0.0, center[0]],
            [0.0, 1.0, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
        (root / f"transforms_{split}.json").write_text(
            json.dumps({
                "camera_angle_x": math.pi / 2.0,
                "frames": [{"file_path": f"./{split}/r_0", "transform_matrix": matrix}],
            }),
            encoding="utf-8",
        )


def _config(background: str = "black") -> DataConfig:
    return DataConfig(
        camera_file="camera_poses_blender.json",
        resolution_scale=1.0,
        rgba_background=background,
        test_every=8,
    )


def test_auto_detects_nerf_synthetic_and_preserves_source_splits(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    dataset = load_dataset(tmp_path, _config())

    assert dataset.format == "nerf_synthetic"
    assert [camera.image_name for camera in dataset.train] == ["train/r_0.png"]
    assert [camera.image_name for camera in dataset.val] == ["val/r_0.png"]
    assert [camera.image_name for camera in dataset.test] == ["test/r_0.png"]


def test_nerf_intrinsics_pose_conversion_and_black_alpha_composite(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    camera = load_dataset(tmp_path, _config("black")).train[0]

    assert camera.fx == pytest.approx(2.0)
    assert camera.fy == pytest.approx(2.0)
    assert (camera.cx, camera.cy) == pytest.approx((2.0, 1.0))
    torch.testing.assert_close(camera.camera_center_world, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(camera.rotation_cw, torch.diag(torch.tensor([1.0, -1.0, -1.0])))
    torch.testing.assert_close(camera.translation_cw, torch.tensor([-1.0, 2.0, 3.0]))
    torch.testing.assert_close(
        camera.image[:, 0, 0], torch.tensor([128.0 / 255.0, 0.0, 0.0])
    )


def test_nerf_rgba_composites_onto_white_background(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    pixel = load_dataset(tmp_path, _config("white")).train[0].image[:, 0, 0]

    alpha = 128.0 / 255.0
    torch.testing.assert_close(pixel, torch.tensor([1.0, 1.0 - alpha, 1.0 - alpha]))


def test_nerf_resolution_scale_scales_images_and_intrinsics(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    config = DataConfig("unused.json", 0.5, "black", 8)

    camera = load_dataset(tmp_path, config).train[0]

    assert (camera.width, camera.height) == (2, 1)
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == pytest.approx(
        (1.0, 1.0, 1.0, 0.5)
    )
