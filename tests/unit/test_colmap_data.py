from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image
import pytest
import torch

from gaussian_splatting.config import DataConfig, load_config
from gaussian_splatting.data import load_colmap_dataset, load_dataset
from gaussian_splatting.data.colmap_loader import qvec_to_rotation_cw
from gaussian_splatting.model.initialization import initialize_gaussian_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_binary_scene(root: Path) -> None:
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQdddd", 1, 1, 1, 8, 4, 8.0, 6.0, 4.0, 2.0))
    with (sparse / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 10))
        for index in range(10):
            stream.write(struct.pack(
                "<idddddddi", index + 1, 1.0, 0.0, 0.0, 0.0,
                1.0, 2.0, 3.0, 1,
            ))
            stream.write(f"frame_{index:02d}.png".encode() + b"\0")
            stream.write(struct.pack("<Q", 0))
            Image.new("RGBA", (4, 2), (255, 0, 0, 128)).save(
                images / f"frame_{index:02d}.png"
            )
    points = [
        (1, (0.0, 0.0, 0.0), (255, 0, 0)),
        (2, (1.0, 0.0, 0.0), (0, 255, 0)),
        (3, (0.0, 1.0, 0.0), (0, 0, 255)),
        (4, (0.0, 0.0, 1.0), (255, 255, 255)),
    ]
    with (sparse / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb in points:
            stream.write(struct.pack("<QdddBBBd", point_id, *xyz, *rgb, 0.1))
            stream.write(struct.pack("<Q", 0))


def _config(scale: float = 0.5) -> DataConfig:
    return DataConfig("camera_poses_blender.json", scale, "black", 8)


def test_colmap_binary_pose_intrinsics_split_and_point_cloud(tmp_path: Path) -> None:
    _write_binary_scene(tmp_path)

    dataset = load_dataset(tmp_path, _config())

    assert dataset.format == "colmap"
    assert dataset.val == []
    assert [camera.image_name for camera in dataset.test] == [
        "frame_00.png", "frame_08.png"
    ]
    assert [camera.image_name for camera in dataset.train] == [
        f"frame_{index:02d}.png" for index in (1, 2, 3, 4, 5, 6, 7, 9)
    ]
    camera = dataset.test[0]
    assert (camera.width, camera.height) == (2, 1)
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == pytest.approx(
        (2.0, 1.5, 1.0, 0.5)
    )
    torch.testing.assert_close(camera.rotation_cw, torch.eye(3))
    torch.testing.assert_close(camera.translation_cw, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(camera.camera_center_world, torch.tensor([-1.0, -2.0, -3.0]))
    assert dataset.initial_points is not None and dataset.initial_colors is not None
    torch.testing.assert_close(dataset.initial_points[1], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(dataset.initial_colors[1], torch.tensor([0.0, 1.0, 0.0]))
    model = initialize_gaussian_model(
        dataset.initial_points, dataset.initial_colors,
        load_config(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    assert model.num_gaussians == 4


def test_colmap_quaternion_is_world_to_camera_rotation() -> None:
    half = 2.0 ** -0.5
    rotation = qvec_to_rotation_cw((half, 0.0, 0.0, half))

    torch.testing.assert_close(
        rotation,
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_colmap_can_skip_point_cloud_for_rendering(tmp_path: Path) -> None:
    _write_binary_scene(tmp_path)

    dataset = load_colmap_dataset(tmp_path, _config(), splits=("test",), load_points=False)

    assert dataset.train == [] and len(dataset.test) == 2
    assert dataset.initial_points is None and dataset.initial_colors is None


@pytest.mark.data
@pytest.mark.parametrize(
    "relative_scene",
    [
        "data/mipnerf360/bicycle",
        "data/tandt_db/tandt/truck",
        "data/tandt_db/db/drjohnson",
    ],
)
def test_real_colmap_scene_loads_test_split_and_points(relative_scene: str) -> None:
    scene = PROJECT_ROOT / relative_scene
    if not scene.is_dir():
        pytest.skip(f"ignored local dataset is not present: {scene}")

    dataset = load_colmap_dataset(
        scene, _config(scale=0.01), splits=("test",), load_points=True
    )

    assert dataset.format == "colmap"
    assert dataset.train == [] and dataset.val == [] and dataset.test
    assert dataset.initial_points is not None and dataset.initial_points.shape[1] == 3
    assert dataset.initial_colors is not None
    assert torch.all((dataset.initial_colors >= 0.0) & (dataset.initial_colors <= 1.0))
