from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.snapshot import (
    GaussianSnapshotWriter,
    combined_bounds,
    discover_snapshot_frames,
    parse_iteration_list,
    read_snapshot,
    read_snapshot_index,
    snapshot_filename,
    subsample_indices,
    write_snapshot_npz,
)
from gaussian_splatting.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[2]


def _tiny_config() -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(
            config.training,
            iterations=4,
            log_interval=4,
            evaluation_interval=4,
            checkpoint_interval=4,
        ),
        features=replace(
            config.features,
            adaptive_density_control=False,
            opacity_reset=False,
            progressive_sh_degree=False,
            resolution_warmup=False,
        ),
        output=replace(config.output, save_rendered_images=False),
    )


def _tiny_model() -> GaussianModel:
    return GaussianModel(
        means_world=torch.tensor([[-0.10, 0.00, 2.0], [0.15, 0.08, 2.3]]),
        raw_quaternions=torch.tensor(
            [[1.0, 0.10, 0.05, 0.00], [1.0, 0.00, 0.20, -0.10]]
        ),
        raw_scales=torch.log(torch.tensor([[0.12, 0.08, 0.10], [0.10, 0.14, 0.08]])),
        raw_opacities=torch.logit(torch.tensor([[0.35], [0.45]])),
        sh_dc=torch.tensor([[[0.10, -0.05, 0.00]], [[-0.08, 0.02, 0.12]]]),
        sh_rest=torch.zeros((2, 15, 3)),
    )


def _tiny_camera() -> Camera:
    image = torch.full((3, 7, 7), 0.25)
    center = torch.zeros(3)
    return Camera(
        rotation_cw=torch.eye(3),
        translation_cw=-center,
        camera_center_world=center,
        fx=8.0,
        fy=8.0,
        cx=3.0,
        cy=3.0,
        width=7,
        height=7,
        image=image,
        image_name="tiny.png",
    )


def _trainer(tmp_path: Path, writer: GaussianSnapshotWriter | None) -> Trainer:
    config = _tiny_config()
    model = _tiny_model()
    camera = _tiny_camera()
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    return Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[camera],
        evaluation_cameras=[],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path,
        camera_order=[0],
        snapshot_writer=writer,
    )


# ---------------------------------------------------------------- primitives


def test_snapshot_filename_is_zero_padded_and_rejects_negative() -> None:
    assert snapshot_filename(300) == "iteration_00000300.npz"
    with pytest.raises(ValueError):
        snapshot_filename(-1)


def test_subsample_indices_thins_only_when_needed() -> None:
    assert subsample_indices(10, None) is None
    assert subsample_indices(10, 20) is None
    selected = subsample_indices(1000, 50)
    assert selected is not None
    assert len(selected) == 50
    assert selected[0] == 0 and selected[-1] == 999
    assert np.all(np.diff(selected) > 0)


def test_subsample_indices_is_deterministic() -> None:
    first = subsample_indices(5000, 100)
    second = subsample_indices(5000, 100)
    assert np.array_equal(first, second)


def test_parse_iteration_list_sorts_and_deduplicates() -> None:
    assert parse_iteration_list(None) == ()
    assert parse_iteration_list("") == ()
    assert parse_iteration_list("500, 0,100 ,500") == (0, 100, 500)
    with pytest.raises(ValueError):
        parse_iteration_list("-1")


# ------------------------------------------------------------------- writing


def test_write_snapshot_npz_roundtrips(tmp_path: Path) -> None:
    means = np.arange(30, dtype=np.float32).reshape(10, 3)
    opacities = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    entry = write_snapshot_npz(
        tmp_path / "snap.npz",
        means=means,
        opacities=opacities,
        iteration=700,
        frame=3,
    )

    assert entry["num_gaussians"] == 10
    assert entry["stored_points"] == 10
    assert entry["iteration"] == 700 and entry["frame"] == 3

    payload = read_snapshot(tmp_path / "snap.npz")
    assert np.array_equal(payload["means"], means)
    assert np.allclose(payload["opacities"], opacities)
    assert payload["iteration"] == 700
    assert payload["frame"] == 3
    assert payload["num_gaussians"] == 10


def test_write_snapshot_npz_reports_true_count_when_thinned(tmp_path: Path) -> None:
    means = np.zeros((1000, 3), dtype=np.float32)
    opacities = np.zeros(1000, dtype=np.float32)
    entry = write_snapshot_npz(
        tmp_path / "snap.npz",
        means=means,
        opacities=opacities,
        iteration=0,
        frame=1,
        max_points=64,
    )

    assert entry["num_gaussians"] == 1000, "count must survive thinning"
    assert entry["stored_points"] == 64
    assert read_snapshot(tmp_path / "snap.npz")["means"].shape == (64, 3)


def test_write_snapshot_npz_rejects_mismatched_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_snapshot_npz(
            tmp_path / "bad.npz",
            means=np.zeros((4, 3), dtype=np.float32),
            opacities=np.zeros(5, dtype=np.float32),
            iteration=0,
            frame=1,
        )


def test_robust_bounds_ignore_far_outliers(tmp_path: Path) -> None:
    means = np.zeros((1000, 3), dtype=np.float32)
    means[-1] = 10_000.0
    entry = write_snapshot_npz(
        tmp_path / "snap.npz",
        means=means,
        opacities=np.zeros(1000, dtype=np.float32),
        iteration=0,
        frame=1,
    )

    assert entry["bounds"][1][0] == pytest.approx(10_000.0)
    assert entry["robust_bounds"][1][0] < 1.0


def test_combined_bounds_selects_requested_key() -> None:
    entries = [
        {
            "bounds": [[-100.0, -1.0, -1.0], [100.0, 1.0, 1.0]],
            "robust_bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        }
    ]
    wide = combined_bounds(entries, margin=0.0)
    narrow = combined_bounds(entries, margin=0.0, key="robust_bounds")
    assert wide[0][0] == pytest.approx(-100.0)
    assert narrow[0][0] == pytest.approx(-1.0)


def test_combined_bounds_falls_back_when_key_missing() -> None:
    entries = [{"bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}]
    assert combined_bounds(entries, key="robust_bounds", margin=0.0) is not None


# -------------------------------------------------------------------- writer


def test_writer_cadence_and_extra_iterations(tmp_path: Path) -> None:
    writer = GaussianSnapshotWriter(
        tmp_path / "snapshots", interval=100, extra_iterations=(7,)
    )
    assert writer.should_record(0)
    assert writer.should_record(100)
    assert writer.should_record(7)
    assert not writer.should_record(101)


def test_writer_with_zero_interval_records_only_extras(tmp_path: Path) -> None:
    writer = GaussianSnapshotWriter(
        tmp_path / "snapshots", interval=0, extra_iterations=(5,)
    )
    assert writer.should_record(5)
    assert not writer.should_record(0)
    assert not writer.should_record(10)


def test_writer_records_model_opacity_as_sigmoid(tmp_path: Path) -> None:
    model = _tiny_model()
    writer = GaussianSnapshotWriter(tmp_path / "snapshots", interval=1, max_points=None)
    path = writer.record(model, 12)

    payload = read_snapshot(path)
    expected = torch.sigmoid(model.raw_opacities).reshape(-1).detach().numpy()
    assert np.allclose(payload["opacities"], expected)
    assert np.allclose(payload["means"], model.means_world.detach().numpy())


def test_writer_rewrites_index_after_every_record(tmp_path: Path) -> None:
    directory = tmp_path / "snapshots"
    writer = GaussianSnapshotWriter(directory, interval=1, max_points=None)
    model = _tiny_model()
    writer.record(model, 0)
    assert len(read_snapshot_index(directory)) == 1
    writer.record(model, 10)

    entries = read_snapshot_index(directory)
    assert [entry["iteration"] for entry in entries] == [0, 10]
    assert writer.recorded_iterations == (0, 10)


def test_writer_adopts_snapshots_from_an_interrupted_run(tmp_path: Path) -> None:
    directory = tmp_path / "snapshots"
    model = _tiny_model()
    GaussianSnapshotWriter(directory, interval=1, max_points=None).record(model, 0)

    resumed = GaussianSnapshotWriter(directory, interval=1, max_points=None)
    resumed.record(model, 50)

    assert resumed.recorded_iterations == (0, 50)
    assert [entry["iteration"] for entry in read_snapshot_index(directory)] == [0, 50]


def test_writer_drops_index_entries_whose_file_vanished(tmp_path: Path) -> None:
    directory = tmp_path / "snapshots"
    model = _tiny_model()
    writer = GaussianSnapshotWriter(directory, interval=1, max_points=None)
    writer.record(model, 0)
    (directory / snapshot_filename(0)).unlink()

    resumed = GaussianSnapshotWriter(directory, interval=1, max_points=None)
    assert resumed.recorded_iterations == ()


def test_writer_rejects_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        GaussianSnapshotWriter(tmp_path / "a", interval=-1)
    with pytest.raises(ValueError):
        GaussianSnapshotWriter(tmp_path / "b", interval=1, frame=0)
    with pytest.raises(ValueError):
        GaussianSnapshotWriter(tmp_path / "c", interval=1, max_points=0)


# ------------------------------------------------------------------ discovery


def test_discover_snapshot_frames_handles_both_layouts(tmp_path: Path) -> None:
    model = _tiny_model()
    single = tmp_path / "run"
    GaussianSnapshotWriter(single / "snapshots", interval=1).record(model, 0)
    assert discover_snapshot_frames(single) == {1: single / "snapshots"}

    four_d = tmp_path / "run4d"
    for frame in (1, 2):
        GaussianSnapshotWriter(
            four_d / f"frame_{frame:04d}" / "snapshots", interval=1, frame=frame
        ).record(model, 0)
    discovered = discover_snapshot_frames(four_d)
    assert sorted(discovered) == [1, 2]
    assert discovered[2].parent.name == "frame_0002"


def test_discover_snapshot_frames_reports_an_empty_run(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        discover_snapshot_frames(tmp_path / "empty")


# -------------------------------------------------------------- trainer hook


def test_trainer_without_writer_creates_no_snapshots(tmp_path: Path) -> None:
    _trainer(tmp_path, None).train()
    assert not (tmp_path / "snapshots").exists()


def test_trainer_records_initial_state_and_final_iteration(tmp_path: Path) -> None:
    torch.manual_seed(0)
    writer = GaussianSnapshotWriter(
        tmp_path / "snapshots", interval=2, max_points=None
    )
    trainer = _trainer(tmp_path, writer)
    trainer.train()

    recorded = writer.recorded_iterations
    assert recorded[0] == 0, "the pre-training state must open the animation"
    assert recorded[-1] == trainer.config.training.iterations
    assert set(recorded) == {0, 2, 4}


def test_trainer_forces_a_snapshot_at_the_final_iteration(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # The cadence does not divide the final iteration 4.
    writer = GaussianSnapshotWriter(
        tmp_path / "snapshots", interval=3, max_points=None
    )
    _trainer(tmp_path, writer).train()

    assert 4 in writer.recorded_iterations


def test_trainer_snapshots_track_the_moving_centres(tmp_path: Path) -> None:
    torch.manual_seed(0)
    directory = tmp_path / "snapshots"
    writer = GaussianSnapshotWriter(directory, interval=1, max_points=None)
    _trainer(tmp_path, writer).train()

    entries = read_snapshot_index(directory)
    first = read_snapshot(directory / entries[0]["file"])
    last = read_snapshot(directory / entries[-1]["file"])
    assert first["means"].shape == last["means"].shape
    assert not np.allclose(first["means"], last["means"])


def test_trainer_rejects_a_non_writer(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        _trainer(tmp_path, object())  # type: ignore[arg-type]


def test_index_json_is_valid_and_sorted(tmp_path: Path) -> None:
    torch.manual_seed(0)
    directory = tmp_path / "snapshots"
    writer = GaussianSnapshotWriter(directory, interval=1, max_points=None)
    _trainer(tmp_path, writer).train()

    payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    iterations = [entry["iteration"] for entry in payload["snapshots"]]
    assert iterations == sorted(iterations)
    assert payload["frame"] == 1
    assert payload["format_version"] == 1
