from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.training.schedules import (
    compute_scene_extent,
    density_control_event_parameters,
    density_control_schedule,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(
    *,
    adaptive_density_control: bool = True,
    opacity_reset: bool = True,
    rgba_background: str = "black",
) -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        data=replace(config.data, rgba_background=rgba_background),
        features=replace(
            config.features,
            adaptive_density_control=adaptive_density_control,
            opacity_reset=opacity_reset,
        ),
    )


def _camera(center: torch.Tensor) -> Camera:
    return Camera(
        rotation_cw=torch.eye(3, dtype=center.dtype, device=center.device),
        translation_cw=-center,
        camera_center_world=center,
        fx=10.0,
        fy=10.0,
        cx=5.0,
        cy=5.0,
        width=10,
        height=10,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scene_extent_matches_training_camera_center_radius(
    dtype: torch.dtype,
) -> None:
    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=dtype,
    )
    cameras = [_camera(center) for center in centers]
    expected = float(
        (
            torch.linalg.vector_norm(centers - centers.mean(dim=0), dim=-1).max()
            * 1.1
        ).item()
    )

    extent = compute_scene_extent(cameras)

    assert extent == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scene_extent_is_translation_invariant(dtype: torch.dtype) -> None:
    centers = torch.tensor(
        [[-1.0, 0.0, 2.0], [3.0, 2.0, -2.0], [0.0, -4.0, 1.0]],
        dtype=dtype,
    )
    translation = torch.tensor([100.0, -50.0, 25.0], dtype=dtype)

    original = compute_scene_extent([_camera(center) for center in centers])
    translated = compute_scene_extent(
        [_camera(center + translation) for center in centers]
    )

    assert translated == pytest.approx(original, rel=1e-5, abs=1e-5)


def test_scene_extent_uses_only_explicitly_supplied_training_cameras() -> None:
    train_centers = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    evaluation_camera = _camera(torch.tensor([1_000.0, 0.0, 0.0]))

    train_extent = compute_scene_extent(
        [_camera(center) for center in train_centers]
    )
    extent_with_evaluation = compute_scene_extent(
        [_camera(center) for center in train_centers] + [evaluation_camera]
    )

    assert train_extent == pytest.approx(1.1)
    assert extent_with_evaluation > train_extent


def test_scene_extent_rejects_empty_identical_and_nonfinite_centers() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_scene_extent([])

    identical = [_camera(torch.tensor([1.0, 2.0, 3.0])) for _ in range(2)]
    with pytest.raises(ValueError, match="positive"):
        compute_scene_extent(identical)

    invalid_camera = _camera(torch.zeros(3))
    object.__setattr__(
        invalid_camera,
        "camera_center_world",
        torch.tensor([float("inf"), 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="finite"):
        compute_scene_extent([invalid_camera, _camera(torch.ones(3))])


@pytest.mark.parametrize(
    ("iteration", "collect", "event", "reset", "size_pruning"),
    [
        (499, True, False, False, False),
        (500, True, False, False, False),
        (599, True, False, False, False),
        (600, True, True, False, False),
        (2_999, True, False, False, False),
        (3_000, True, True, True, False),
        (3_001, True, False, False, True),
        (14_900, True, True, False, True),
        (14_999, True, False, False, True),
        (15_000, False, False, False, False),
    ],
)
def test_default_schedule_boundaries(
    iteration: int,
    collect: bool,
    event: bool,
    reset: bool,
    size_pruning: bool,
) -> None:
    decision = density_control_schedule(_config(), iteration)

    assert decision.collect_statistics is collect
    assert decision.run_density_control_event is event
    assert decision.run_opacity_reset is reset
    assert decision.size_pruning_active is size_pruning


def test_white_background_special_reset_and_periodic_reset() -> None:
    black = _config(rgba_background="black")
    white = _config(rgba_background="white")

    assert not density_control_schedule(black, 500).run_opacity_reset
    assert density_control_schedule(white, 500).run_opacity_reset
    assert density_control_schedule(black, 3_000).run_opacity_reset
    assert density_control_schedule(white, 3_000).run_opacity_reset


def test_feature_flags_are_independent_and_disabled_policy_is_inert() -> None:
    disabled = density_control_schedule(
        _config(adaptive_density_control=False, opacity_reset=False),
        3_000,
    )
    opacity_only = density_control_schedule(
        _config(adaptive_density_control=False, opacity_reset=True),
        3_000,
    )
    density_only = density_control_schedule(
        _config(adaptive_density_control=True, opacity_reset=False),
        3_000,
    )

    assert disabled.collect_statistics is False
    assert disabled.run_density_control_event is False
    assert disabled.run_opacity_reset is False
    assert disabled.size_pruning_active is False
    assert opacity_only.collect_statistics is False
    assert opacity_only.run_density_control_event is False
    assert opacity_only.run_opacity_reset is True
    assert opacity_only.size_pruning_active is False
    assert density_only.collect_statistics is True
    assert density_only.run_density_control_event is True
    assert density_only.run_opacity_reset is False


def test_schedule_and_threshold_policy_do_not_consume_rng() -> None:
    config = _config(adaptive_density_control=False, opacity_reset=False)
    torch.manual_seed(90)
    rng_state = torch.get_rng_state().clone()

    density_control_schedule(config, 600)
    density_control_event_parameters(config, 600, scene_extent=10.0)

    assert torch.equal(torch.get_rng_state(), rng_state)


def test_event_parameters_derive_scale_thresholds_and_activate_size_pruning() -> None:
    config = _config()

    early = density_control_event_parameters(config, 3_000, scene_extent=200.0)
    late = density_control_event_parameters(config, 3_001, scene_extent=200.0)

    assert early.gradient_threshold == pytest.approx(0.0002)
    assert early.densify_world_scale_threshold == pytest.approx(2.0)
    assert early.prune_opacity_threshold == pytest.approx(0.005)
    assert early.prune_screen_radius_threshold is None
    assert early.prune_world_scale_threshold is None
    assert late.prune_screen_radius_threshold == pytest.approx(20.0)
    assert late.prune_world_scale_threshold == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("iteration", "exception", "message"),
    [
        (True, TypeError, "integer"),
        (-1, ValueError, "non-negative"),
    ],
)
def test_schedule_rejects_invalid_iterations(
    iteration: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        density_control_schedule(_config(), iteration)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("scene_extent", "exception", "message"),
    [
        (True, TypeError, "real number"),
        (0.0, ValueError, "positive"),
        (-1.0, ValueError, "positive"),
        (float("nan"), ValueError, "finite"),
        (float("inf"), ValueError, "finite"),
    ],
)
def test_event_parameters_reject_invalid_scene_extent(
    scene_extent: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        density_control_event_parameters(
            _config(),
            600,
            scene_extent=scene_extent,  # type: ignore[arg-type]
        )
