from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from gaussian_splatting.config import (
    ConfigError,
    config_from_mapping,
    load_config,
    resolve_device,
    resolve_dtype,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def test_load_config_accepts_complete_documented_configuration() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.model.sh_degree == 3
    assert config.initialization.aabb_half_extent == (1.0, 1.0, 1.0)
    assert config.rendering.background == (0.0, 0.0, 0.0)
    assert resolve_dtype(config.runtime).is_floating_point
    assert resolve_device(config.runtime).type in {"cpu", "cuda"}
    assert asdict(config.density_control) == {
        "densify_from_iteration": 500,
        "densify_until_iteration": 15_000,
        "densification_interval": 100,
        "position_gradient_threshold": 0.0002,
        "percent_dense": 0.01,
        "prune_opacity_threshold": 0.005,
        "opacity_reset_interval": 3_000,
        "opacity_reset_maximum": 0.01,
        "prune_screen_radius_threshold": 20.0,
        "prune_world_scale_fraction": 0.1,
    }


def test_config_dataclasses_are_frozen() -> None:
    config = load_config(DEFAULT_CONFIG)

    with pytest.raises(FrozenInstanceError):
        config.model.sh_degree = 2  # type: ignore[misc]


def test_config_from_mapping_round_trips_checkpoint_mapping() -> None:
    config = load_config(DEFAULT_CONFIG)

    restored = config_from_mapping(asdict(config))

    assert restored == config


@pytest.mark.parametrize(
    "replacement",
    [
        "  unknown: 0\n",
        "  sh_degree: 3.0\n  epsilon_q: 1.0e-8\n",
        "  sh_degree: 2\n  epsilon_q: 1.0e-8\n",
    ],
)
def test_load_config_rejects_unknown_wrong_type_and_unsupported_degree(
    tmp_path: Path, replacement: str
) -> None:
    source = DEFAULT_CONFIG.read_text(encoding="utf-8")
    start = source.index("model:\n")
    end = source.index("\ninitialization:\n")
    invalid = source[:start] + "model:\n" + replacement + source[end:]
    path = tmp_path / "invalid.yaml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_rejects_missing_required_key(tmp_path: Path) -> None:
    source = DEFAULT_CONFIG.read_text(encoding="utf-8")
    invalid = source.replace("  camera_file: camera_poses_blender.json\n", "")
    path = tmp_path / "missing.yaml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(ConfigError, match="camera_file"):
        load_config(path)


@pytest.mark.parametrize(
    ("adaptive_density_control", "opacity_reset"),
    [(True, False), (False, True), (True, True)],
)
def test_density_control_feature_flags_are_independently_supported(
    adaptive_density_control: bool,
    opacity_reset: bool,
) -> None:
    values = asdict(load_config(DEFAULT_CONFIG))
    values["features"]["adaptive_density_control"] = adaptive_density_control
    values["features"]["opacity_reset"] = opacity_reset

    config = config_from_mapping(values)

    assert config.features.adaptive_density_control is adaptive_density_control
    assert config.features.opacity_reset is opacity_reset


def test_progressive_sh_degree_remains_unsupported() -> None:
    values = asdict(load_config(DEFAULT_CONFIG))
    values["features"]["progressive_sh_degree"] = True

    with pytest.raises(ConfigError, match="progressive_sh_degree"):
        config_from_mapping(values)


@pytest.mark.parametrize("key", ["densification_interval", "percent_dense"])
def test_density_control_schema_rejects_missing_keys(key: str) -> None:
    values = asdict(load_config(DEFAULT_CONFIG))
    del values["density_control"][key]

    with pytest.raises(ConfigError, match=key):
        config_from_mapping(values)


def test_density_control_schema_rejects_unknown_keys() -> None:
    values = asdict(load_config(DEFAULT_CONFIG))
    values["density_control"]["unknown_threshold"] = 1.0

    with pytest.raises(ConfigError, match="unknown_threshold"):
        config_from_mapping(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("densify_from_iteration", -1, "non-negative"),
        ("densify_from_iteration", True, "must be int"),
        ("densify_until_iteration", 500, "greater than"),
        ("densification_interval", 0, "positive"),
        ("densification_interval", False, "must be int"),
        ("position_gradient_threshold", -0.1, "non-negative"),
        ("position_gradient_threshold", float("nan"), "finite"),
        ("percent_dense", 0.0, "0 < value <= 1"),
        ("percent_dense", 1.1, "0 < value <= 1"),
        ("percent_dense", True, "must be float"),
        ("percent_dense", float("inf"), "finite"),
        ("prune_opacity_threshold", -0.1, "0 <= value < 1"),
        ("prune_opacity_threshold", 1.0, "0 <= value < 1"),
        ("opacity_reset_interval", 0, "positive"),
        ("opacity_reset_maximum", 0.0, "0 < value < 1"),
        ("opacity_reset_maximum", 1.0, "0 < value < 1"),
        ("prune_screen_radius_threshold", 0.0, "positive"),
        ("prune_screen_radius_threshold", float("-inf"), "finite"),
        ("prune_world_scale_fraction", 0.0, "positive"),
        ("prune_world_scale_fraction", float("nan"), "finite"),
    ],
)
def test_density_control_validation_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    values = asdict(load_config(DEFAULT_CONFIG))
    values["density_control"][field] = value

    with pytest.raises(ConfigError, match=message):
        config_from_mapping(values)


def test_short_training_allows_default_densify_until_iteration() -> None:
    values = asdict(load_config(DEFAULT_CONFIG))
    values["training"]["iterations"] = 1_000

    config = config_from_mapping(values)

    assert config.training.iterations == 1_000
    assert config.density_control.densify_until_iteration == 15_000
