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
