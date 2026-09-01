"""Complete, reproducible training checkpoints."""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch
from torch.optim import Optimizer

from gaussian_splatting.config import Config
from gaussian_splatting.model.gaussian_model import GaussianModel

if TYPE_CHECKING:
    from gaussian_splatting.training.density_control import (
        ScreenSpaceDensityStatistics,
    )
    from gaussian_splatting.training.schedules import PositionLearningRateScheduler


CHECKPOINT_VERSION = 2
_SUPPORTED_CHECKPOINT_VERSIONS = frozenset({1, CHECKPOINT_VERSION})
_BASE_CHECKPOINT_KEYS = frozenset(
    {
        "iteration",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_random_state",
        "torch_cuda_random_states",
        "camera_order",
        "camera_cursor",
        "best_mean_psnr",
    }
)
_DENSITY_STATISTICS_KEYS = frozenset(
    {
        "position_gradient_accumulator",
        "position_gradient_denominator",
        "max_screen_radius",
    }
)


def _config_dict(config: Config | Mapping[str, Any]) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a Config dataclass or mapping")


def save_checkpoint(
    path: str | Path,
    *,
    iteration: int,
    model: GaussianModel,
    optimizer: Optimizer,
    scheduler: PositionLearningRateScheduler,
    config: Config | Mapping[str, Any],
    camera_order: list[int],
    camera_cursor: int,
    best_mean_psnr: float | None,
    density_statistics: ScreenSpaceDensityStatistics | None = None,
) -> dict[str, Any]:
    """Save model, optimizer, scheduler, RNG, and view-sampling state."""

    from gaussian_splatting.training.density_control import (
        ScreenSpaceDensityStatistics,
    )

    if density_statistics is not None:
        if not isinstance(density_statistics, ScreenSpaceDensityStatistics):
            raise TypeError(
                "density_statistics must be ScreenSpaceDensityStatistics or None"
            )
        density_statistics.validate_compatible(model)
        density_statistics_state = density_statistics.state_dict()
    else:
        density_statistics_state = None
    config_state = _config_dict(config)
    if not isinstance(config_state.get("density_control"), Mapping):
        raise ValueError(
            "version 2 checkpoint config must contain density_control mapping"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config_state,
        "density_statistics_state": density_statistics_state,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": cuda_states,
        "camera_order": [int(index) for index in camera_order],
        "camera_cursor": int(camera_cursor),
        "best_mean_psnr": None if best_mean_psnr is None else float(best_mean_psnr),
    }
    torch.save(state, destination)
    return state


def _validated_checkpoint_version(state: Mapping[str, Any]) -> int:
    version = state.get("checkpoint_version", 1)
    if type(version) is not int:
        raise TypeError("checkpoint_version must be an integer")
    if version not in _SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint_version {version}; supported versions are "
            f"{sorted(_SUPPORTED_CHECKPOINT_VERSIONS)}"
        )
    return version


def _validate_density_statistics_structure(state: object) -> None:
    if state is None:
        return
    if not isinstance(state, Mapping):
        raise TypeError("density_statistics_state must be a mapping or None")
    actual_keys = set(state)
    if actual_keys != _DENSITY_STATISTICS_KEYS:
        missing = sorted(_DENSITY_STATISTICS_KEYS - actual_keys)
        unexpected = sorted(
            actual_keys - _DENSITY_STATISTICS_KEYS,
            key=repr,
        )
        raise ValueError(
            "invalid density_statistics_state keys; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in _DENSITY_STATISTICS_KEYS:
        if not isinstance(state[name], torch.Tensor):
            raise TypeError(f"density_statistics_state {name!r} must be a Tensor")


def read_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Read and validate checkpoint structure without mutating runtime state."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    try:
        state = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        state = torch.load(source, map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError("invalid checkpoint; expected a mapping")
    version = _validated_checkpoint_version(state)
    required = set(_BASE_CHECKPOINT_KEYS)
    if version == CHECKPOINT_VERSION:
        required.update({"checkpoint_version", "density_statistics_state"})
    if not required.issubset(state):
        missing = sorted(required.difference(state))
        raise ValueError(f"invalid checkpoint; missing keys: {missing}")
    normalized = dict(state)
    if version == 1:
        legacy_statistics = normalized.get("density_statistics_state")
        if legacy_statistics is not None:
            raise ValueError(
                "version 1 checkpoint must not contain density statistics"
            )
        normalized["checkpoint_version"] = 1
        normalized["density_statistics_state"] = None
    else:
        config = normalized["config"]
        if not isinstance(config, Mapping) or not isinstance(
            config.get("density_control"), Mapping
        ):
            raise ValueError(
                "version 2 checkpoint config must contain density_control mapping"
            )
        _validate_density_statistics_structure(
            normalized["density_statistics_state"]
        )
    return normalized


def model_from_checkpoint_state(
    state: Mapping[str, Any],
    config: Config,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> GaussianModel:
    """Construct a model directly from checkpoint tensors without initialization."""

    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    parameter_names = (
        "means_world",
        "raw_quaternions",
        "raw_scales",
        "raw_opacities",
        "sh_dc",
        "sh_rest",
    )
    missing = [name for name in parameter_names if name not in model_state]
    if missing:
        raise ValueError(f"checkpoint model state is missing parameters: {missing}")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    tensors = {
        name: model_state[name].detach().to(device=device, dtype=dtype)
        for name in parameter_names
    }
    model = GaussianModel(
        **tensors,
        epsilon_q=config.model.epsilon_q,
        sh_degree=config.model.sh_degree,
    )
    iteration = int(state.get("iteration", 0))
    model.set_active_sh_degree(
        min(iteration // 1000, config.model.sh_degree)
        if config.features.progressive_sh_degree
        else config.model.sh_degree
    )
    return model


def load_checkpoint(
    path: str | Path,
    *,
    model: GaussianModel,
    optimizer: Optimizer | None = None,
    scheduler: PositionLearningRateScheduler | None = None,
    density_statistics: ScreenSpaceDensityStatistics | None = None,
    map_location: str | torch.device | None = None,
    restore_random_state: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore every supplied mutable training object."""

    state = read_checkpoint(path, map_location=map_location)
    checkpoint_statistics = state["density_statistics_state"]
    if checkpoint_statistics is not None and density_statistics is None:
        raise ValueError(
            "checkpoint contains density statistics but no runtime "
            "density_statistics object was supplied"
        )
    if checkpoint_statistics is None and density_statistics is not None:
        raise ValueError(
            "runtime density_statistics was supplied but checkpoint contains none"
        )
    if density_statistics is not None:
        from gaussian_splatting.training.density_control import (
            ScreenSpaceDensityStatistics,
        )

        if not isinstance(density_statistics, ScreenSpaceDensityStatistics):
            raise TypeError(
                "density_statistics must be ScreenSpaceDensityStatistics or None"
            )
        density_statistics.validate_compatible(model)
        if not isinstance(checkpoint_statistics, Mapping):  # pragma: no cover
            raise RuntimeError("validated checkpoint statistics are not a mapping")
        density_statistics.validate_state_dict(checkpoint_statistics)

    model_state = state["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    current_model_state = model.state_dict()
    if set(model_state) != set(current_model_state):
        raise ValueError("checkpoint model state keys do not match runtime model")
    for name, current_value in current_model_state.items():
        checkpoint_value = model_state[name]
        if not isinstance(checkpoint_value, torch.Tensor):
            raise TypeError(f"checkpoint model state {name!r} must be a Tensor")
        if checkpoint_value.shape != current_value.shape:
            raise ValueError(
                f"checkpoint model state {name!r} shape does not match runtime model"
            )

    model.load_state_dict(model_state, strict=True)
    model.set_active_sh_degree(
        min(int(state["iteration"]) // 1000, model.sh_degree)
        if isinstance(state["config"], Mapping)
        and isinstance(state["config"].get("features"), Mapping)
        and state["config"]["features"].get("progressive_sh_degree") is True
        else model.sh_degree
    )
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if density_statistics is not None:
        if not isinstance(checkpoint_statistics, Mapping):  # pragma: no cover
            raise RuntimeError("validated checkpoint statistics are not a mapping")
        density_statistics.load_state_dict(checkpoint_statistics)

    if restore_random_state:
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_cpu_random_state"].cpu())
        cuda_states = state["torch_cuda_random_states"]
        if cuda_states is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])
    return state


def validate_resume_config(
    saved: Mapping[str, Any],
    current: Config | Mapping[str, Any],
) -> None:
    """Reject resume-time configuration changes except output and log intervals."""

    saved_copy = dict(saved)
    current_copy = _config_dict(current)
    if "density_control" not in saved_copy:
        saved_features = saved_copy.get("features")
        current_features = current_copy.get("features")
        saved_fixed = (
            isinstance(saved_features, Mapping)
            and saved_features.get("adaptive_density_control") is False
            and saved_features.get("opacity_reset") is False
        )
        current_fixed = (
            isinstance(current_features, Mapping)
            and current_features.get("adaptive_density_control") is False
            and current_features.get("opacity_reset") is False
        )
        if saved_fixed and current_fixed:
            saved_copy["density_control"] = current_copy.get("density_control")
    saved_copy.pop("output", None)
    current_copy.pop("output", None)
    for config_dict in (saved_copy, current_copy):
        training = config_dict.get("training")
        if isinstance(training, Mapping):
            training = dict(training)
            training.pop("log_interval", None)
            training.pop("evaluation_interval", None)
            training.pop("checkpoint_interval", None)
            config_dict["training"] = training
    if saved_copy != current_copy:
        raise ValueError(
            "resume configuration differs from the checkpoint outside output/log intervals"
        )
