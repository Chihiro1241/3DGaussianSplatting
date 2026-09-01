"""Strict configuration loading for the reference 3DGS implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


class ConfigError(ValueError):
    """Raised when a configuration file does not match the documented schema."""


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    dtype: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    camera_file: str
    resolution_scale: float
    rgba_background: str
    test_every: int


@dataclass(frozen=True)
class ModelConfig:
    sh_degree: int
    epsilon_q: float


@dataclass(frozen=True)
class InitializationConfig:
    num_gaussians: int
    aabb_half_extent: tuple[float, float, float]
    initial_rgb: tuple[float, float, float]
    neighbor_count: int
    epsilon_scale: float
    opacity: float


@dataclass(frozen=True)
class RenderingConfig:
    background: tuple[float, float, float]
    sigma_extent: float
    epsilon_covariance: float
    epsilon_determinant: float
    alpha_max: float
    alpha_min: float
    transmittance_min: float
    tile_based: bool


@dataclass(frozen=True)
class LossConfig:
    lambda_dssim: float
    ssim_window_size: int
    ssim_sigma: float
    ssim_k1: float
    ssim_k2: float


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int
    views_per_iteration: int
    optimizer: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    position_lr_initial: float
    position_lr_final: float
    sh_dc_lr: float
    sh_rest_lr: float
    opacity_lr: float
    scale_lr: float
    quaternion_lr: float
    log_interval: int
    evaluation_interval: int
    checkpoint_interval: int
    save_best_by: str


@dataclass(frozen=True)
class AdaptiveDensityControlConfig:
    densify_from_iteration: int
    densify_until_iteration: int
    densification_interval: int
    position_gradient_threshold: float
    percent_dense: float
    prune_opacity_threshold: float
    opacity_reset_interval: int
    opacity_reset_maximum: float
    prune_screen_radius_threshold: float
    prune_world_scale_fraction: float


@dataclass(frozen=True)
class OutputConfig:
    exist_policy: str
    save_rendered_images: bool


@dataclass(frozen=True)
class FeatureConfig:
    adaptive_density_control: bool
    opacity_reset: bool
    progressive_sh_degree: bool
    resolution_warmup: bool
    random_initial_sh_dc: bool


@dataclass(frozen=True)
class Config:
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    initialization: InitializationConfig
    rendering: RenderingConfig
    loss: LossConfig
    training: TrainingConfig
    density_control: AdaptiveDensityControlConfig
    output: OutputConfig
    features: FeatureConfig


_SCHEMA: dict[str, tuple[type[Any], dict[str, object]]] = {
    "runtime": (
        RuntimeConfig,
        {"device": str, "dtype": str, "seed": int},
    ),
    "data": (
        DataConfig,
        {
            "camera_file": str,
            "resolution_scale": float,
            "rgba_background": str,
            "test_every": int,
        },
    ),
    "model": (
        ModelConfig,
        {"sh_degree": int, "epsilon_q": float},
    ),
    "initialization": (
        InitializationConfig,
        {
            "num_gaussians": int,
            "aabb_half_extent": (float, 3),
            "initial_rgb": (float, 3),
            "neighbor_count": int,
            "epsilon_scale": float,
            "opacity": float,
        },
    ),
    "rendering": (
        RenderingConfig,
        {
            "background": (float, 3),
            "sigma_extent": float,
            "epsilon_covariance": float,
            "epsilon_determinant": float,
            "alpha_max": float,
            "alpha_min": float,
            "transmittance_min": float,
            "tile_based": bool,
        },
    ),
    "loss": (
        LossConfig,
        {
            "lambda_dssim": float,
            "ssim_window_size": int,
            "ssim_sigma": float,
            "ssim_k1": float,
            "ssim_k2": float,
        },
    ),
    "training": (
        TrainingConfig,
        {
            "iterations": int,
            "views_per_iteration": int,
            "optimizer": str,
            "adam_beta1": float,
            "adam_beta2": float,
            "adam_epsilon": float,
            "weight_decay": float,
            "position_lr_initial": float,
            "position_lr_final": float,
            "sh_dc_lr": float,
            "sh_rest_lr": float,
            "opacity_lr": float,
            "scale_lr": float,
            "quaternion_lr": float,
            "log_interval": int,
            "evaluation_interval": int,
            "checkpoint_interval": int,
            "save_best_by": str,
        },
    ),
    "density_control": (
        AdaptiveDensityControlConfig,
        {
            "densify_from_iteration": int,
            "densify_until_iteration": int,
            "densification_interval": int,
            "position_gradient_threshold": float,
            "percent_dense": float,
            "prune_opacity_threshold": float,
            "opacity_reset_interval": int,
            "opacity_reset_maximum": float,
            "prune_screen_radius_threshold": float,
            "prune_world_scale_fraction": float,
        },
    ),
    "output": (
        OutputConfig,
        {"exist_policy": str, "save_rendered_images": bool},
    ),
    "features": (
        FeatureConfig,
        {
            "adaptive_density_control": bool,
            "opacity_reset": bool,
            "progressive_sh_degree": bool,
            "resolution_warmup": bool,
            "random_initial_sh_dc": bool,
        },
    ),
}


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
    if not all(type(key) is str for key in value):
        raise ConfigError(f"{path} keys must all be strings")
    return value


def _check_keys(
    values: Mapping[str, object], expected: set[str], path: str
) -> None:
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ConfigError(f"{path}: {'; '.join(details)}")


def _convert_value(value: object, specification: object, path: str) -> object:
    if isinstance(specification, tuple):
        element_type, length = specification
        if type(value) not in (list, tuple):
            raise ConfigError(
                f"{path} must be a list or tuple, got {type(value).__name__}"
            )
        sequence = value
        if len(sequence) != length:
            raise ConfigError(f"{path} must contain exactly {length} values")
        for index, element in enumerate(sequence):
            if type(element) is not element_type:
                raise ConfigError(
                    f"{path}[{index}] must be {element_type.__name__}, "
                    f"got {type(element).__name__}"
                )
        return tuple(sequence)

    expected_type = specification
    if type(value) is not expected_type:
        raise ConfigError(
            f"{path} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _build_section(
    section_name: str,
    values: object,
    section_type: type[Any],
    fields: Mapping[str, object],
) -> Any:
    mapping = _mapping(values, section_name)
    _check_keys(mapping, set(fields), section_name)
    converted = {
        field_name: _convert_value(
            mapping[field_name], field_specification, f"{section_name}.{field_name}"
        )
        for field_name, field_specification in fields.items()
    }
    return section_type(**converted)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate_config(config: Config) -> None:
    _require(config.runtime.device in {"auto", "cpu", "cuda"},
             "runtime.device must be one of: auto, cpu, cuda")
    _require(config.runtime.dtype in {"float32", "float64"},
             "runtime.dtype must be float32 or float64")

    _require(bool(config.data.camera_file), "data.camera_file must not be empty")
    _require(0.0 < config.data.resolution_scale <= 1.0,
             "data.resolution_scale must satisfy 0 < value <= 1")
    _require(config.data.rgba_background in {"black", "white"},
             "data.rgba_background must be black or white")
    _require(config.data.test_every > 1, "data.test_every must be greater than 1")

    _require(config.model.sh_degree == 3,
             "model.sh_degree must be 3 in the initial implementation")
    _require(config.model.epsilon_q > 0.0, "model.epsilon_q must be positive")

    initialization = config.initialization
    _require(initialization.num_gaussians >= 4,
             "initialization.num_gaussians must be at least 4")
    _require(initialization.neighbor_count > 0,
             "initialization.neighbor_count must be positive")
    _require(initialization.num_gaussians > initialization.neighbor_count,
             "initialization.num_gaussians must exceed neighbor_count")
    _require(all(value > 0.0 for value in initialization.aabb_half_extent),
             "initialization.aabb_half_extent values must be positive")
    _require(all(0.0 <= value <= 1.0 for value in initialization.initial_rgb),
             "initialization.initial_rgb values must be in [0, 1]")
    _require(initialization.epsilon_scale > 0.0,
             "initialization.epsilon_scale must be positive")
    _require(0.0 < initialization.opacity < 1.0,
             "initialization.opacity must be strictly between 0 and 1")

    rendering = config.rendering
    _require(all(0.0 <= value <= 1.0 for value in rendering.background),
             "rendering.background values must be in [0, 1]")
    _require(rendering.sigma_extent > 0.0, "rendering.sigma_extent must be positive")
    _require(rendering.epsilon_covariance > 0.0,
             "rendering.epsilon_covariance must be positive")
    _require(rendering.epsilon_determinant > 0.0,
             "rendering.epsilon_determinant must be positive")
    _require(0.0 < rendering.alpha_min <= rendering.alpha_max < 1.0,
             "rendering alpha bounds must satisfy 0 < alpha_min <= alpha_max < 1")
    _require(0.0 < rendering.transmittance_min < 1.0,
             "rendering.transmittance_min must be strictly between 0 and 1")
    _require(not rendering.tile_based,
             "rendering.tile_based is not supported by the initial implementation")

    loss = config.loss
    _require(0.0 <= loss.lambda_dssim <= 1.0,
             "loss.lambda_dssim must be in [0, 1]")
    _require(loss.ssim_window_size > 0 and loss.ssim_window_size % 2 == 1,
             "loss.ssim_window_size must be a positive odd integer")
    _require(loss.ssim_sigma > 0.0, "loss.ssim_sigma must be positive")
    _require(loss.ssim_k1 > 0.0 and loss.ssim_k2 > 0.0,
             "loss.ssim_k1 and loss.ssim_k2 must be positive")

    training = config.training
    _require(training.iterations > 0, "training.iterations must be positive")
    _require(training.views_per_iteration == 1,
             "training.views_per_iteration must be 1 (batch training is unsupported)")
    _require(training.optimizer == "adam", "training.optimizer must be adam")
    _require(0.0 <= training.adam_beta1 < 1.0 and 0.0 <= training.adam_beta2 < 1.0,
             "training Adam beta values must be in [0, 1)")
    _require(training.adam_epsilon > 0.0, "training.adam_epsilon must be positive")
    _require(training.weight_decay >= 0.0, "training.weight_decay must be non-negative")
    learning_rates = (
        training.position_lr_initial,
        training.position_lr_final,
        training.sh_dc_lr,
        training.sh_rest_lr,
        training.opacity_lr,
        training.scale_lr,
        training.quaternion_lr,
    )
    _require(all(value > 0.0 for value in learning_rates),
             "all training learning rates must be positive")
    _require(training.log_interval > 0 and training.evaluation_interval > 0
             and training.checkpoint_interval > 0,
             "training intervals must be positive")
    _require(training.save_best_by == "mean_psnr",
             "training.save_best_by must be mean_psnr")

    density_control = config.density_control
    _require(
        density_control.densify_from_iteration >= 0,
        "density_control.densify_from_iteration must be non-negative",
    )
    _require(
        density_control.densify_until_iteration
        > density_control.densify_from_iteration,
        "density_control.densify_until_iteration must be greater than "
        "densify_from_iteration",
    )
    _require(
        density_control.densification_interval > 0,
        "density_control.densification_interval must be positive",
    )
    _require(
        density_control.opacity_reset_interval > 0,
        "density_control.opacity_reset_interval must be positive",
    )
    finite_fields = {
        "position_gradient_threshold": density_control.position_gradient_threshold,
        "percent_dense": density_control.percent_dense,
        "prune_opacity_threshold": density_control.prune_opacity_threshold,
        "opacity_reset_maximum": density_control.opacity_reset_maximum,
        "prune_screen_radius_threshold": (
            density_control.prune_screen_radius_threshold
        ),
        "prune_world_scale_fraction": density_control.prune_world_scale_fraction,
    }
    for field_name, value in finite_fields.items():
        _require(
            math.isfinite(value),
            f"density_control.{field_name} must be finite",
        )
    _require(
        density_control.position_gradient_threshold >= 0.0,
        "density_control.position_gradient_threshold must be non-negative",
    )
    _require(
        0.0 < density_control.percent_dense <= 1.0,
        "density_control.percent_dense must satisfy 0 < value <= 1",
    )
    _require(
        0.0 <= density_control.prune_opacity_threshold < 1.0,
        "density_control.prune_opacity_threshold must satisfy 0 <= value < 1",
    )
    _require(
        0.0 < density_control.opacity_reset_maximum < 1.0,
        "density_control.opacity_reset_maximum must satisfy 0 < value < 1",
    )
    _require(
        density_control.prune_screen_radius_threshold > 0.0,
        "density_control.prune_screen_radius_threshold must be positive",
    )
    _require(
        density_control.prune_world_scale_fraction > 0.0,
        "density_control.prune_world_scale_fraction must be positive",
    )

    _require(config.output.exist_policy == "error",
             "output.exist_policy must be error")


def config_from_mapping(values: Mapping[str, object]) -> Config:
    """Build a :class:`Config` from a complete strict mapping.

    This is also the checkpoint reconstruction path for ``dataclasses.asdict``
    output, whose fixed-length configuration sequences are tuples.

    Unknown keys, missing keys, type mismatches, and unsupported values all
    raise :class:`ConfigError`.
    """

    top_level = _mapping(values, "config")
    _check_keys(top_level, set(_SCHEMA), "config")
    sections = {
        name: _build_section(name, top_level[name], section_type, fields)
        for name, (section_type, fields) in _SCHEMA.items()
    }
    config = Config(**sections)
    _validate_config(config)
    return config


def load_config(path: str | Path) -> Config:
    """Load a complete YAML configuration without applying implicit defaults.

    Unknown keys, missing keys, YAML type mismatches, and unsupported values all
    raise :class:`ConfigError`.
    """

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

    return config_from_mapping(_mapping(raw, "config"))


def resolve_device(runtime: RuntimeConfig) -> torch.device:
    """Resolve ``runtime.device``, preferring CUDA when it is set to ``auto``."""

    if runtime.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda, but CUDA is not available")
    return torch.device(runtime.device)


def resolve_dtype(runtime: RuntimeConfig) -> torch.dtype:
    """Resolve the configured floating-point dtype."""

    dtypes = {"float32": torch.float32, "float64": torch.float64}
    try:
        return dtypes[runtime.dtype]
    except KeyError as error:
        raise ConfigError(f"unsupported runtime.dtype: {runtime.dtype}") from error


__all__ = [
    "AdaptiveDensityControlConfig",
    "Config",
    "ConfigError",
    "DataConfig",
    "FeatureConfig",
    "InitializationConfig",
    "LossConfig",
    "ModelConfig",
    "OutputConfig",
    "RenderingConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "config_from_mapping",
    "load_config",
    "resolve_device",
    "resolve_dtype",
]
