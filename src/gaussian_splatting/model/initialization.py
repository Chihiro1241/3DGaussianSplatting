"""Initialization of raw Gaussian model parameters."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor

from gaussian_splatting.config import Config, resolve_device, resolve_dtype
from gaussian_splatting.model.gaussian_model import GaussianModel


SH_DC_BASIS = 1.0 / (2.0 * math.sqrt(math.pi))
_FLOAT_DTYPES = (torch.float32, torch.float64)


def _points(points: Tensor, name: str = "points") -> None:
    if not isinstance(points, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(points.shape)}")
    if points.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
    if not torch.isfinite(points).all().item():
        raise ValueError(f"{name} contains NaN or Inf")


def _dtype(dtype: torch.dtype) -> None:
    if dtype not in _FLOAT_DTYPES:
        raise TypeError("dtype must be torch.float32 or torch.float64")


def implementation_position_initialization(points: Tensor) -> Tensor:
    """Use input points as Gaussian centers, preserving ``(N,3)`` and dtype.

    TeX: eq:implementation_position_initialization
    """

    _points(points)
    return points.clone()


def initial_neighbor_distance(points: Tensor, k: int = 3) -> Tensor:
    """Return mean squared distances to the ``k`` nearest other points.

    Args:
        points: Point positions with shape ``(N,3)``.
        k: Number of neighbors. The documented initial value is ``3``.

    Returns:
        Mean squared distances with shape ``(N,)`` and the input dtype/device.

    TeX: eq:initial_neighbor_distance
    """

    _points(points)
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if points.shape[0] < 4:
        raise ValueError("at least 4 points are required to select 3 neighbors")
    if points.shape[0] <= k:
        raise ValueError(f"point count must exceed k={k}")

    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - dependency installation failure
        raise RuntimeError("scipy is required for Gaussian initialization") from error

    point_array = points.detach().cpu().numpy()
    tree = cKDTree(point_array)
    distances, _ = tree.query(point_array, k=k + 1)
    neighbor_distances = distances[:, 1:]
    mean_squared = (neighbor_distances * neighbor_distances).mean(axis=1)
    return torch.as_tensor(mean_squared, dtype=points.dtype, device=points.device)


def initial_gaussian_scale(
    mean_squared_distances: Tensor, epsilon_scale: float = 1e-7
) -> Tensor:
    """Create isotropic scales ``(N,3)`` from neighbor distances ``(N,)``.

    TeX: eq:initial_gaussian_scale
    """

    if not isinstance(mean_squared_distances, Tensor):
        raise TypeError("mean_squared_distances must be a torch.Tensor")
    if mean_squared_distances.ndim != 1:
        raise ValueError("mean_squared_distances must have shape (N,)")
    if mean_squared_distances.dtype not in _FLOAT_DTYPES:
        raise TypeError("mean_squared_distances must be float32 or float64")
    if not torch.isfinite(mean_squared_distances).all().item():
        raise ValueError("mean_squared_distances contains NaN or Inf")
    if torch.any(mean_squared_distances < 0.0).item():
        raise ValueError("mean_squared_distances must be non-negative")
    if isinstance(epsilon_scale, bool) or not isinstance(epsilon_scale, Real):
        raise TypeError("epsilon_scale must be a real number")
    if not math.isfinite(float(epsilon_scale)) or epsilon_scale <= 0.0:
        raise ValueError("epsilon_scale must be finite and positive")

    scale = torch.sqrt(mean_squared_distances.clamp_min(float(epsilon_scale)))
    return scale.unsqueeze(1).expand(-1, 3)


def initial_raw_scale(scales: Tensor) -> Tensor:
    """Map positive scales ``(N,3)`` to raw log-scales ``(N,3)``.

    TeX: eq:initial_raw_scale
    """

    _points(scales, "scales")
    if torch.any(scales <= 0.0).item():
        raise ValueError("scales must be positive")
    return torch.log(scales)


def initial_raw_quaternion(
    num_gaussians: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return identity raw quaternions ``(N,4)`` in ``(qw,qx,qy,qz)`` order.

    TeX: eq:initial_raw_quaternion
    """

    if type(num_gaussians) is not int or num_gaussians < 0:
        raise ValueError("num_gaussians must be a non-negative integer")
    _dtype(dtype)
    values = torch.zeros((num_gaussians, 4), dtype=dtype, device=device)
    values[:, 0] = 1.0
    return values


def initial_opacity(
    num_gaussians: int,
    alpha_init: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return constant initial opacities with shape ``(N,1)``.

    TeX: eq:initial_opacity
    """

    if type(num_gaussians) is not int or num_gaussians < 0:
        raise ValueError("num_gaussians must be a non-negative integer")
    if isinstance(alpha_init, bool) or not isinstance(alpha_init, Real):
        raise TypeError("alpha_init must be a real number")
    if not math.isfinite(float(alpha_init)) or not 0.0 < alpha_init < 1.0:
        raise ValueError("alpha_init must be strictly between 0 and 1")
    _dtype(dtype)
    return torch.full(
        (num_gaussians, 1), float(alpha_init), dtype=dtype, device=device
    )


def initial_raw_opacity(opacities: Tensor) -> Tensor:
    """Apply logit to opacities ``(N,1)`` and preserve dtype/device.

    TeX: eq:initial_raw_opacity
    """

    if not isinstance(opacities, Tensor):
        raise TypeError("opacities must be a torch.Tensor")
    if opacities.ndim != 2 or opacities.shape[1] != 1:
        raise ValueError("opacities must have shape (N, 1)")
    if opacities.dtype not in _FLOAT_DTYPES:
        raise TypeError("opacities must be float32 or float64")
    if not torch.isfinite(opacities).all().item():
        raise ValueError("opacities contains NaN or Inf")
    if torch.any((opacities <= 0.0) | (opacities >= 1.0)).item():
        raise ValueError("opacities must be strictly between 0 and 1")
    return torch.log(opacities / (1.0 - opacities))


def initial_sh_dc(colors: Tensor) -> Tensor:
    """Initialize degree-zero SH coefficients from RGB ``(N,3)``.

    Returns a tensor of shape ``(N,1,3)``.

    TeX: eq:initial_sh_dc
    """

    _points(colors, "colors")
    if torch.any((colors < 0.0) | (colors > 1.0)).item():
        raise ValueError("colors must be in [0, 1]")
    return ((colors - 0.5) / SH_DC_BASIS).unsqueeze(1)


def initial_sh_rest(
    num_gaussians: int,
    sh_degree: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Initialize higher-order SH coefficients as zeros ``(N,15,3)``.

    TeX: eq:initial_sh_rest
    """

    if type(num_gaussians) is not int or num_gaussians < 0:
        raise ValueError("num_gaussians must be a non-negative integer")
    if type(sh_degree) is not int or sh_degree != 3:
        raise ValueError("initial_sh_rest only supports sh_degree=3")
    _dtype(dtype)
    coefficient_count = (sh_degree + 1) ** 2 - 1
    return torch.zeros(
        (num_gaussians, coefficient_count, 3), dtype=dtype, device=device
    )


def generate_initial_points(
    target: Tensor | Sequence[float],
    config: Config,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Generate uniform points in the target-centered configured AABB.

    Returns ``(points, colors)`` with shapes ``(N,3)`` and ``(N,3)``. The
    supplied generator must have been initialized with ``runtime.seed`` and is
    the only source of random values.
    """

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if generator.initial_seed() != config.runtime.seed:
        raise ValueError("generator must be initialized with runtime.seed")

    dtype = resolve_dtype(config.runtime)
    output_device = resolve_device(config.runtime)
    generator_device = torch.device(generator.device)
    target_tensor = torch.as_tensor(target, dtype=dtype, device=generator_device)
    if tuple(target_tensor.shape) != (3,):
        raise ValueError(f"target must have shape (3,), got {tuple(target_tensor.shape)}")
    if not torch.isfinite(target_tensor).all().item():
        raise ValueError("target contains NaN or Inf")

    initialization = config.initialization
    half_extent = torch.tensor(
        initialization.aabb_half_extent, dtype=dtype, device=generator_device
    )
    unit_points = torch.rand(
        (initialization.num_gaussians, 3),
        dtype=dtype,
        device=generator_device,
        generator=generator,
    )
    points = target_tensor + (2.0 * unit_points - 1.0) * half_extent
    if config.features.random_initial_sh_dc:
        random_sh_dc = torch.rand(
            (initialization.num_gaussians, 3),
            dtype=dtype,
            device=generator_device,
            generator=generator,
        ) / 255.0
        colors = 0.5 + SH_DC_BASIS * random_sh_dc
    else:
        colors = torch.tensor(
            initialization.initial_rgb, dtype=dtype, device=generator_device
        ).expand(initialization.num_gaussians, -1).clone()
    return points.to(output_device), colors.to(output_device)


def initialize_gaussian_model(
    points: Tensor, colors: Tensor, config: Config
) -> GaussianModel:
    """Build a :class:`GaussianModel` by applying all documented initializers."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    _points(points)
    _points(colors, "colors")
    if points.shape != colors.shape:
        raise ValueError("points and colors must both have shape (N, 3)")
    if points.dtype != colors.dtype:
        raise TypeError("points and colors must have the same dtype")
    if points.device != colors.device:
        raise ValueError("points and colors must be on the same device")
    if points.shape[0] < 4:
        raise ValueError("at least 4 points are required to initialize a model")

    initialization = config.initialization
    means_world = implementation_position_initialization(points)
    neighbor_distances = initial_neighbor_distance(
        points, k=initialization.neighbor_count
    )
    scales = initial_gaussian_scale(
        neighbor_distances, epsilon_scale=initialization.epsilon_scale
    )
    raw_scales = initial_raw_scale(scales)
    num_gaussians = points.shape[0]
    raw_quaternions = initial_raw_quaternion(
        num_gaussians, dtype=points.dtype, device=points.device
    )
    opacities = initial_opacity(
        num_gaussians,
        initialization.opacity,
        dtype=points.dtype,
        device=points.device,
    )
    raw_opacities = initial_raw_opacity(opacities)
    sh_dc = initial_sh_dc(colors)
    sh_rest = initial_sh_rest(
        num_gaussians,
        config.model.sh_degree,
        dtype=points.dtype,
        device=points.device,
    )
    model = GaussianModel(
        means_world=means_world,
        raw_quaternions=raw_quaternions,
        raw_scales=raw_scales,
        raw_opacities=raw_opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        epsilon_q=config.model.epsilon_q,
        sh_degree=config.model.sh_degree,
    )
    model.set_active_sh_degree(
        0 if config.features.progressive_sh_degree else config.model.sh_degree
    )
    return model


__all__ = [
    "SH_DC_BASIS",
    "generate_initial_points",
    "implementation_position_initialization",
    "initial_gaussian_scale",
    "initial_neighbor_distance",
    "initial_opacity",
    "initial_raw_opacity",
    "initial_raw_quaternion",
    "initial_raw_scale",
    "initial_sh_dc",
    "initial_sh_rest",
    "initialize_gaussian_model",
]
