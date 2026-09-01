"""Trainable Gaussian parameter storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


_FLOAT_DTYPES = (torch.float32, torch.float64)

GAUSSIAN_PARAMETER_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("means_world", (3,)),
    ("raw_quaternions", (4,)),
    ("raw_scales", (3,)),
    ("raw_opacities", (1,)),
    ("sh_dc", (1, 3)),
    ("sh_rest", (15, 3)),
)
GAUSSIAN_PARAMETER_NAMES = tuple(name for name, _ in GAUSSIAN_PARAMETER_SPECS)


def _validate_parameter_tensor(
    name: str, value: Tensor, shape_tail: tuple[int, ...]
) -> int:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
        expected = ("N", *shape_tail)
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if value.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or Inf")
    return value.shape[0]


@dataclass
class GaussianParameters:
    """Transformed rendering parameters.

    Shapes are ``(N,3)``, ``(N,4)``, ``(N,3)``, ``(N,1)``, and
    ``(N,16,3)`` respectively. Together with :class:`GaussianModel`, this is
    the implementation location for TeX labels ``eq:gaussian_parameters`` and
    ``eq:raw_gaussian_parameters``.
    """

    means_world: Tensor
    quaternions: Tensor
    scales: Tensor
    opacities: Tensor
    sh_coefficients: Tensor

    def __post_init__(self) -> None:
        counts = (
            _validate_parameter_tensor("means_world", self.means_world, (3,)),
            _validate_parameter_tensor("quaternions", self.quaternions, (4,)),
            _validate_parameter_tensor("scales", self.scales, (3,)),
            _validate_parameter_tensor("opacities", self.opacities, (1,)),
            _validate_parameter_tensor(
                "sh_coefficients", self.sh_coefficients, (16, 3)
            ),
        )
        if len(set(counts)) != 1:
            raise ValueError("all Gaussian parameter tensors must have the same N")
        tensors = (
            self.means_world,
            self.quaternions,
            self.scales,
            self.opacities,
            self.sh_coefficients,
        )
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise TypeError("all Gaussian parameter tensors must have the same dtype")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all Gaussian parameter tensors must be on the same device")


class GaussianModel(nn.Module):
    """Raw trainable Gaussian parameters (TeX: eq:raw_gaussian_parameters)."""

    means_world: nn.Parameter
    raw_quaternions: nn.Parameter
    raw_scales: nn.Parameter
    raw_opacities: nn.Parameter
    sh_dc: nn.Parameter
    sh_rest: nn.Parameter

    def __init__(
        self,
        means_world: Tensor,
        raw_quaternions: Tensor,
        raw_scales: Tensor,
        raw_opacities: Tensor,
        sh_dc: Tensor,
        sh_rest: Tensor,
        epsilon_q: float = 1e-8,
        sh_degree: int = 3,
    ) -> None:
        super().__init__()
        if type(sh_degree) is not int or sh_degree != 3:
            raise ValueError("GaussianModel only supports sh_degree=3")
        if isinstance(epsilon_q, bool) or not isinstance(epsilon_q, (int, float)):
            raise TypeError("epsilon_q must be a real number")
        if not math.isfinite(float(epsilon_q)) or epsilon_q <= 0:
            raise ValueError("epsilon_q must be finite and positive")

        named_tensors = {
            "means_world": means_world,
            "raw_quaternions": raw_quaternions,
            "raw_scales": raw_scales,
            "raw_opacities": raw_opacities,
            "sh_dc": sh_dc,
            "sh_rest": sh_rest,
        }
        self.validate_gaussian_parameter_tensors(named_tensors)

        self.epsilon_q = float(epsilon_q)
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.means_world = nn.Parameter(means_world.detach().clone())
        self.raw_quaternions = nn.Parameter(raw_quaternions.detach().clone())
        self.raw_scales = nn.Parameter(raw_scales.detach().clone())
        self.raw_opacities = nn.Parameter(raw_opacities.detach().clone())
        self.sh_dc = nn.Parameter(sh_dc.detach().clone())
        self.sh_rest = nn.Parameter(sh_rest.detach().clone())

    def set_active_sh_degree(self, degree: int) -> None:
        """Select the highest SH degree used by both rendering backends."""
        if type(degree) is not int or not 0 <= degree <= self.sh_degree:
            raise ValueError(f"active SH degree must be in [0, {self.sh_degree}]")
        self.active_sh_degree = degree

    def one_up_sh_degree(self) -> int:
        """Increase the active SH degree by one, capped at the model maximum."""
        self.active_sh_degree = min(self.active_sh_degree + 1, self.sh_degree)
        return self.active_sh_degree

    @classmethod
    def validate_gaussian_parameter_tensors(
        cls,
        tensors: Mapping[str, Tensor],
    ) -> tuple[int, torch.dtype, torch.device]:
        """Validate one complete, index-aligned set of raw Gaussian tensors."""

        if not isinstance(tensors, Mapping):
            raise TypeError("Gaussian parameter tensors must be provided as a mapping")
        expected_names = set(GAUSSIAN_PARAMETER_NAMES)
        actual_names = set(tensors)
        missing = sorted(expected_names - actual_names)
        unknown = sorted(actual_names - expected_names)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing parameters: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown parameters: {', '.join(unknown)}")
            raise ValueError(f"invalid Gaussian parameter mapping; {'; '.join(details)}")

        counts = {
            _validate_parameter_tensor(name, tensors[name], shape_tail)
            for name, shape_tail in GAUSSIAN_PARAMETER_SPECS
        }
        if len(counts) != 1:
            raise ValueError("all raw Gaussian parameter tensors must have the same N")
        values = tuple(tensors[name] for name in GAUSSIAN_PARAMETER_NAMES)
        if len({tensor.dtype for tensor in values}) != 1:
            raise TypeError("all raw Gaussian parameter tensors must have the same dtype")
        if len({tensor.device for tensor in values}) != 1:
            raise ValueError("all raw Gaussian parameter tensors must be on the same device")
        return counts.pop(), values[0].dtype, values[0].device

    def gaussian_parameter_dict(self) -> dict[str, nn.Parameter]:
        """Return all raw Parameters in their shared Gaussian-index order."""

        return {name: getattr(self, name) for name in GAUSSIAN_PARAMETER_NAMES}

    def replace_gaussian_parameters(
        self,
        parameters: Mapping[str, nn.Parameter],
    ) -> None:
        """Atomically install one complete, validated set of raw Parameters.

        Callers that already own an optimizer must also replace its parameter
        references and state.  The transaction helpers in
        :mod:`gaussian_splatting.training.optimizer` perform both operations.
        """

        self.validate_gaussian_parameter_tensors(parameters)
        for name in GAUSSIAN_PARAMETER_NAMES:
            parameter = parameters[name]
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(f"{name} must be an nn.Parameter")
            if not parameter.requires_grad:
                raise ValueError(f"{name} must require gradients")
        if len({id(parameters[name]) for name in GAUSSIAN_PARAMETER_NAMES}) != len(
            GAUSSIAN_PARAMETER_NAMES
        ):
            raise ValueError("each Gaussian parameter must be a distinct nn.Parameter")

        previous = self.gaussian_parameter_dict()
        installed: list[str] = []
        try:
            for name in GAUSSIAN_PARAMETER_NAMES:
                setattr(self, name, parameters[name])
                installed.append(name)
        except BaseException:
            for name in installed:
                setattr(self, name, previous[name])
            raise

    @property
    def num_gaussians(self) -> int:
        """Return the number of Gaussians stored by the model."""

        return self.means_world.shape[0]

    @property
    def sh_coefficients(self) -> Tensor:
        """Concatenate the independently optimized DC and higher-order SH terms."""

        return torch.cat((self.sh_dc, self.sh_rest), dim=1)

    def transformed_parameters(self) -> GaussianParameters:
        """Apply raw transforms and return renderer values.

        TeX: eq:raw_parameter_transformations, eq:gaussian_parameters
        """

        from gaussian_splatting.math.parameterization import (  # noqa: PLC0415
            raw_parameter_transformations,
        )

        quaternions, scales, opacities = raw_parameter_transformations(
            self.raw_quaternions,
            self.raw_scales,
            self.raw_opacities,
            epsilon_q=self.epsilon_q,
        )
        return GaussianParameters(
            means_world=self.means_world,
            quaternions=quaternions,
            scales=scales,
            opacities=opacities,
            sh_coefficients=self.sh_coefficients,
        )


__all__ = [
    "GAUSSIAN_PARAMETER_NAMES",
    "GAUSSIAN_PARAMETER_SPECS",
    "GaussianModel",
    "GaussianParameters",
]
