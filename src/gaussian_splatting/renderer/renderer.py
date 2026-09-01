"""High-level Gaussian renderer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.cuda_rasterizer import (
    CudaProjectedGaussians,
    prepare_cuda_rasterization,
    rasterize_gaussians_cuda,
)
from gaussian_splatting.renderer.projection import ProjectedGaussians, project_gaussians
from gaussian_splatting.renderer.rasterizer import rasterize_gaussians

if TYPE_CHECKING:
    from gaussian_splatting.training.profiling import WallClockIterationTimer


@dataclass
class RenderResult:
    """Complete renderer output without duplicating full-model intermediates."""

    image: Tensor  # (3, H, W)
    final_transmittance: Tensor | None  # (H, W); unavailable from CUDA API
    projected: ProjectedGaussians | CudaProjectedGaussians
    visible_mask: Tensor  # (N,), bool
    screen_space_points: Tensor | None = None  # CUDA ADC gradient carrier


class GaussianRenderer(nn.Module):
    """Compose parameter transformation, projection, and rasterization."""

    def __init__(
        self,
        config: RenderingConfig,
        *,
        backend: str = "reference",
    ) -> None:
        super().__init__()
        if not isinstance(config, RenderingConfig):
            raise TypeError("config must be a RenderingConfig")
        self.config = config
        if backend not in {"reference", "cuda"}:
            raise ValueError("renderer backend must be 'reference' or 'cuda'")
        self.backend = backend

    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        retain_screen_grad: bool = False,
        profile_timer: WallClockIterationTimer | None = None,
    ) -> RenderResult:
        """Render ``model`` from ``camera`` exactly once per processing stage."""

        if profile_timer is None:
            parameters = model.transformed_parameters()
        else:
            with profile_timer.measure("parameter_transform"):
                parameters = model.transformed_parameters()
        active_sh_degree = getattr(model, "active_sh_degree", 3)

        if self.backend == "cuda":
            if profile_timer is None:
                preparation = prepare_cuda_rasterization(
                    parameters, camera, self.config, active_sh_degree
                )
                image, projected, visible_mask = rasterize_gaussians_cuda(
                    parameters, camera, preparation
                )
            else:
                with profile_timer.measure("projection"):
                    preparation = prepare_cuda_rasterization(
                        parameters, camera, self.config, active_sh_degree
                    )
                with profile_timer.measure("rasterization"):
                    image, projected, visible_mask = rasterize_gaussians_cuda(
                        parameters, camera, preparation
                    )
            return RenderResult(
                image=image,
                final_transmittance=None,
                projected=projected,
                visible_mask=visible_mask,
                screen_space_points=preparation.screen_space_points,
            )

        if profile_timer is None:
            projected, visible_mask = project_gaussians(
                parameters, camera, self.config, active_sh_degree
            )
        else:
            with profile_timer.measure("projection"):
                projected, visible_mask = project_gaussians(
                    parameters, camera, self.config, active_sh_degree
                )
        if retain_screen_grad and projected.means_screen.requires_grad:
            projected.means_screen.retain_grad()

        background = torch.tensor(
            self.config.background,
            dtype=parameters.means_world.dtype,
            device=parameters.means_world.device,
        )
        if profile_timer is None:
            image, final_transmittance = rasterize_gaussians(
                projected,
                camera.height,
                camera.width,
                background,
                alpha_max=self.config.alpha_max,
                alpha_min=self.config.alpha_min,
                transmittance_min=self.config.transmittance_min,
            )
        else:
            with profile_timer.measure("rasterization"):
                image, final_transmittance = rasterize_gaussians(
                    projected,
                    camera.height,
                    camera.width,
                    background,
                    alpha_max=self.config.alpha_max,
                    alpha_min=self.config.alpha_min,
                    transmittance_min=self.config.transmittance_min,
                )
        return RenderResult(
            image=image,
            final_transmittance=final_transmittance,
            projected=projected,
            visible_mask=visible_mask,
        )
