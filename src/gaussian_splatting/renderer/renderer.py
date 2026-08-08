"""High-level Gaussian renderer orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.projection import ProjectedGaussians, project_gaussians
from gaussian_splatting.renderer.rasterizer import rasterize_gaussians


@dataclass
class RenderResult:
    """Complete renderer output without duplicating full-model intermediates."""

    image: Tensor  # (3, H, W)
    final_transmittance: Tensor  # (H, W)
    projected: ProjectedGaussians
    visible_mask: Tensor  # (N,), bool


class GaussianRenderer(nn.Module):
    """Compose parameter transformation, projection, and rasterization."""

    def __init__(self, config: RenderingConfig) -> None:
        super().__init__()
        if not isinstance(config, RenderingConfig):
            raise TypeError("config must be a RenderingConfig")
        self.config = config

    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        retain_screen_grad: bool = False,
    ) -> RenderResult:
        """Render ``model`` from ``camera`` exactly once per processing stage."""

        parameters = model.transformed_parameters()
        projected, visible_mask = project_gaussians(parameters, camera, self.config)
        if retain_screen_grad and projected.means_screen.requires_grad:
            projected.means_screen.retain_grad()

        background = torch.tensor(
            self.config.background,
            dtype=parameters.means_world.dtype,
            device=parameters.means_world.device,
        )
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
