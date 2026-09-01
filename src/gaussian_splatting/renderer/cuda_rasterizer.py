"""Adapter for GraphDeco's official differentiable CUDA rasterizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianParameters


@dataclass
class CudaProjectedGaussians:
    """CUDA-native projection information required by density control."""

    original_indices: Tensor  # (Nv,), int64
    means_screen: Tensor  # (Nv, 2), diagnostic pixel coordinates
    depths: Tensor  # (Nv,)
    radii: Tensor  # (Nv,), int64


@dataclass
class CudaRasterizationPreparation:
    """Inputs created outside the timed CUDA rasterizer invocation."""

    rasterizer: Any
    screen_space_points: Tensor  # (N, 3), official ADC gradient carrier


def _load_cuda_rasterizer() -> tuple[type[Any], type[Any]]:
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as error:
        raise RuntimeError(
            "render backend 'cuda' requires the official "
            "diff-gaussian-rasterization extension"
        ) from error
    return GaussianRasterizationSettings, GaussianRasterizer


def _validate_cuda_configuration(
    parameters: GaussianParameters,
    config: RenderingConfig,
) -> None:
    if parameters.means_world.device.type != "cuda":
        raise RuntimeError("render backend 'cuda' requires CUDA model tensors")
    if parameters.means_world.dtype != torch.float32:
        raise TypeError("diff-gaussian-rasterization requires float32 tensors")
    expected = {
        "sigma_extent": (config.sigma_extent, 3.0),
        "epsilon_covariance": (config.epsilon_covariance, 0.3),
        "alpha_max": (config.alpha_max, 0.99),
        "alpha_min": (config.alpha_min, 1.0 / 255.0),
        "transmittance_min": (config.transmittance_min, 1e-4),
    }
    mismatched = [
        name for name, (actual, required) in expected.items() if actual != required
    ]
    if mismatched:
        raise ValueError(
            "CUDA backend uses fixed official rasterizer constants; unsupported "
            f"rendering settings: {', '.join(mismatched)}"
        )


def _camera_matrices(camera: Camera, reference: Tensor) -> tuple[Tensor, Tensor]:
    """Build the transposed column-major matrices expected by GraphDeco."""

    near = 0.2
    far = 1000.0
    world_to_view = torch.eye(4, dtype=reference.dtype, device=reference.device)
    world_to_view[:3, :3] = camera.rotation_cw
    world_to_view[:3, 3] = camera.translation_cw

    projection = torch.zeros(
        (4, 4), dtype=reference.dtype, device=reference.device
    )
    projection[0, 0] = 2.0 * camera.fx / camera.width
    projection[1, 1] = 2.0 * camera.fy / camera.height
    # GraphDeco maps NDC to pixels with ((v + 1) * size - 1) / 2.
    projection[0, 2] = (2.0 * camera.cx + 1.0) / camera.width - 1.0
    projection[1, 2] = (2.0 * camera.cy + 1.0) / camera.height - 1.0
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -(far * near) / (far - near)
    projection[3, 2] = 1.0

    view_matrix = world_to_view.transpose(0, 1).contiguous()
    projection_matrix = projection.transpose(0, 1).contiguous()
    full_projection = (view_matrix @ projection_matrix).contiguous()
    return view_matrix, full_projection


def prepare_cuda_rasterization(
    parameters: GaussianParameters,
    camera: Camera,
    config: RenderingConfig,
    active_sh_degree: int = 3,
) -> CudaRasterizationPreparation:
    """Create camera settings and the official screen-gradient carrier."""

    _validate_cuda_configuration(parameters, config)
    settings_type, rasterizer_type = _load_cuda_rasterizer()
    reference = parameters.means_world
    view_matrix, full_projection = _camera_matrices(camera, reference)
    background = reference.new_tensor(config.background)
    settings = settings_type(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=float(camera.width / (2.0 * camera.fx)),
        tanfovy=float(camera.height / (2.0 * camera.fy)),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_projection,
        sh_degree=active_sh_degree,
        campos=camera.camera_center_world.contiguous(),
        prefiltered=False,
        debug=False,
    )
    screen_space_points = torch.zeros_like(
        reference,
        requires_grad=True,
    )
    return CudaRasterizationPreparation(
        rasterizer=rasterizer_type(raster_settings=settings),
        screen_space_points=screen_space_points,
    )


def rasterize_gaussians_cuda(
    parameters: GaussianParameters,
    camera: Camera,
    preparation: CudaRasterizationPreparation,
) -> tuple[Tensor, CudaProjectedGaussians, Tensor]:
    """Rasterize all model Gaussians and expose CUDA visibility/ADC metadata."""

    image, radii_int32 = preparation.rasterizer(
        means3D=parameters.means_world.contiguous(),
        means2D=preparation.screen_space_points,
        opacities=parameters.opacities.contiguous(),
        shs=parameters.sh_coefficients.contiguous(),
        colors_precomp=None,
        scales=parameters.scales.contiguous(),
        rotations=parameters.quaternions.contiguous(),
        cov3D_precomp=None,
    )
    visible_mask = radii_int32 > 0
    original_indices = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    radii = radii_int32[original_indices].to(dtype=torch.int64)
    with torch.no_grad():
        means_world = parameters.means_world[original_indices]
        means_camera = (
            means_world @ camera.rotation_cw.transpose(0, 1)
            + camera.translation_cw
        )
        depths = means_camera[:, 2]
        means_screen = torch.stack(
            (
                camera.fx * means_camera[:, 0] / depths + camera.cx,
                camera.fy * means_camera[:, 1] / depths + camera.cy,
            ),
            dim=-1,
        )
    projected = CudaProjectedGaussians(
        original_indices=original_indices,
        means_screen=means_screen,
        depths=depths,
        radii=radii,
    )
    return image, projected, visible_mask


__all__ = [
    "CudaProjectedGaussians",
    "CudaRasterizationPreparation",
    "prepare_cuda_rasterization",
    "rasterize_gaussians_cuda",
]
