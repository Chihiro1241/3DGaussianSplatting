"""Projection and differentiable reference rasterization APIs."""

from gaussian_splatting.renderer.cuda_rasterizer import CudaProjectedGaussians

from gaussian_splatting.renderer.projection import (
    ProjectedGaussians,
    gaussian_rendering_radius,
    gaussian_rendering_rectangle,
    implementation_depth_order,
    maximum_2d_covariance_eigenvalue,
    pixel_coordinate_convention,
    project_gaussians,
    visible_depth_condition,
)
from gaussian_splatting.renderer.rasterizer import (
    clamped_projected_opacity,
    implementation_pixel_color_with_background,
    implementation_projected_opacity,
    opacity_contribution_threshold,
    rasterize_gaussians,
    stable_color_accumulation,
    stable_transmittance_accumulation,
    transmittance_termination,
)
from gaussian_splatting.renderer.renderer import GaussianRenderer, RenderResult

__all__ = [
    "GaussianRenderer",
    "CudaProjectedGaussians",
    "ProjectedGaussians",
    "RenderResult",
    "clamped_projected_opacity",
    "gaussian_rendering_radius",
    "gaussian_rendering_rectangle",
    "implementation_depth_order",
    "implementation_pixel_color_with_background",
    "implementation_projected_opacity",
    "maximum_2d_covariance_eigenvalue",
    "opacity_contribution_threshold",
    "pixel_coordinate_convention",
    "project_gaussians",
    "rasterize_gaussians",
    "stable_color_accumulation",
    "stable_transmittance_accumulation",
    "transmittance_termination",
    "visible_depth_condition",
]
