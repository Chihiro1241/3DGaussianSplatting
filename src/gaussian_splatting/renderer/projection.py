"""Gaussian projection and screen-space culling.

The functions in this module intentionally mirror the equation names in
``document/3DGS_定式化.tex``.  The discontinuous culling and ordering values are
kept separate from the differentiable projected Gaussian attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.math.covariance import (
    covariance_2d,
    covariance_3d,
    quaternion_rotation_matrix,
    stabilized_2d_covariance,
    stabilized_inverse_2d_covariance,
    symmetrized_2d_covariance,
)
from gaussian_splatting.math.spherical_harmonics import sh_color_implementation
from gaussian_splatting.math.transform import (
    perspective_projection,
    projection_jacobian,
    view_direction,
    world_to_camera,
)
from gaussian_splatting.model.gaussian_model import GaussianParameters


NEAR_PLANE = 0.2
FOV_GUARD_BAND = 1.3
_MAX_SAFE_RADIUS = 2_147_483_647

@dataclass
class ProjectedGaussians:
    """Differentiable attributes for the ``Nv`` visible Gaussians.

    Every field is ordered front-to-back.  ``original_indices`` maps each
    entry back to the corresponding row in the input :class:`GaussianModel`.
    """

    original_indices: Tensor  # (Nv,), int64
    means_camera: Tensor  # (Nv, 3)
    means_screen: Tensor  # (Nv, 2)
    depths: Tensor  # (Nv,)
    covariances_3d: Tensor  # (Nv, 3, 3)
    covariances_2d: Tensor  # (Nv, 2, 2), stabilized
    inverse_covariances_2d: Tensor  # (Nv, 2, 2)
    colors: Tensor  # (Nv, 3)
    opacities: Tensor  # (Nv, 1)
    radii: Tensor  # (Nv,), int64
    rectangles: Tensor  # (Nv, 4), inclusive xmin, xmax, ymin, ymax


def _require_float_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")


def _require_last_dimensions(name: str, value: Tensor, dimensions: tuple[int, ...]) -> None:
    if value.ndim < len(dimensions) or tuple(value.shape[-len(dimensions) :]) != dimensions:
        raise ValueError(f"{name} must end with shape {dimensions}, got {tuple(value.shape)}")


def visible_depth_condition(means_camera: Tensor) -> Tensor:
    """Return the mask selecting Gaussian centers with positive depth.

    TeX: eq:visible_depth_condition
    """

    _require_float_tensor("means_camera", means_camera)
    _require_last_dimensions("means_camera", means_camera, (3,))
    return means_camera[..., 2] > 0


def maximum_2d_covariance_eigenvalue(covariances_2d: Tensor) -> Tensor:
    """Compute the larger eigenvalue of each symmetric 2D covariance.

    TeX: eq:maximum_2d_covariance_eigenvalue
    """

    _require_float_tensor("covariances_2d", covariances_2d)
    _require_last_dimensions("covariances_2d", covariances_2d, (2, 2))
    a = covariances_2d[..., 0, 0]
    b = covariances_2d[..., 0, 1]
    c = covariances_2d[..., 1, 1]
    discriminant = (a - c).square() + 4.0 * b.square()
    return 0.5 * (a + c + torch.sqrt(discriminant))


def gaussian_rendering_radius(
    maximum_eigenvalues: Tensor,
    sigma_extent: float = 3.0,
) -> Tensor:
    """Return integer screen-space radii from maximum covariance eigenvalues.

    TeX: eq:gaussian_rendering_radius
    """

    _require_float_tensor("maximum_eigenvalues", maximum_eigenvalues)
    if sigma_extent <= 0:
        raise ValueError("sigma_extent must be positive")
    # Radius selection is a discontinuous culling decision.  Detaching makes
    # the intended absence of gradients explicit.
    values = maximum_eigenvalues.detach().clamp(
        min=0.0,
        max=float(_MAX_SAFE_RADIUS) ** 2,
    )
    return torch.ceil(float(sigma_extent) * torch.sqrt(values)).to(torch.int64)


def gaussian_rendering_rectangle(
    means_screen: Tensor,
    radii: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Return inclusive ``(xmin, xmax, ymin, ymax)`` rendering rectangles.

    Intersection is decided in floating point before conversion to integers.
    Intersecting bounds are clipped on both sides to the valid image range;
    non-intersecting or non-finite bounds receive the in-range empty sentinel
    ``(1, 0, 1, 0)``. This prevents extreme coordinates from wrapping during
    float-to-int conversion.

    TeX: eq:gaussian_rendering_rectangle
    """

    _require_float_tensor("means_screen", means_screen)
    _require_last_dimensions("means_screen", means_screen, (2,))
    if not isinstance(radii, Tensor):
        raise TypeError("radii must be a torch.Tensor")
    if radii.shape != means_screen.shape[:-1]:
        raise ValueError(
            "radii must match the leading dimensions of means_screen, "
            f"got {tuple(radii.shape)} and {tuple(means_screen.shape)}"
        )
    if radii.device != means_screen.device:
        raise ValueError("radii and means_screen must be on the same device")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    centers = means_screen.detach()
    radii_float = radii.detach().to(dtype=centers.dtype)
    raw_xmin = torch.floor(centers[..., 0] - radii_float)
    raw_xmax = torch.ceil(centers[..., 0] + radii_float)
    raw_ymin = torch.floor(centers[..., 1] - radii_float)
    raw_ymax = torch.ceil(centers[..., 1] + radii_float)
    finite = torch.isfinite(centers).all(dim=-1)
    intersects = finite & (raw_xmax >= 0.0) & (raw_xmin <= width - 1)
    intersects &= (raw_ymax >= 0.0) & (raw_ymin <= height - 1)
    xmin = raw_xmin.clamp(0.0, float(width - 1))
    xmax = raw_xmax.clamp(0.0, float(width - 1))
    ymin = raw_ymin.clamp(0.0, float(height - 1))
    ymax = raw_ymax.clamp(0.0, float(height - 1))
    # Invalid/non-intersecting rows use an in-range empty sentinel, avoiding
    # undefined float-to-int conversion for extreme projected coordinates.
    xmin = torch.where(intersects, xmin, torch.ones_like(xmin))
    xmax = torch.where(intersects, xmax, torch.zeros_like(xmax))
    ymin = torch.where(intersects, ymin, torch.ones_like(ymin))
    ymax = torch.where(intersects, ymax, torch.zeros_like(ymax))
    return torch.stack((xmin, xmax, ymin, ymax), dim=-1).to(torch.int64)


def pixel_coordinate_convention(
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create the ``(H, W, 2)`` integer-centered ``(x, y)`` pixel grid.

    TeX: eq:pixel_coordinate_convention
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    y = torch.arange(height, dtype=dtype, device=device)
    x = torch.arange(width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def implementation_depth_order(depths: Tensor, original_indices: Tensor) -> Tensor:
    """Return a deterministic lexicographic order by depth and source index.

    TeX: eq:implementation_depth_order
    """

    _require_float_tensor("depths", depths)
    if depths.ndim != 1:
        raise ValueError(f"depths must have shape (N,), got {tuple(depths.shape)}")
    if not isinstance(original_indices, Tensor) or original_indices.ndim != 1:
        raise ValueError("original_indices must be a one-dimensional tensor")
    if original_indices.dtype != torch.int64:
        raise TypeError("original_indices must have dtype torch.int64")
    if depths.shape != original_indices.shape:
        raise ValueError("depths and original_indices must have equal shape")
    if depths.device != original_indices.device:
        raise ValueError("depths and original_indices must be on the same device")
    if not bool(torch.isfinite(depths).all()):
        raise ValueError("depths contains NaN or Inf")

    # First establish the secondary-key ordering, then use a stable primary
    # sort.  This remains correct even if callers provide indices out of order.
    source_order = torch.argsort(original_indices, stable=True)
    depth_order = torch.argsort(depths.detach()[source_order], stable=True)
    return source_order[depth_order]


def _validate_parameter_batch(
    parameters: GaussianParameters,
) -> tuple[int, torch.dtype, torch.device]:
    fields = {
        "means_world": (parameters.means_world, (3,)),
        "quaternions": (parameters.quaternions, (4,)),
        "scales": (parameters.scales, (3,)),
        "opacities": (parameters.opacities, (1,)),
        "sh_coefficients": (parameters.sh_coefficients, (16, 3)),
    }
    first = parameters.means_world
    _require_float_tensor("means_world", first)
    if first.ndim != 2 or first.shape[1] != 3:
        raise ValueError(f"means_world must have shape (N, 3), got {tuple(first.shape)}")
    count = first.shape[0]
    for name, (value, trailing_shape) in fields.items():
        _require_float_tensor(name, value)
        expected = (count, *trailing_shape)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if value.dtype != first.dtype:
            raise TypeError("all Gaussian parameter tensors must have the same dtype")
        if value.device != first.device:
            raise ValueError("all Gaussian parameter tensors must be on the same device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
    return count, first.dtype, first.device


def project_gaussians(
    parameters: GaussianParameters,
    camera: Camera,
    config: RenderingConfig,
) -> tuple[ProjectedGaussians, Tensor]:
    """Perform all per-Gaussian projection work exactly once.

    The returned :class:`ProjectedGaussians` contains only Gaussians beyond
    the near plane, inside the 1.3x field-of-view guard band, and whose
    rendering rectangle intersects the image. Its rows are ordered by
    increasing depth, using the original index as a tie-breaker. The second
    return value is the visibility mask over all input Gaussians.
    """

    count, _, device = _validate_parameter_batch(parameters)
    if config.tile_based:
        raise ValueError("tile_based rendering is outside the initial implementation")

    means_camera_all = world_to_camera(
        parameters.means_world,
        camera.rotation_cw,
        camera.translation_cw,
    )
    depth_mask = means_camera_all[:, 2] > NEAR_PLANE
    depth_indices = torch.nonzero(depth_mask, as_tuple=False).squeeze(1)
    depth_candidates = means_camera_all[depth_indices]
    normalized_x = depth_candidates[:, 0] / depth_candidates[:, 2]
    normalized_y = depth_candidates[:, 1] / depth_candidates[:, 2]
    limit_x = FOV_GUARD_BAND * camera.width / (2.0 * camera.fx)
    limit_y = FOV_GUARD_BAND * camera.height / (2.0 * camera.fy)
    guard_mask = (normalized_x.abs() <= limit_x) & (normalized_y.abs() <= limit_y)
    candidate_indices = depth_indices[guard_mask]

    means_world = parameters.means_world[candidate_indices]
    means_camera = means_camera_all[candidate_indices]
    means_screen = perspective_projection(
        means_camera,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
    )
    projection_jacobians = projection_jacobian(means_camera, camera.fx, camera.fy)

    rotation_matrices = quaternion_rotation_matrix(parameters.quaternions[candidate_indices])
    covariances_3d = covariance_3d(
        rotation_matrices,
        parameters.scales[candidate_indices],
    )
    covariances_2d = covariance_2d(
        projection_jacobians,
        camera.rotation_cw,
        covariances_3d,
    )
    covariances_2d = symmetrized_2d_covariance(covariances_2d)
    covariances_2d = stabilized_2d_covariance(
        covariances_2d,
        config.epsilon_covariance,
    )
    inverse_covariances_2d = stabilized_inverse_2d_covariance(
        covariances_2d,
        config.epsilon_determinant,
    )

    directions = view_direction(means_world, camera.camera_center_world)
    colors = sh_color_implementation(
        directions,
        parameters.sh_coefficients[candidate_indices],
        active_degree=3,
    )
    opacities = parameters.opacities[candidate_indices]
    depths = means_camera[:, 2]

    maximum_eigenvalues = maximum_2d_covariance_eigenvalue(covariances_2d)
    radii = gaussian_rendering_radius(maximum_eigenvalues, config.sigma_extent)
    rectangles = gaussian_rendering_rectangle(
        means_screen,
        radii,
        camera.height,
        camera.width,
    )
    intersects_image = (rectangles[:, 0] <= rectangles[:, 1]) & (
        rectangles[:, 2] <= rectangles[:, 3]
    )

    visible_mask = torch.zeros(count, dtype=torch.bool, device=device)
    visible_indices = candidate_indices[intersects_image]
    visible_mask[visible_indices] = True

    means_camera = means_camera[intersects_image]
    means_screen = means_screen[intersects_image]
    depths = depths[intersects_image]
    covariances_3d = covariances_3d[intersects_image]
    covariances_2d = covariances_2d[intersects_image]
    inverse_covariances_2d = inverse_covariances_2d[intersects_image]
    colors = colors[intersects_image]
    opacities = opacities[intersects_image]
    radii = radii[intersects_image]
    rectangles = rectangles[intersects_image]

    order = implementation_depth_order(depths, visible_indices)
    projected = ProjectedGaussians(
        original_indices=visible_indices[order],
        means_camera=means_camera[order],
        means_screen=means_screen[order],
        depths=depths[order],
        covariances_3d=covariances_3d[order],
        covariances_2d=covariances_2d[order],
        inverse_covariances_2d=inverse_covariances_2d[order],
        colors=colors[order],
        opacities=opacities[order],
        radii=radii[order],
        rectangles=rectangles[order],
    )
    return projected, visible_mask


__all__ = [
    "FOV_GUARD_BAND", "NEAR_PLANE", "ProjectedGaussians",
    "gaussian_rendering_radius", "gaussian_rendering_rectangle",
    "implementation_depth_order",
    "maximum_2d_covariance_eigenvalue", "pixel_coordinate_convention",
    "project_gaussians", "visible_depth_condition",
]
