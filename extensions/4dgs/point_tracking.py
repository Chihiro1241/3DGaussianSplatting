"""Dense point tracking with Dynamic 3D Gaussians.

This module implements "Tracking with Dynamic 3D Gaussians" of Luiten et al.,
"Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
(arXiv:2308.09713, Sec. 3).  It only produces trajectories; scoring them
against ground truth lives in :mod:`tracking_metrics`, so that trajectories of
any other method can be scored the same way.

* **Anchoring (3D query).**  A query point ``p`` at the reference frame is
  attached to the Gaussian with the largest influence

      f_i(p) = sigmoid(o_i) exp(-1/2 (p - mu_i)^T Sigma_i^-1 (p - mu_i)),
      Sigma_i = R_i S_i S_i^T R_i^T,

  evaluated with the reference frame's parameters.  If ``f_i(p) < 0.5`` for
  every Gaussian, the point belongs to the static background and stays where
  it is.  Otherwise it is expressed in that Gaussian's local frame,
  ``p_local = R_i,0^T (p - mu_i,0)``.

* **Propagation.**  At frame ``t`` the point is ``R_i,t p_local + mu_i,t``.

* **2D queries.**  A pixel of camera ``c`` at the reference frame is lifted to
  3D with its *median* depth: the centre depth of the Gaussian at which the
  ray's transmittance first drops below 0.5.  That Gaussian carries the
  point.  The point is then tracked as above and projected back into ``c``
  with ``K (E mu / (E mu)_z)``.  Only a ray that never becomes half opaque
  gives a static background point.

  The paper describes the depth as alpha-blended centre depth (a mean
  depth) and picks the carrier by the arg-max of ``f``.  The median depth
  follows the official ``diff-gaussian-rasterization-w-depth`` instead, and
  the carrier follows the author's note that "the median depth is the depth
  of the highest influence Gaussian" (JonathonLuiten/Dynamic3DGaussians
  issue #20).  The reason is practical: trained Gaussians are flat and often
  below 0.5 opacity, so a lifted surface point rarely has ``f >= 0.5`` for
  any Gaussian, and the paper's background rule would freeze most points.
  :func:`render_depth` (the mean depth) remains available.

The influence is evaluated exactly against every Gaussian, in chunks on the
GPU.  Pre-selecting candidates by centre distance would be faster but can
miss a large Gaussian whose centre is far from ``p``, so it is not done.  The
comparison is made in log space, ``log f = log sigmoid(o) - m / 2``, so that
points far from every centre do not underflow to a tie at zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from gaussian_splatting.config import RenderingConfig
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.math.covariance import quaternion_rotation_matrix
from gaussian_splatting.model import GaussianModel


#: A point is static background when no Gaussian influences it by this much.
BACKGROUND_INFLUENCE_THRESHOLD = 0.5

#: Index stored for background points, which have no anchoring Gaussian.
NO_GAUSSIAN = -1


@dataclass(frozen=True)
class FrameGaussians:
    """The per-frame motion parameters of every Gaussian.

    ``means`` is ``(N, 3)`` and ``rotations`` is ``(N, 3, 3)``.  Scale and
    opacity are only needed at the reference frame and live in
    :class:`ReferenceGaussians`.
    """

    means: Tensor
    rotations: Tensor

    @classmethod
    def from_model(cls, model: GaussianModel) -> FrameGaussians:
        parameters = model.transformed_parameters()
        return cls(
            means=parameters.means_world.detach(),
            rotations=quaternion_rotation_matrix(parameters.quaternions.detach()),
        )


@dataclass(frozen=True)
class ReferenceGaussians:
    """Everything needed to evaluate ``f_i(p)`` at the reference frame."""

    means: Tensor  # (N, 3)
    rotations: Tensor  # (N, 3, 3)
    scales: Tensor  # (N, 3), standard deviations
    opacities: Tensor  # (N,), after the sigmoid

    @classmethod
    def from_model(cls, model: GaussianModel) -> ReferenceGaussians:
        parameters = model.transformed_parameters()
        return cls(
            means=parameters.means_world.detach(),
            rotations=quaternion_rotation_matrix(parameters.quaternions.detach()),
            scales=parameters.scales.detach(),
            opacities=parameters.opacities.detach().reshape(-1),
        )

    @property
    def motion(self) -> FrameGaussians:
        return FrameGaussians(means=self.means, rotations=self.rotations)


@dataclass(frozen=True)
class Anchors:
    """Query points attached to the Gaussians that carry them.

    ``gaussian_indices`` is ``(Q,)`` with :data:`NO_GAUSSIAN` for background
    points, ``local_offsets`` is ``(Q, 3)`` (the point itself, in world
    coordinates, for background points), and ``influences`` is the largest
    ``f_i(p)`` found for each query.
    """

    gaussian_indices: Tensor
    local_offsets: Tensor
    influences: Tensor

    @property
    def is_background(self) -> Tensor:
        return self.gaussian_indices == NO_GAUSSIAN


def max_influence(
    points: Tensor,
    gaussians: ReferenceGaussians,
    *,
    query_chunk: int = 1024,
    gaussian_chunk: int = 65536,
) -> tuple[Tensor, Tensor]:
    """Return the most influential Gaussian of every point and its influence.

    ``points`` is ``(Q, 3)``.  Returns ``(indices, influences)``, both ``(Q,)``.
    The Mahalanobis distance is computed in each Gaussian's local frame,
    ``m = || S^-1 R^T (p - mu) ||^2``, which is exactly
    ``(p - mu)^T Sigma^-1 (p - mu)`` without inverting ``Sigma``.
    """

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (Q, 3), got {tuple(points.shape)}")
    count = gaussians.means.shape[0]
    if count == 0:
        raise ValueError("there are no Gaussians to anchor to")
    log_opacity = torch.log(gaussians.opacities.clamp_min(torch.finfo(points.dtype).tiny))
    inverse_scales = 1.0 / gaussians.scales

    best_log = torch.full(
        (points.shape[0],), -math.inf, dtype=points.dtype, device=points.device
    )
    best_index = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    for q_start in range(0, points.shape[0], query_chunk):
        query = points[q_start:q_start + query_chunk]
        chunk_best_log = best_log[q_start:q_start + query_chunk]
        chunk_best_index = best_index[q_start:q_start + query_chunk]
        for g_start in range(0, count, gaussian_chunk):
            g_end = min(g_start + gaussian_chunk, count)
            offsets = query[:, None, :] - gaussians.means[None, g_start:g_end]
            # R^T (p - mu): contract over the rotation's rows.
            local = torch.einsum("gji,qgj->qgi", gaussians.rotations[g_start:g_end], offsets)
            mahalanobis = (local * inverse_scales[None, g_start:g_end]).square().sum(-1)
            log_f = log_opacity[None, g_start:g_end] - 0.5 * mahalanobis
            value, index = log_f.max(dim=1)
            better = value > chunk_best_log
            chunk_best_log[better] = value[better]
            chunk_best_index[better] = index[better] + g_start
    return best_index, torch.exp(best_log)


def anchor_points(
    points: Tensor,
    gaussians: ReferenceGaussians,
    *,
    background_threshold: float = BACKGROUND_INFLUENCE_THRESHOLD,
    **chunks: int,
) -> Anchors:
    """Attach 3D query points at the reference frame to their Gaussians."""

    indices, influences = max_influence(points, gaussians, **chunks)
    background = influences < background_threshold
    offsets = torch.einsum(
        "qji,qj->qi", gaussians.rotations[indices], points - gaussians.means[indices]
    )
    offsets = torch.where(background[:, None], points, offsets)
    indices = torch.where(background, torch.full_like(indices, NO_GAUSSIAN), indices)
    return Anchors(gaussian_indices=indices, local_offsets=offsets, influences=influences)


def compact_anchors(anchors: Anchors) -> tuple[Tensor, Anchors]:
    """Renumber anchors onto only the Gaussians they use.

    Returns ``(used, compacted)``: ``used`` holds the original indices of the
    anchoring Gaussians, and ``compacted`` refers to them by position in
    ``used``.  A tracker then only needs ``means[used]`` and the rotations of
    those few Gaussians per frame instead of all ``N``.
    """

    foreground = ~anchors.is_background
    used, remapped = torch.unique(anchors.gaussian_indices[foreground], return_inverse=True)
    indices = anchors.gaussian_indices.clone()
    indices[foreground] = remapped
    return used, Anchors(
        gaussian_indices=indices,
        local_offsets=anchors.local_offsets,
        influences=anchors.influences,
    )


def frame_subset(model: GaussianModel, used: Tensor) -> FrameGaussians:
    """Return the motion parameters of the Gaussians ``used`` at one frame."""

    parameters = model.transformed_parameters()
    quaternions = parameters.quaternions.detach()[used]
    return FrameGaussians(
        means=parameters.means_world.detach()[used],
        rotations=quaternion_rotation_matrix(quaternions) if len(used) else
        quaternions.new_zeros((0, 3, 3)),
    )


def propagate(anchors: Anchors, frame: FrameGaussians) -> Tensor:
    """Return the ``(Q, 3)`` positions of the anchored points at one frame."""

    background = anchors.is_background
    if bool(background.all()):
        return anchors.local_offsets.clone()
    indices = anchors.gaussian_indices.clamp_min(0)
    moved = (
        torch.einsum("qij,qj->qi", frame.rotations[indices], anchors.local_offsets)
        + frame.means[indices]
    )
    return torch.where(background[:, None], anchors.local_offsets, moved)


def track_points(anchors: Anchors, frames: list[FrameGaussians]) -> Tensor:
    """Return ``(Q, T, 3)`` trajectories over ``frames`` (reference frame first)."""

    return torch.stack([propagate(anchors, frame) for frame in frames], dim=1)


# --------------------------------------------------------------------------
# cameras: projection, depth rendering and unprojection
# --------------------------------------------------------------------------


def project_points(points: Tensor, camera: Camera) -> tuple[Tensor, Tensor]:
    """Project ``(..., 3)`` world points into ``camera``.

    Returns ``(pixels, depths)`` with pixels ``(..., 2)`` from
    ``K (E mu / (E mu)_z)``, in the renderer's pixel convention (the centre
    of pixel ``(u, v)`` is at integer ``u, v``).
    """

    rotation = camera.rotation_cw.to(points)
    translation = camera.translation_cw.to(points)
    in_camera = points @ rotation.transpose(0, 1) + translation
    depths = in_camera[..., 2]
    pixels = torch.stack(
        (
            camera.fx * in_camera[..., 0] / depths + camera.cx,
            camera.fy * in_camera[..., 1] / depths + camera.cy,
        ),
        dim=-1,
    )
    return pixels, depths


def unproject_pixels(pixels: Tensor, depths: Tensor, camera: Camera) -> Tensor:
    """Lift ``(Q, 2)`` pixels with ``(Q,)`` camera depths to world points."""

    in_camera = torch.stack(
        (
            (pixels[:, 0] - camera.cx) / camera.fx * depths,
            (pixels[:, 1] - camera.cy) / camera.fy * depths,
            depths,
        ),
        dim=-1,
    )
    rotation = camera.rotation_cw.to(in_camera)
    translation = camera.translation_cw.to(in_camera)
    return (in_camera - translation) @ rotation


def render_depth(
    model: GaussianModel, camera: Camera, config: RenderingConfig
) -> tuple[Tensor, Tensor]:
    """Render ``(H, W)`` depth and opacity maps with the CUDA rasterizer.

    Each Gaussian's colour is replaced by the camera depth of its centre, as
    in the paper; a second pass with colour one gives the accumulated
    opacity.  The returned depth is the alpha-weighted depth divided by the
    accumulated opacity (zero where nothing was hit), i.e. the expected depth
    of the surface rather than one darkened toward the black background.
    """

    from gaussian_splatting.renderer.cuda_rasterizer import (  # noqa: PLC0415
        prepare_cuda_rasterization,
    )

    parameters = model.transformed_parameters()
    means = parameters.means_world.detach().contiguous()
    _, center_depths = project_points(means, camera)
    black = replace(config, background=(0.0, 0.0, 0.0))
    maps = []
    for colour in (center_depths, torch.ones_like(center_depths)):
        preparation = prepare_cuda_rasterization(parameters, camera, black, 0)
        with torch.no_grad():
            image, _ = preparation.rasterizer(
                means3D=means,
                means2D=preparation.screen_space_points,
                opacities=parameters.opacities.detach().contiguous(),
                shs=None,
                colors_precomp=colour[:, None].expand(-1, 3).contiguous(),
                scales=parameters.scales.detach().contiguous(),
                rotations=parameters.quaternions.detach().contiguous(),
                cov3D_precomp=None,
            )
        maps.append(image[0])
    weighted_depth, alpha = maps
    depth = torch.where(alpha > 1e-6, weighted_depth / alpha.clamp_min(1e-6), 0.0)
    return depth, alpha


def median_depth_gaussians(
    model: GaussianModel,
    camera: Camera,
    config: RenderingConfig,
    pixels: Tensor,
    *,
    query_chunk: int = 32,
) -> tuple[Tensor, Tensor]:
    """Return the Gaussian that sets each pixel's median depth, and that depth.

    Walking front to back along the ray of pixel ``(u, v)``, the median-depth
    Gaussian is the one at which the transmittance first drops below 0.5
    (``T_before > 0.5`` and ``T_after < 0.5``, as in the official
    ``diff-gaussian-rasterization-w-depth``).  Its centre depth is the median
    depth.  The alphas follow the renderer exactly: ``min(alpha_max,
    o exp(-m / 2))``, dropped below ``alpha_min`` or outside the Gaussian's
    screen rectangle.

    Returns ``(indices, depths)``, both ``(Q,)``.  ``indices`` are rows of the
    model, or :data:`NO_GAUSSIAN` (depth NaN) where the ray never becomes
    half opaque.  Only the queried pixels are evaluated, so the CUDA
    rasterizer (which has no median-depth output) is not needed.
    """

    from gaussian_splatting.renderer.projection import project_gaussians  # noqa: PLC0415

    with torch.no_grad():
        projected, _ = project_gaussians(model.transformed_parameters(), camera, config, 0)
        screen = projected.means_screen.detach()
        inverse = projected.inverse_covariances_2d.detach()
        opacities = projected.opacities.detach().reshape(-1)
        rectangles = projected.rectangles
        indices = torch.full(
            (pixels.shape[0],), NO_GAUSSIAN, dtype=torch.long, device=pixels.device
        )
        depths = torch.full(
            (pixels.shape[0],), math.nan, dtype=pixels.dtype, device=pixels.device
        )
        if screen.shape[0] == 0:
            return indices, depths
        for start in range(0, pixels.shape[0], query_chunk):
            query = pixels[start:start + query_chunk].to(screen)
            offsets = query[:, None, :] - screen[None]
            mahalanobis = torch.einsum("qni,nij,qnj->qn", offsets, inverse, offsets)
            alpha = (opacities[None] * torch.exp(-0.5 * mahalanobis)).clamp_max(
                config.alpha_max
            )
            column = query[:, 0].round().long()[:, None]
            row = query[:, 1].round().long()[:, None]
            inside = (
                (column >= rectangles[None, :, 0]) & (column <= rectangles[None, :, 1])
                & (row >= rectangles[None, :, 2]) & (row <= rectangles[None, :, 3])
            )
            alpha = torch.where(inside & (alpha >= config.alpha_min), alpha, 0.0)
            after = torch.cumprod(1.0 - alpha, dim=1)
            before = torch.cat((torch.ones_like(after[:, :1]), after[:, :-1]), dim=1)
            crossing = (before > 0.5) & (after < 0.5)
            hit = crossing.any(dim=1)
            first = crossing.to(torch.int8).argmax(dim=1)
            chosen = projected.original_indices[first]
            indices[start:start + query_chunk] = torch.where(
                hit, chosen, torch.full_like(chosen, NO_GAUSSIAN)
            )
            depths[start:start + query_chunk] = torch.where(
                hit, projected.depths.detach()[first].to(depths), math.nan
            )
    return indices, depths


def anchor_to_gaussians(
    points: Tensor, indices: Tensor, gaussians: ReferenceGaussians
) -> Anchors:
    """Attach points to given Gaussians (``NO_GAUSSIAN`` keeps a point static).

    Used for 2D queries, where the carrying Gaussian is the one that set the
    pixel's median depth rather than the arg-max of ``f``.  ``influences``
    still records ``f_i(p)`` for that Gaussian, for diagnostics.
    """

    background = indices == NO_GAUSSIAN
    safe = indices.clamp_min(0)
    offsets_world = points - gaussians.means[safe]
    local = torch.einsum("qji,qj->qi", gaussians.rotations[safe], offsets_world)
    mahalanobis = (local / gaussians.scales[safe]).square().sum(-1)
    influences = gaussians.opacities[safe] * torch.exp(-0.5 * mahalanobis)
    return Anchors(
        gaussian_indices=indices,
        local_offsets=torch.where(background[:, None], points, local),
        influences=torch.where(background, torch.zeros_like(influences), influences),
    )


def sample_map(values: Tensor, pixels: Tensor) -> Tensor:
    """Bilinearly sample an ``(H, W)`` map at ``(Q, 2)`` pixel positions."""

    height, width = values.shape
    u = pixels[:, 0].clamp(0, width - 1)
    v = pixels[:, 1].clamp(0, height - 1)
    u0 = u.floor().long()
    v0 = v.floor().long()
    u1 = (u0 + 1).clamp(max=width - 1)
    v1 = (v0 + 1).clamp(max=height - 1)
    du = u - u0
    dv = v - v0
    return (
        values[v0, u0] * (1 - du) * (1 - dv)
        + values[v0, u1] * du * (1 - dv)
        + values[v1, u0] * (1 - du) * dv
        + values[v1, u1] * du * dv
    )


__all__ = [
    "Anchors",
    "BACKGROUND_INFLUENCE_THRESHOLD",
    "FrameGaussians",
    "NO_GAUSSIAN",
    "ReferenceGaussians",
    "anchor_points",
    "anchor_to_gaussians",
    "compact_anchors",
    "frame_subset",
    "max_influence",
    "median_depth_gaussians",
    "project_points",
    "propagate",
    "render_depth",
    "sample_map",
    "track_points",
    "unproject_pixels",
]
