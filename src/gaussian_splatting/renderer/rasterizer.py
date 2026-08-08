"""Reference PyTorch Gaussian rasterizer.

This implementation deliberately uses a Python loop over Gaussians while
vectorizing every pixel operation within a Gaussian's rendering rectangle.
It is intended for equation validation and small integration tests.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gaussian_splatting.renderer.projection import (
    ProjectedGaussians,
    pixel_coordinate_convention,
)


def _require_float_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")


def _require_same_dtype_device(reference: Tensor, **values: Tensor) -> None:
    for name, value in values.items():
        _require_float_tensor(name, value)
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} must have dtype {reference.dtype}")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on device {reference.device}")


def implementation_projected_opacity(
    opacity: Tensor,
    pixel_coordinates: Tensor,
    mean_screen: Tensor,
    inverse_covariance_2d: Tensor,
) -> Tensor:
    """Evaluate one projected Gaussian's opacity at one or more pixels.

    ``pixel_coordinates`` may have any leading dimensions and must end in
    ``2``.  The other geometric inputs describe one Gaussian.

    TeX: eq:implementation_projected_opacity
    TeX: eq:pixel_opacity
    TeX: eq:screen_opacity_backward
    TeX: eq:pixel_displacement_backward
    TeX: eq:gaussian_weight_backward
    """

    _require_float_tensor("opacity", opacity)
    _require_same_dtype_device(
        opacity,
        pixel_coordinates=pixel_coordinates,
        mean_screen=mean_screen,
        inverse_covariance_2d=inverse_covariance_2d,
    )
    if opacity.numel() != 1:
        raise ValueError(f"opacity must contain one value, got shape {tuple(opacity.shape)}")
    if pixel_coordinates.ndim < 1 or pixel_coordinates.shape[-1] != 2:
        raise ValueError("pixel_coordinates must end with shape (2,)")
    if mean_screen.shape != (2,):
        raise ValueError(f"mean_screen must have shape (2,), got {tuple(mean_screen.shape)}")
    if inverse_covariance_2d.shape != (2, 2):
        raise ValueError(
            "inverse_covariance_2d must have shape (2, 2), "
            f"got {tuple(inverse_covariance_2d.shape)}"
        )

    # TeX aliases: eq:pixel_displacement_backward and
    # eq:screen_opacity_backward.
    displacement = pixel_coordinates - mean_screen
    mahalanobis_squared = (
        torch.matmul(displacement, inverse_covariance_2d) * displacement
    ).sum(dim=-1)
    # TeX alias: eq:gaussian_weight_backward.
    gaussian_weight = torch.exp(-0.5 * mahalanobis_squared)
    return opacity.reshape(()) * gaussian_weight


def clamped_projected_opacity(
    projected_opacity: Tensor,
    alpha_max: float = 0.99,
) -> Tensor:
    """Limit projected opacity from above before compositing.

    TeX: eq:clamped_projected_opacity
    """

    _require_float_tensor("projected_opacity", projected_opacity)
    if not 0.0 < alpha_max < 1.0:
        raise ValueError("alpha_max must lie strictly between zero and one")
    return torch.clamp_max(projected_opacity, float(alpha_max))


def opacity_contribution_threshold(
    clamped_opacity: Tensor,
    alpha_min: float = 1.0 / 255.0,
) -> Tensor:
    """Replace projected opacity below the contribution threshold with zero.

    TeX: eq:opacity_contribution_threshold
    """

    _require_float_tensor("clamped_opacity", clamped_opacity)
    if alpha_min < 0.0:
        raise ValueError("alpha_min must be non-negative")
    contributes = clamped_opacity.detach() >= float(alpha_min)
    return torch.where(contributes, clamped_opacity, torch.zeros_like(clamped_opacity))


def stable_transmittance_accumulation(
    transmittance: Tensor,
    thresholded_opacity: Tensor,
) -> Tensor:
    """Compute the candidate transmittance after one Gaussian.

    TeX: eq:stable_transmittance_accumulation
    TeX: eq:transmittance
    """

    _require_float_tensor("transmittance", transmittance)
    _require_same_dtype_device(
        transmittance,
        thresholded_opacity=thresholded_opacity,
    )
    if transmittance.shape != thresholded_opacity.shape:
        raise ValueError("transmittance and thresholded_opacity must have equal shape")
    return transmittance * (1.0 - thresholded_opacity)


def transmittance_termination(
    test_transmittance: Tensor,
    transmittance_min: float = 1e-4,
) -> Tensor:
    """Return where candidate transmittance requires early termination.

    TeX: eq:transmittance_termination
    """

    _require_float_tensor("test_transmittance", test_transmittance)
    if transmittance_min < 0.0:
        raise ValueError("transmittance_min must be non-negative")
    return test_transmittance.detach() < float(transmittance_min)


def stable_color_accumulation(
    accumulated_color: Tensor,
    transmittance: Tensor,
    thresholded_opacity: Tensor,
    color: Tensor,
) -> Tensor:
    """Accumulate one non-terminated Gaussian's color contribution.

    The caller applies the discrete termination mask; this function contains
    exactly the differentiable branch of the accumulation equation.

    TeX: eq:stable_color_accumulation
    TeX: eq:pixel_color
    """

    _require_float_tensor("accumulated_color", accumulated_color)
    _require_same_dtype_device(
        accumulated_color,
        transmittance=transmittance,
        thresholded_opacity=thresholded_opacity,
        color=color,
    )
    if accumulated_color.ndim < 1 or accumulated_color.shape[-1] != 3:
        raise ValueError("accumulated_color must end with shape (3,)")
    if transmittance.shape != accumulated_color.shape[:-1]:
        raise ValueError("transmittance must match accumulated_color's leading shape")
    if thresholded_opacity.shape != transmittance.shape:
        raise ValueError("thresholded_opacity and transmittance must have equal shape")
    if color.shape != (3,):
        raise ValueError(f"color must have shape (3,), got {tuple(color.shape)}")
    contribution = (
        transmittance * thresholded_opacity
    ).unsqueeze(-1) * color
    return accumulated_color + contribution


def implementation_pixel_color_with_background(
    accumulated_color: Tensor,
    final_transmittance: Tensor,
    background: Tensor,
) -> Tensor:
    """Composite the remaining transmittance with the background color.

    TeX: eq:implementation_pixel_color_with_background
    """

    _require_float_tensor("accumulated_color", accumulated_color)
    _require_same_dtype_device(
        accumulated_color,
        final_transmittance=final_transmittance,
        background=background,
    )
    if accumulated_color.ndim < 1 or accumulated_color.shape[-1] != 3:
        raise ValueError("accumulated_color must end with shape (3,)")
    if final_transmittance.shape != accumulated_color.shape[:-1]:
        raise ValueError("final_transmittance must match accumulated_color's leading shape")
    if background.shape != (3,):
        raise ValueError(f"background must have shape (3,), got {tuple(background.shape)}")
    return accumulated_color + final_transmittance.unsqueeze(-1) * background


def _validate_projected_for_rasterization(projected: ProjectedGaussians) -> tuple[int, Tensor]:
    reference = projected.means_screen
    _require_float_tensor("projected.means_screen", reference)
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("projected.means_screen must have shape (Nv, 2)")
    count = reference.shape[0]
    float_fields = {
        "projected.inverse_covariances_2d": (
            projected.inverse_covariances_2d,
            (count, 2, 2),
        ),
        "projected.colors": (projected.colors, (count, 3)),
        "projected.opacities": (projected.opacities, (count, 1)),
    }
    for name, (value, expected_shape) in float_fields.items():
        _require_same_dtype_device(reference, **{name: value})
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("projected.means_screen contains NaN or Inf")
    if projected.rectangles.shape != (count, 4):
        raise ValueError("projected.rectangles must have shape (Nv, 4)")
    if projected.rectangles.dtype != torch.int64:
        raise TypeError("projected.rectangles must have dtype torch.int64")
    if projected.rectangles.device != reference.device:
        raise ValueError("projected.rectangles must be on the projection device")
    return count, reference


def rasterize_gaussians(
    projected: ProjectedGaussians,
    height: int,
    width: int,
    background: Tensor,
    alpha_max: float = 0.99,
    alpha_min: float = 1.0 / 255.0,
    transmittance_min: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Rasterize depth-ordered Gaussians by vectorized rectangle updates.

    Returns an image of shape ``(3, H, W)`` and the accepted final
    transmittance of shape ``(H, W)``.  Pixels whose *candidate*
    transmittance falls below ``transmittance_min`` accept neither that
    Gaussian's color nor its candidate transmittance, and remain inactive for
    every later Gaussian.

    TeX: eq:pixel_color
    TeX: eq:transmittance
    """

    count, reference = _validate_projected_for_rasterization(projected)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    _require_same_dtype_device(reference, background=background)
    if background.shape != (3,):
        raise ValueError(f"background must have shape (3,), got {tuple(background.shape)}")
    if not bool(torch.isfinite(background).all()):
        raise ValueError("background contains NaN or Inf")

    pixel_coordinates = pixel_coordinate_convention(
        height,
        width,
        dtype=reference.dtype,
        device=reference.device,
    )
    accumulated_color = reference.new_zeros((height, width, 3))
    transmittance = reference.new_ones((height, width))
    active = torch.ones((height, width), dtype=torch.bool, device=reference.device)

    for gaussian_index in range(count):
        xmin, xmax, ymin, ymax = (
            int(value)
            for value in projected.rectangles[gaussian_index].detach().cpu().tolist()
        )
        if xmin > xmax or ymin > ymax:
            continue
        if xmin < 0 or ymin < 0 or xmax >= width or ymax >= height:
            raise ValueError("projected rectangle lies outside the image")

        region = (slice(ymin, ymax + 1), slice(xmin, xmax + 1))
        region_active = active[region]
        if not bool(region_active.any()):
            continue

        projected_opacity = implementation_projected_opacity(
            projected.opacities[gaussian_index],
            pixel_coordinates[region],
            projected.means_screen[gaussian_index],
            projected.inverse_covariances_2d[gaussian_index],
        )
        clamped_opacity = clamped_projected_opacity(projected_opacity, alpha_max)
        thresholded_opacity = opacity_contribution_threshold(clamped_opacity, alpha_min)

        # Clone only the rectangle values that subsequent in-place indexed
        # writes would otherwise invalidate for autograd.  This keeps the
        # Gaussian loop practical without cloning the full image per splat.
        previous_transmittance = transmittance[region].clone()
        previous_color = accumulated_color[region].clone()
        test_transmittance = stable_transmittance_accumulation(
            previous_transmittance,
            thresholded_opacity,
        )
        terminates = region_active & transmittance_termination(
            test_transmittance,
            transmittance_min,
        )
        accepts = region_active & ~terminates

        candidate_color = stable_color_accumulation(
            previous_color,
            previous_transmittance,
            thresholded_opacity,
            projected.colors[gaussian_index],
        )
        updated_color_region = torch.where(
            accepts.unsqueeze(-1),
            candidate_color,
            previous_color,
        )
        updated_transmittance_region = torch.where(
            accepts,
            test_transmittance,
            previous_transmittance,
        )

        accumulated_color[region] = updated_color_region
        transmittance[region] = updated_transmittance_region
        active[region] = region_active & ~terminates

    image_hwc = implementation_pixel_color_with_background(
        accumulated_color,
        transmittance,
        background,
    )
    image = image_hwc.permute(2, 0, 1).contiguous()
    if not bool(torch.isfinite(image).all()):
        raise ValueError("rendered image contains NaN or Inf")
    return image, transmittance
