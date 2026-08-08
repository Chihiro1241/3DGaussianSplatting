from __future__ import annotations

import re
import inspect
from pathlib import Path

from gaussian_splatting.math.covariance import (
    covariance_2d,
    pack_symmetric_2d,
    safe_2d_covariance_determinant,
)
from gaussian_splatting.math.jacobians import (
    inverse_covariance_jacobian,
    recursive_color_opacity_jacobian,
)
from gaussian_splatting.math.transform import view_direction
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.rasterizer import (
    implementation_projected_opacity,
    rasterize_gaussians,
    stable_color_accumulation,
    stable_transmittance_accumulation,
)


ROOT = Path(__file__).resolve().parents[2]


def test_all_tex_equation_labels_are_traceable_from_source() -> None:
    tex_path = ROOT / "document" / "3DGS_定式化.tex"
    tex_source = tex_path.read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{(eq:[A-Za-z0-9_]+)\}", tex_source))
    assert len(labels) == 116

    python_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src").rglob("*.py"))
    )
    missing = sorted(label for label in labels if label not in python_source)
    assert missing == []


def test_documented_definition_and_intermediate_labels_are_on_required_targets() -> None:
    required_targets = {
        "eq:raw_gaussian_parameters": (GaussianModel,),
        "eq:camera_covariance_definition": (covariance_2d,),
        "eq:backward_view_direction": (view_direction,),
        "eq:pixel_color": (rasterize_gaussians, stable_color_accumulation),
        "eq:transmittance": (
            rasterize_gaussians,
            stable_transmittance_accumulation,
        ),
        "eq:pixel_opacity": (implementation_projected_opacity,),
        "eq:recursive_pixel_color": (recursive_color_opacity_jacobian,),
        "eq:screen_opacity_backward": (implementation_projected_opacity,),
        "eq:screen_covariance_inverse_components": (
            inverse_covariance_jacobian,
        ),
        "eq:screen_covariance_vectors": (pack_symmetric_2d,),
        "eq:pixel_displacement_backward": (implementation_projected_opacity,),
        "eq:gaussian_weight_backward": (implementation_projected_opacity,),
        "eq:screen_covariance_determinant": (
            safe_2d_covariance_determinant,
        ),
        "eq:inverse_covariance_vector": (pack_symmetric_2d,),
    }

    for label, targets in required_targets.items():
        for target in targets:
            assert label in inspect.getsource(target), (
                f"{label} must be traceable from {target.__module__}.{target.__name__}"
            )
