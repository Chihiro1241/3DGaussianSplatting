"""Official 3D Gaussian Splatting PLY interchange."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from gaussian_splatting.model.gaussian_model import GaussianModel


_BASE_PROPERTIES = ["x", "y", "z", "nx", "ny", "nz"]
_DC_PROPERTIES = [f"f_dc_{index}" for index in range(3)]
_REST_PROPERTIES = [f"f_rest_{index}" for index in range(45)]
_TAIL_PROPERTIES = [
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]
_ALL_PROPERTIES = _BASE_PROPERTIES + _DC_PROPERTIES + _REST_PROPERTIES + _TAIL_PROPERTIES


def _model_arrays(model: GaussianModel) -> dict[str, np.ndarray]:
    tensors = {
        "means": model.means_world.detach().cpu().to(torch.float32).numpy(),
        "sh_dc": model.sh_dc.detach().cpu().to(torch.float32).transpose(1, 2).flatten(1).numpy(),
        "sh_rest": model.sh_rest.detach().cpu().to(torch.float32).transpose(1, 2).flatten(1).numpy(),
        "opacity": model.raw_opacities.detach().cpu().to(torch.float32).numpy(),
        "scales": model.raw_scales.detach().cpu().to(torch.float32).numpy(),
        "rotations": model.raw_quaternions.detach().cpu().to(torch.float32).numpy(),
    }
    if tensors["sh_dc"].shape[1] != 3 or tensors["sh_rest"].shape[1] != 45:
        raise ValueError("official degree-3 PLY output requires 3 DC and 45 rest values")
    for name, values in tensors.items():
        if not np.isfinite(values).all():
            raise ValueError(f"model field {name} contains NaN or Inf")
    return tensors


def save_gaussians_ply(
    path: str | Path,
    model: GaussianModel,
    *,
    text: bool = False,
) -> None:
    """Save raw model parameters using official implementation property names."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _model_arrays(model)
    count = arrays["means"].shape[0]
    vertex = np.empty(count, dtype=[(name, "<f4") for name in _ALL_PROPERTIES])
    for component, name in enumerate(("x", "y", "z")):
        vertex[name] = arrays["means"][:, component]
    for name in ("nx", "ny", "nz"):
        vertex[name] = 0.0
    for index, name in enumerate(_DC_PROPERTIES):
        vertex[name] = arrays["sh_dc"][:, index]
    for index, name in enumerate(_REST_PROPERTIES):
        vertex[name] = arrays["sh_rest"][:, index]
    vertex["opacity"] = arrays["opacity"][:, 0]
    for index in range(3):
        vertex[f"scale_{index}"] = arrays["scales"][:, index]
    for index in range(4):
        vertex[f"rot_{index}"] = arrays["rotations"][:, index]
    PlyData(
        [PlyElement.describe(vertex, "vertex")],
        text=text,
        byte_order="<",
    ).write(str(destination))


def _stack_properties(vertex: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def load_gaussians_ply(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    epsilon_q: float = 1.0e-8,
    sh_degree: int = 3,
) -> GaussianModel:
    """Restore raw parameters from an ASCII or binary official-format PLY."""

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    if sh_degree != 3:
        raise ValueError("the initial implementation only supports sh_degree=3")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PLY does not exist: {source}")
    ply = PlyData.read(str(source))
    if "vertex" not in ply:
        raise ValueError("PLY is missing the vertex element")
    vertex = ply["vertex"].data
    available = set(vertex.dtype.names or ())
    missing = [name for name in _ALL_PROPERTIES if name not in available]
    if missing:
        raise ValueError(f"PLY is missing required properties: {missing}")
    actual_dc = {name for name in available if name.startswith("f_dc_")}
    actual_rest = {name for name in available if name.startswith("f_rest_")}
    if actual_dc != set(_DC_PROPERTIES):
        raise ValueError(
            f"degree-3 PLY must contain exactly {_DC_PROPERTIES}; got {sorted(actual_dc)}"
        )
    if actual_rest != set(_REST_PROPERTIES):
        raise ValueError(
            "degree-3 PLY must contain exactly f_rest_0 through f_rest_44"
        )

    means = _stack_properties(vertex, ["x", "y", "z"])
    sh_dc_flat = _stack_properties(vertex, _DC_PROPERTIES)
    sh_rest_flat = _stack_properties(vertex, _REST_PROPERTIES)
    raw_opacities = _stack_properties(vertex, ["opacity"])
    raw_scales = _stack_properties(vertex, [f"scale_{i}" for i in range(3)])
    raw_quaternions = _stack_properties(vertex, [f"rot_{i}" for i in range(4)])
    named_arrays = {
        "means_world": means,
        "sh_dc": sh_dc_flat.reshape(-1, 3, 1).transpose(0, 2, 1),
        "sh_rest": sh_rest_flat.reshape(-1, 3, 15).transpose(0, 2, 1),
        "raw_opacities": raw_opacities,
        "raw_scales": raw_scales,
        "raw_quaternions": raw_quaternions,
    }
    for name, values in named_arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"PLY property group {name} contains NaN or Inf")

    tensor_args = {
        name: torch.as_tensor(values.copy(), device=device, dtype=dtype)
        for name, values in named_arrays.items()
    }
    return GaussianModel(
        means_world=tensor_args["means_world"],
        raw_quaternions=tensor_args["raw_quaternions"],
        raw_scales=tensor_args["raw_scales"],
        raw_opacities=tensor_args["raw_opacities"],
        sh_dc=tensor_args["sh_dc"],
        sh_rest=tensor_args["sh_rest"],
        epsilon_q=epsilon_q,
        sh_degree=sh_degree,
    )
