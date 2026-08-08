"""Mathematical building blocks for the reference 3DGS implementation."""

from .covariance import (
    covariance_2d,
    covariance_2d_components,
    covariance_3d,
    inverse_2d_covariance,
    pack_symmetric_2d,
    pack_symmetric_3d,
    quaternion_rotation_matrix,
    safe_2d_covariance_determinant,
    scale_matrix,
    stabilized_2d_covariance,
    stabilized_inverse_2d_covariance,
    symmetrized_2d_covariance,
    unpack_symmetric_2d,
    unpack_symmetric_3d,
)
from .parameterization import raw_parameter_transformations
from .spherical_harmonics import (
    real_sh_degree_3,
    sh_color_implementation,
    sh_color_theory,
)
from .transform import (
    perspective_projection,
    projection_jacobian,
    view_direction,
    world_to_camera,
)

__all__ = [
    "covariance_2d",
    "covariance_2d_components",
    "covariance_3d",
    "inverse_2d_covariance",
    "pack_symmetric_2d",
    "pack_symmetric_3d",
    "perspective_projection",
    "projection_jacobian",
    "quaternion_rotation_matrix",
    "raw_parameter_transformations",
    "real_sh_degree_3",
    "safe_2d_covariance_determinant",
    "scale_matrix",
    "sh_color_implementation",
    "sh_color_theory",
    "stabilized_2d_covariance",
    "stabilized_inverse_2d_covariance",
    "symmetrized_2d_covariance",
    "unpack_symmetric_2d",
    "unpack_symmetric_3d",
    "view_direction",
    "world_to_camera",
]
