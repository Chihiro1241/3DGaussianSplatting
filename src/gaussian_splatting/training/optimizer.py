"""Adam parameter groups for a Gaussian model."""

from __future__ import annotations

import torch

from gaussian_splatting.config import Config, TrainingConfig
from gaussian_splatting.model.gaussian_model import GaussianModel


def create_optimizer(
    model: GaussianModel,
    config: Config | TrainingConfig,
) -> torch.optim.Adam:
    """Create the six named Adam parameter groups required by the design.

    TeX: eq:gradient_descent_update (implemented by ``Adam.step``).
    """

    training = config.training if isinstance(config, Config) else config
    if training.optimizer != "adam":
        raise ValueError("the initial implementation only supports optimizer='adam'")
    parameter_groups = [
        {"params": [model.means_world], "lr": training.position_lr_initial, "name": "means_world"},
        {"params": [model.sh_dc], "lr": training.sh_dc_lr, "name": "sh_dc"},
        {"params": [model.sh_rest], "lr": training.sh_rest_lr, "name": "sh_rest"},
        {"params": [model.raw_opacities], "lr": training.opacity_lr, "name": "raw_opacities"},
        {"params": [model.raw_scales], "lr": training.scale_lr, "name": "raw_scales"},
        {"params": [model.raw_quaternions], "lr": training.quaternion_lr, "name": "raw_quaternions"},
    ]
    return torch.optim.Adam(
        parameter_groups,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_epsilon,
        weight_decay=training.weight_decay,
    )

