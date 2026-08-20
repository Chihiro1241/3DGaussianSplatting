"""Training utilities for Gaussian Splatting."""

from gaussian_splatting.training.density_control import (
    GaussianCloneResult,
    GaussianDensityControlResult,
    GaussianOpacityResetResult,
    GaussianPruneResult,
    GaussianSplitResult,
    ScreenSpaceDensityStatistics,
    clone_gaussians,
    prune_gaussians,
    reset_gaussian_opacity,
    run_density_control_event,
    split_gaussians,
)
from gaussian_splatting.training.losses import LossResult, total_loss
from gaussian_splatting.training.optimizer import (
    append_gaussian_parameters,
    create_optimizer,
    keep_and_append_gaussian_parameters,
    keep_gaussian_parameters,
    replace_named_gaussian_parameter,
)
from gaussian_splatting.training.schedules import (
    PositionLearningRateScheduler,
    position_learning_rate_schedule,
)
from gaussian_splatting.training.trainer import (
    EvaluationResult,
    Trainer,
    TrainStepResult,
    camera_to,
)

__all__ = [
    "LossResult",
    "GaussianCloneResult",
    "GaussianDensityControlResult",
    "GaussianOpacityResetResult",
    "GaussianPruneResult",
    "GaussianSplitResult",
    "PositionLearningRateScheduler",
    "ScreenSpaceDensityStatistics",
    "EvaluationResult",
    "Trainer",
    "TrainStepResult",
    "append_gaussian_parameters",
    "camera_to",
    "clone_gaussians",
    "create_optimizer",
    "keep_gaussian_parameters",
    "keep_and_append_gaussian_parameters",
    "position_learning_rate_schedule",
    "prune_gaussians",
    "replace_named_gaussian_parameter",
    "reset_gaussian_opacity",
    "run_density_control_event",
    "split_gaussians",
    "total_loss",
]
