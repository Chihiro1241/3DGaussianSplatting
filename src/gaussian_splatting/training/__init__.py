"""Training utilities for Gaussian Splatting."""

from gaussian_splatting.training.density_control import (
    GaussianPruneResult,
    ScreenSpaceDensityStatistics,
    prune_gaussians,
)
from gaussian_splatting.training.losses import LossResult, total_loss
from gaussian_splatting.training.optimizer import (
    append_gaussian_parameters,
    create_optimizer,
    keep_gaussian_parameters,
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
    "GaussianPruneResult",
    "PositionLearningRateScheduler",
    "ScreenSpaceDensityStatistics",
    "EvaluationResult",
    "Trainer",
    "TrainStepResult",
    "append_gaussian_parameters",
    "camera_to",
    "create_optimizer",
    "keep_gaussian_parameters",
    "position_learning_rate_schedule",
    "prune_gaussians",
    "total_loss",
]
