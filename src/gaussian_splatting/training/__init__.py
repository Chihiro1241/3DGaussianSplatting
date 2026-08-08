"""Training utilities for Gaussian Splatting."""

from gaussian_splatting.training.losses import LossResult, total_loss
from gaussian_splatting.training.optimizer import create_optimizer
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
    "PositionLearningRateScheduler",
    "EvaluationResult",
    "Trainer",
    "TrainStepResult",
    "camera_to",
    "create_optimizer",
    "position_learning_rate_schedule",
    "total_loss",
]
