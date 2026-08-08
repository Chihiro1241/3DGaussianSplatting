"""Learning-rate schedules used by Gaussian Splatting training."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer


def position_learning_rate_schedule(
    iteration: int,
    total_iterations: int = 30_000,
    initial_learning_rate: float = 1.6e-4,
    final_learning_rate: float = 1.6e-6,
) -> float:
    """Return the exponentially interpolated position learning rate.

    TeX: eq:position_learning_rate_schedule
    """

    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if initial_learning_rate <= 0.0 or final_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    clamped_iteration = min(max(int(iteration), 0), int(total_iterations))
    fraction = clamped_iteration / total_iterations
    return math.exp(
        (1.0 - fraction) * math.log(initial_learning_rate)
        + fraction * math.log(final_learning_rate)
    )


class PositionLearningRateScheduler:
    """Update only the optimizer's ``means_world`` parameter group."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_iterations: int = 30_000,
        initial_learning_rate: float = 1.6e-4,
        final_learning_rate: float = 1.6e-6,
    ) -> None:
        self.optimizer = optimizer
        self.total_iterations = total_iterations
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.current_iteration = 0
        self._position_group()

    def _position_group(self) -> dict[str, Any]:
        groups = [g for g in self.optimizer.param_groups if g.get("name") == "means_world"]
        if len(groups) != 1:
            raise ValueError("optimizer must contain exactly one 'means_world' group")
        return groups[0]

    def step(self, iteration: int) -> float:
        """Set and return the absolute position learning rate for an iteration."""

        self.current_iteration = min(max(int(iteration), 0), self.total_iterations)
        learning_rate = position_learning_rate_schedule(
            self.current_iteration,
            self.total_iterations,
            self.initial_learning_rate,
            self.final_learning_rate,
        )
        self._position_group()["lr"] = learning_rate
        return learning_rate

    def state_dict(self) -> dict[str, int | float]:
        return {
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
        }

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        required = {
            "current_iteration",
            "total_iterations",
            "initial_learning_rate",
            "final_learning_rate",
        }
        missing = required.difference(state_dict)
        if missing:
            raise ValueError(f"scheduler state is missing keys: {sorted(missing)}")
        if int(state_dict["total_iterations"]) != self.total_iterations:
            raise ValueError("scheduler total_iterations does not match")
        if float(state_dict["initial_learning_rate"]) != self.initial_learning_rate:
            raise ValueError("scheduler initial learning rate does not match")
        if float(state_dict["final_learning_rate"]) != self.final_learning_rate:
            raise ValueError("scheduler final learning rate does not match")
        self.step(int(state_dict["current_iteration"]))

