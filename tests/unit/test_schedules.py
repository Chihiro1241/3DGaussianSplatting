from __future__ import annotations

import math

import torch

from gaussian_splatting.training.schedules import (
    PositionLearningRateScheduler,
    position_learning_rate_schedule,
)


def test_position_learning_rate_schedule__eq_position_learning_rate_schedule() -> None:
    initial = 1.6e-4
    final = 1.6e-6
    assert position_learning_rate_schedule(0) == initial
    assert math.isclose(position_learning_rate_schedule(30_000), final)
    assert math.isclose(position_learning_rate_schedule(15_000), math.sqrt(initial * final))


def test_position_learning_rate_schedule_clamps_iteration() -> None:
    assert position_learning_rate_schedule(-1) == position_learning_rate_schedule(0)
    assert position_learning_rate_schedule(30_001) == position_learning_rate_schedule(30_000)


def test_position_scheduler_round_trip() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([{"params": [parameter], "name": "means_world", "lr": 1.6e-4}])
    scheduler = PositionLearningRateScheduler(optimizer)
    expected = scheduler.step(123)
    state = scheduler.state_dict()

    restored = PositionLearningRateScheduler(optimizer)
    restored.load_state_dict(state)
    assert restored.current_iteration == 123
    assert optimizer.param_groups[0]["lr"] == expected

