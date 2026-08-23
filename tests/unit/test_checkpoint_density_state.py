from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.io.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    model_from_checkpoint_state,
    read_checkpoint,
    save_checkpoint,
    validate_resume_config,
)
from gaussian_splatting.model.gaussian_model import (
    GAUSSIAN_PARAMETER_NAMES,
    GAUSSIAN_PARAMETER_SPECS,
    GaussianModel,
)
from gaussian_splatting.training.density_control import (
    ScreenSpaceDensityStatistics,
)
from gaussian_splatting.training.optimizer import (
    append_gaussian_parameters,
    create_optimizer,
)
from gaussian_splatting.training.schedules import PositionLearningRateScheduler


ROOT = Path(__file__).resolve().parents[2]


def _config() -> Config:
    return load_config(ROOT / "configs" / "default.yaml")


def _parameter_tensors(
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    offset: float = 0.0,
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for parameter_index, (name, shape_tail) in enumerate(
        GAUSSIAN_PARAMETER_SPECS
    ):
        row_size = math.prod(shape_tail)
        values = torch.arange(
            count * row_size,
            dtype=dtype,
        ).reshape(count, *shape_tail)
        tensors[name] = values + offset + 100.0 * parameter_index
    return tensors


def _model(
    count: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
) -> GaussianModel:
    return GaussianModel(**_parameter_tensors(count, dtype=dtype))


def _training_objects(
    model: GaussianModel,
    config: Config,
) -> tuple[torch.optim.Adam, PositionLearningRateScheduler]:
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    return optimizer, scheduler


def _initialize_adam_state(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()


def _save(
    path: Path,
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    scheduler: PositionLearningRateScheduler,
    *,
    config: Config,
    statistics: ScreenSpaceDensityStatistics | None = None,
    iteration: int = 17,
) -> dict[str, Any]:
    return save_checkpoint(
        path,
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        camera_order=[2, 0, 1],
        camera_cursor=1,
        best_mean_psnr=23.5,
        density_statistics=statistics,
    )


def _raw_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _raw_config_mapping(config: Config) -> dict[str, Any]:
    return asdict(config)


def _optimizer_state_by_name(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
) -> dict[str, dict[str, object]]:
    return {
        name: optimizer.state[getattr(model, name)]
        for name in GAUSSIAN_PARAMETER_NAMES
    }


def test_version_2_save_and_read_records_optional_statistics(tmp_path: Path) -> None:
    config = _config()
    model = _model()
    optimizer, scheduler = _training_objects(model, config)
    path = tmp_path / "without_statistics.pt"

    saved = _save(path, model, optimizer, scheduler, config=config)
    read = read_checkpoint(path, map_location="cpu")

    assert CHECKPOINT_VERSION == 2
    assert saved["checkpoint_version"] == 2
    assert saved["density_statistics_state"] is None
    assert read["checkpoint_version"] == 2
    assert read["density_statistics_state"] is None


def test_version_2_requires_density_statistics_state_key(tmp_path: Path) -> None:
    config = _config()
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    path = tmp_path / "missing_statistics_key.pt"
    _save(path, model, optimizer, scheduler, config=config)
    state = _raw_checkpoint(path)
    del state["density_statistics_state"]
    torch.save(state, path)

    with pytest.raises(ValueError, match="density_statistics_state"):
        read_checkpoint(path, map_location="cpu")


def test_save_rejects_incompatible_statistics_before_writing(tmp_path: Path) -> None:
    config = _config()
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    statistics = ScreenSpaceDensityStatistics(
        3,
        dtype=model.means_world.dtype,
        device=model.means_world.device,
    )
    path = tmp_path / "incompatible.pt"

    with pytest.raises(ValueError, match="Gaussian count"):
        _save(
            path,
            model,
            optimizer,
            scheduler,
            config=config,
            statistics=statistics,
        )

    assert not path.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dynamic_model_optimizer_and_mid_window_statistics_round_trip(
    tmp_path: Path,
    dtype: torch.dtype,
) -> None:
    config = _config()
    model = _model(4, dtype=dtype)
    optimizer, scheduler = _training_objects(model, config)
    _initialize_adam_state(model, optimizer)
    append_gaussian_parameters(
        model,
        optimizer,
        _parameter_tensors(2, dtype=dtype, offset=10_000.0),
    )
    scheduler.step(23)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.position_gradient_accumulator.copy_(
        torch.tensor([1, 2, 3, 4, 5, 6], dtype=dtype)
    )
    statistics.position_gradient_denominator.copy_(
        torch.tensor([1, 1, 2, 2, 3, 3])
    )
    statistics.max_screen_radius.copy_(torch.tensor([2, 4, 6, 8, 10, 12]))
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = {
        name: {
            key: value.detach().clone() if isinstance(value, Tensor) else value
            for key, value in state.items()
        }
        for name, state in _optimizer_state_by_name(model, optimizer).items()
    }
    expected_statistics = statistics.state_dict()
    path = tmp_path / f"dynamic_{dtype}.pt"

    _save(
        path,
        model,
        optimizer,
        scheduler,
        config=config,
        statistics=statistics,
        iteration=23,
    )
    state = read_checkpoint(path, map_location="cpu")
    restored_model = model_from_checkpoint_state(
        state,
        config,
        device="cpu",
        dtype=dtype,
    )
    restored_optimizer, restored_scheduler = _training_objects(
        restored_model, config
    )
    restored_statistics = ScreenSpaceDensityStatistics.for_model(restored_model)
    load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        density_statistics=restored_statistics,
        map_location="cpu",
        restore_random_state=False,
    )

    assert model.num_gaussians == 6
    assert restored_model.num_gaussians == 6
    for name, expected in expected_model.items():
        torch.testing.assert_close(
            restored_model.state_dict()[name], expected, rtol=0.0, atol=0.0
        )
    restored_optimizer_states = _optimizer_state_by_name(
        restored_model, restored_optimizer
    )
    for name, expected_state in expected_optimizer.items():
        actual_state = restored_optimizer_states[name]
        for key, expected in expected_state.items():
            actual = actual_state[key]
            if isinstance(expected, Tensor):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            else:
                assert actual == expected
        assert actual_state["exp_avg"].shape == getattr(restored_model, name).shape
        assert actual_state["exp_avg_sq"].shape == getattr(restored_model, name).shape
    assert restored_scheduler.current_iteration == 23
    for name, expected in expected_statistics.items():
        torch.testing.assert_close(
            getattr(restored_statistics, name), expected, rtol=0.0, atol=0.0
        )


def test_checkpoint_statistics_presence_must_match_runtime_caller(
    tmp_path: Path,
) -> None:
    config = _config()
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    with_statistics = tmp_path / "with_statistics.pt"
    without_statistics = tmp_path / "without_statistics.pt"
    _save(
        with_statistics,
        model,
        optimizer,
        scheduler,
        config=config,
        statistics=statistics,
    )
    _save(without_statistics, model, optimizer, scheduler, config=config)
    before = model.means_world.detach().clone()

    with pytest.raises(ValueError, match="no runtime density_statistics"):
        load_checkpoint(with_statistics, model=model, map_location="cpu")
    with pytest.raises(ValueError, match="checkpoint contains none"):
        load_checkpoint(
            without_statistics,
            model=model,
            density_statistics=statistics,
            map_location="cpu",
        )

    torch.testing.assert_close(model.means_world, before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("corruption", ["shape", "dtype"])
def test_invalid_checkpoint_statistics_fail_before_runtime_mutation(
    tmp_path: Path,
    corruption: str,
) -> None:
    config = _config()
    source_model = _model(2)
    source_optimizer, source_scheduler = _training_objects(source_model, config)
    source_statistics = ScreenSpaceDensityStatistics.for_model(source_model)
    path = tmp_path / f"invalid_{corruption}.pt"
    _save(
        path,
        source_model,
        source_optimizer,
        source_scheduler,
        config=config,
        statistics=source_statistics,
    )
    state = _raw_checkpoint(path)
    if corruption == "shape":
        state["density_statistics_state"][
            "position_gradient_accumulator"
        ] = torch.zeros(3)
    else:
        state["density_statistics_state"][
            "position_gradient_accumulator"
        ] = torch.zeros(2, dtype=torch.float64)
    torch.save(state, path)

    runtime_model = _model(2)
    runtime_optimizer, runtime_scheduler = _training_objects(runtime_model, config)
    runtime_scheduler.step(9)
    runtime_statistics = ScreenSpaceDensityStatistics.for_model(runtime_model)
    with torch.no_grad():
        runtime_model.means_world.add_(50.0)
        runtime_statistics.position_gradient_accumulator.fill_(7.0)
    model_before = {
        name: value.detach().clone()
        for name, value in runtime_model.state_dict().items()
    }
    statistics_before = runtime_statistics.state_dict()
    torch.manual_seed(44)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises((TypeError, ValueError), match=corruption):
        load_checkpoint(
            path,
            model=runtime_model,
            optimizer=runtime_optimizer,
            scheduler=runtime_scheduler,
            density_statistics=runtime_statistics,
            map_location="cpu",
        )

    for name, expected in model_before.items():
        torch.testing.assert_close(
            runtime_model.state_dict()[name], expected, rtol=0.0, atol=0.0
        )
    for name, expected in statistics_before.items():
        torch.testing.assert_close(
            getattr(runtime_statistics, name), expected, rtol=0.0, atol=0.0
        )
    assert len(runtime_optimizer.state) == 0
    assert runtime_scheduler.current_iteration == 9
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "non_tensor"])
def test_read_checkpoint_rejects_invalid_statistics_structure(
    tmp_path: Path,
    corruption: str,
) -> None:
    config = _config()
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    path = tmp_path / f"invalid_structure_{corruption}.pt"
    _save(
        path,
        model,
        optimizer,
        scheduler,
        config=config,
        statistics=statistics,
    )
    state = _raw_checkpoint(path)
    if corruption == "missing":
        del state["density_statistics_state"]["max_screen_radius"]
    elif corruption == "unexpected":
        state["density_statistics_state"]["unexpected"] = torch.zeros(2)
    else:
        state["density_statistics_state"]["max_screen_radius"] = [0, 0]
    torch.save(state, path)

    with pytest.raises((TypeError, ValueError), match="density_statistics_state"):
        read_checkpoint(path, map_location="cpu")


def test_zero_gaussian_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _config()
    model = _model(0)
    optimizer, scheduler = _training_objects(model, config)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    path = tmp_path / "zero_gaussians.pt"
    _save(
        path,
        model,
        optimizer,
        scheduler,
        config=config,
        statistics=statistics,
    )

    state = read_checkpoint(path, map_location="cpu")
    restored_model = model_from_checkpoint_state(
        state, config, device="cpu", dtype=torch.float32
    )
    restored_optimizer, restored_scheduler = _training_objects(
        restored_model, config
    )
    restored_statistics = ScreenSpaceDensityStatistics.for_model(restored_model)
    load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        density_statistics=restored_statistics,
        map_location="cpu",
    )

    assert restored_model.num_gaussians == 0
    assert restored_statistics.num_gaussians == 0
    assert restored_statistics.position_gradient_accumulator.shape == (0,)
    assert restored_statistics.position_gradient_denominator.shape == (0,)
    assert restored_statistics.max_screen_radius.shape == (0,)


def test_statistics_checkpoint_preserves_all_rng_restore_semantics(
    tmp_path: Path,
) -> None:
    config = _config()
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    statistics.position_gradient_accumulator.copy_(torch.tensor([1.0, 2.0]))
    statistics.position_gradient_denominator.copy_(torch.tensor([3, 4]))
    statistics.max_screen_radius.copy_(torch.tensor([5, 6]))
    path = tmp_path / "rng.pt"
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(44)
    _save(
        path,
        model,
        optimizer,
        scheduler,
        config=config,
        statistics=statistics,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(())
    expected_cuda = (
        torch.rand((), device="cuda") if torch.cuda.is_available() else None
    )
    random.random()
    np.random.rand()
    torch.rand(())
    if torch.cuda.is_available():
        torch.rand((), device="cuda")
    restored_statistics = ScreenSpaceDensityStatistics.for_model(model)

    load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        density_statistics=restored_statistics,
        map_location="cpu",
        restore_random_state=True,
    )

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(()), expected_torch, rtol=0.0, atol=0.0)
    if expected_cuda is not None:
        torch.testing.assert_close(
            torch.rand((), device="cuda"), expected_cuda, rtol=0.0, atol=0.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mapped_checkpoint_restores_cuda_rng_state(tmp_path: Path) -> None:
    config = _config()
    model = _model(2).to("cuda")
    optimizer, scheduler = _training_objects(model, config)
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    path = tmp_path / "cuda_checkpoint.pt"
    _save(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        statistics=statistics,
        iteration=1,
    )

    load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        density_statistics=statistics,
        map_location="cuda",
        restore_random_state=True,
    )


def test_legacy_version_1_fixed_checkpoint_and_config_are_supported(
    tmp_path: Path,
) -> None:
    base_config = _config()
    config = replace(
        base_config,
        features=replace(
            base_config.features,
            adaptive_density_control=False,
            opacity_reset=False,
        ),
    )
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    path = tmp_path / "legacy.pt"
    _save(path, model, optimizer, scheduler, config=config)
    legacy = _raw_checkpoint(path)
    legacy.pop("checkpoint_version")
    legacy.pop("density_statistics_state")
    legacy["config"].pop("density_control")
    torch.save(legacy, path)

    normalized = read_checkpoint(path, map_location="cpu")
    validate_resume_config(normalized["config"], config)
    restored_model = _model(2)
    load_checkpoint(path, model=restored_model, map_location="cpu")

    assert normalized["checkpoint_version"] == 1
    assert normalized["density_statistics_state"] is None
    assert "checkpoint_version" not in _raw_checkpoint(path)
    assert "density_statistics_state" not in _raw_checkpoint(path)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            restored_model.state_dict()[name], expected, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("feature", ["adaptive_density_control", "opacity_reset"])
def test_legacy_checkpoint_cannot_enable_density_features(
    feature: str,
) -> None:
    config = _config()
    saved = _raw_config_mapping(config)
    saved.pop("density_control")
    current = replace(
        config,
        features=replace(config.features, **{feature: True}),
    )

    with pytest.raises(ValueError, match="resume configuration differs"):
        validate_resume_config(saved, current)

def test_version_2_density_config_remains_strict() -> None:
    config = _config()
    saved = _raw_config_mapping(config)
    current = replace(
        config,
        density_control=replace(config.density_control, percent_dense=0.02),
    )

    with pytest.raises(ValueError, match="resume configuration differs"):
        validate_resume_config(saved, current)


@pytest.mark.parametrize(
    ("version", "exception"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("2", TypeError),
        (-1, ValueError),
        (0, ValueError),
        (999, ValueError),
    ],
)
def test_read_checkpoint_rejects_invalid_or_unsupported_versions(
    tmp_path: Path,
    version: object,
    exception: type[Exception],
) -> None:
    config = _config()
    model = _model(2)
    optimizer, scheduler = _training_objects(model, config)
    path = tmp_path / f"version_{version!s}.pt"
    _save(path, model, optimizer, scheduler, config=config)
    state = _raw_checkpoint(path)
    state["checkpoint_version"] = version
    torch.save(state, path)

    with pytest.raises(exception, match="checkpoint_version"):
        read_checkpoint(path, map_location="cpu")
