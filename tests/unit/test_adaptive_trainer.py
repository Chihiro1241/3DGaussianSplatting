from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.density_control import (
    GaussianCloneResult,
    GaussianDensityControlResult,
    GaussianOpacityResetResult,
    GaussianPruneResult,
    GaussianSplitResult,
    ScreenSpaceDensityStatistics,
)
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer
import gaussian_splatting.training.trainer as trainer_module


ROOT = Path(__file__).resolve().parents[2]


def _config(
    *,
    adaptive_density_control: bool,
    opacity_reset: bool,
    iterations: int = 6,
    densify_from: int = 1,
    densify_until: int = 6,
    densification_interval: int = 2,
    gradient_threshold: float = 1.0e9,
    percent_dense: float = 1.0,
    prune_opacity_threshold: float = 0.0,
    opacity_reset_interval: int = 3,
) -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(
            config.training,
            iterations=iterations,
            log_interval=iterations,
            evaluation_interval=iterations,
            checkpoint_interval=iterations,
        ),
        density_control=replace(
            config.density_control,
            densify_from_iteration=densify_from,
            densify_until_iteration=densify_until,
            densification_interval=densification_interval,
            position_gradient_threshold=gradient_threshold,
            percent_dense=percent_dense,
            prune_opacity_threshold=prune_opacity_threshold,
            opacity_reset_interval=opacity_reset_interval,
        ),
        features=replace(
            config.features,
            adaptive_density_control=adaptive_density_control,
            opacity_reset=opacity_reset,
        ),
        output=replace(config.output, save_rendered_images=False),
    )


def _model() -> GaussianModel:
    return GaussianModel(
        means_world=torch.tensor([[-0.10, 0.00, 2.0], [0.15, 0.08, 2.3]]),
        raw_quaternions=torch.tensor(
            [[1.0, 0.10, 0.05, 0.00], [1.0, 0.00, 0.20, -0.10]]
        ),
        raw_scales=torch.log(
            torch.tensor([[0.12, 0.08, 0.10], [0.10, 0.14, 0.08]])
        ),
        raw_opacities=torch.logit(torch.tensor([[0.35], [0.45]])),
        sh_dc=torch.tensor(
            [[[0.10, -0.05, 0.00]], [[-0.08, 0.02, 0.12]]]
        ),
        sh_rest=torch.zeros((2, 15, 3)),
    )


def _empty_model() -> GaussianModel:
    model = _model()
    return GaussianModel(
        **{
            name: parameter.detach()[:0]
            for name, parameter in model.gaussian_parameter_dict().items()
        }
    )


def _camera(center_x: float, name: str) -> Camera:
    center = torch.tensor([center_x, 0.0, 0.0])
    image = torch.full((3, 7, 7), 0.25)
    image[0].add_(0.05)
    image[2].sub_(0.05)
    return Camera(
        rotation_cw=torch.eye(3),
        translation_cw=-center,
        camera_center_world=center,
        fx=8.0,
        fy=8.0,
        cx=3.0,
        cy=3.0,
        width=7,
        height=7,
        image=image,
        image_name=name,
    )


def _trainer(
    tmp_path: Path,
    config: Config,
    *,
    model: GaussianModel | None = None,
    statistics: ScreenSpaceDensityStatistics | None = None,
    train_cameras: list[Camera] | None = None,
    start_iteration: int = 0,
) -> Trainer:
    model = _model() if model is None else model
    train_cameras = train_cameras or [
        _camera(-1.0, "left.png"),
        _camera(1.0, "right.png"),
    ]
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    return Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=train_cameras,
        evaluation_cameras=[_camera(100.0, "evaluation.png")],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path,
        start_iteration=start_iteration,
        camera_order=list(range(len(train_cameras))),
        density_statistics=statistics,
    )


def _no_op_density_result(count: int) -> GaussianDensityControlResult:
    return GaussianDensityControlResult(
        num_gaussians_before=count,
        num_gaussians_after=count,
        clone_result=GaussianCloneResult(count, count, 0),
        split_result=GaussianSplitResult(count, count, 0, 0),
        prune_result=GaussianPruneResult(count, count, 0, 0, 0, 0),
        statistics_reset=True,
    )


def test_constructor_strictly_matches_statistics_to_feature_and_scene_extent(
    tmp_path: Path,
) -> None:
    model = _model()
    disabled = _config(
        adaptive_density_control=False,
        opacity_reset=False,
    )
    enabled = _config(
        adaptive_density_control=True,
        opacity_reset=False,
    )

    fixed = _trainer(
        tmp_path / "fixed",
        disabled,
        model=model,
        train_cameras=[_camera(0.0, "only.png")],
    )
    assert fixed.scene_extent is None
    with pytest.raises(ValueError, match="must be None"):
        _trainer(
            tmp_path / "unexpected",
            disabled,
            statistics=ScreenSpaceDensityStatistics.for_model(_model()),
        )
    with pytest.raises(ValueError, match="is required"):
        _trainer(tmp_path / "missing", enabled)

    adc_model = _model()
    adc = _trainer(
        tmp_path / "adc",
        enabled,
        model=adc_model,
        statistics=ScreenSpaceDensityStatistics.for_model(adc_model),
    )
    assert adc.scene_extent == pytest.approx(1.1)


def test_opacity_only_mode_does_not_create_statistics_or_scene_extent(
    tmp_path: Path,
) -> None:
    config = _config(
        adaptive_density_control=False,
        opacity_reset=True,
    )

    trainer = _trainer(
        tmp_path,
        config,
        train_cameras=[_camera(0.0, "only.png")],
    )

    assert trainer.density_statistics is None
    assert trainer.scene_extent is None
    result = trainer.train_step(trainer.train_cameras[0], iteration=1)
    assert not result.render.projected.means_screen.retains_grad
    assert result.density_control_result is None


def test_schedule_boundaries_control_collection_and_event(
    tmp_path: Path,
) -> None:
    config = _config(
        adaptive_density_control=True,
        opacity_reset=True,
        iterations=5,
        densify_from=2,
        densify_until=5,
        densification_interval=2,
        opacity_reset_interval=5,
    )
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = _trainer(tmp_path, config, model=model, statistics=statistics)
    camera = trainer.train_cameras[0]

    before_start = trainer.train_step(camera, iteration=1)
    at_start = trainer.train_step(camera, iteration=2)
    first_event = trainer.train_step(camera, iteration=4)
    assert first_event.density_control_result is not None
    assert torch.count_nonzero(
        statistics.position_gradient_accumulator
    ).item() == 0
    assert torch.count_nonzero(
        statistics.position_gradient_denominator
    ).item() == 0
    assert torch.count_nonzero(statistics.max_screen_radius).item() == 0
    statistics.position_gradient_accumulator.fill_(7.0)
    statistics.position_gradient_denominator.fill_(2)
    statistics.max_screen_radius.fill_(3)
    state_before_until = statistics.state_dict()
    at_until = trainer.train_step(camera, iteration=5)

    assert before_start.render.projected.means_screen.retains_grad
    assert before_start.density_control_result is None
    assert at_start.render.projected.means_screen.retains_grad
    assert at_start.density_control_result is None
    assert torch.count_nonzero(
        statistics.position_gradient_accumulator
    ).item() == model.num_gaussians
    assert not at_until.render.projected.means_screen.retains_grad
    assert at_until.density_control_result is None
    assert at_until.opacity_reset_result is None
    for name, expected in state_before_until.items():
        torch.testing.assert_close(
            getattr(statistics, name), expected, rtol=0.0, atol=0.0
        )


def test_size_pruning_thresholds_activate_only_after_reset_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        adaptive_density_control=True,
        opacity_reset=False,
        iterations=6,
        densify_from=0,
        densify_until=6,
        densification_interval=2,
        opacity_reset_interval=3,
    )
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = _trainer(tmp_path, config, model=model, statistics=statistics)
    observed: list[tuple[float | None, float | None]] = []

    def event_spy(*args: object, **kwargs: object) -> GaussianDensityControlResult:
        observed.append(
            (
                kwargs["prune_screen_radius_threshold"],
                kwargs["prune_world_scale_threshold"],
            )
        )
        statistics.reset()
        return _no_op_density_result(model.num_gaussians)

    monkeypatch.setattr(trainer_module, "run_density_control_event", event_spy)
    for iteration in range(1, 5):
        trainer.train_step(trainer.train_cameras[0], iteration)

    assert observed[0] == (None, None)
    assert observed[1][0] == pytest.approx(20.0)
    assert observed[1][1] == pytest.approx(0.11)


def test_execution_order_is_accumulate_event_reset_then_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        adaptive_density_control=True,
        opacity_reset=True,
        iterations=3,
        densify_from=0,
        densify_until=3,
        densification_interval=2,
        opacity_reset_interval=2,
    )
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = _trainer(tmp_path, config, model=model, statistics=statistics)
    order: list[str] = []
    original_accumulate = statistics.accumulate
    original_step = trainer.optimizer.step
    original_finite_gradients = trainer._assert_finite_gradients
    original_finite_parameters = trainer._assert_finite_parameters

    def finite_gradients() -> None:
        assert all(
            parameter.grad is not None for parameter in trainer.model.parameters()
        )
        order.append("finite_gradients")
        original_finite_gradients()

    def finite_parameters() -> None:
        order.append("finite_parameters")
        original_finite_parameters()

    def accumulate(render: object) -> None:
        assert getattr(render, "projected").means_screen.grad is not None
        order.append("accumulate")
        original_accumulate(render)  # type: ignore[arg-type]

    def event(*args: object, **kwargs: object) -> GaussianDensityControlResult:
        order.append("event")
        statistics.reset()
        return _no_op_density_result(model.num_gaussians)

    class DiagnosticSpy:
        def observe(self, *args: object, **kwargs: object) -> None:
            assert torch.any(statistics.position_gradient_denominator > 0)
            order.append("diagnostic")

    def reset(*args: object, **kwargs: object) -> GaussianOpacityResetResult:
        order.append("reset")
        return GaussianOpacityResetResult(model.num_gaussians, 0, 0.01)

    def step(*args: object, **kwargs: object) -> object:
        order.append("step")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(statistics, "accumulate", accumulate)
    monkeypatch.setattr(trainer, "_assert_finite_gradients", finite_gradients)
    monkeypatch.setattr(trainer, "_assert_finite_parameters", finite_parameters)
    monkeypatch.setattr(trainer_module, "run_density_control_event", event)
    monkeypatch.setattr(trainer_module, "reset_gaussian_opacity", reset)
    monkeypatch.setattr(trainer.optimizer, "step", step)
    trainer.screen_radius_diagnostic = DiagnosticSpy()  # type: ignore[assignment]

    trainer.train_step(trainer.train_cameras[0], iteration=2)

    assert order == [
        "finite_gradients",
        "accumulate",
        "diagnostic",
        "event",
        "reset",
        "finite_parameters",
        "step",
        "finite_parameters",
    ]


def test_trainer_rejects_zero_gaussians_at_start_and_after_event(
    tmp_path: Path,
) -> None:
    fixed = _config(
        adaptive_density_control=False,
        opacity_reset=False,
    )
    with pytest.raises(RuntimeError, match="zero Gaussians"):
        _trainer(tmp_path / "start", fixed, model=_empty_model())
    completed = _trainer(
        tmp_path / "completed",
        fixed,
        model=_empty_model(),
        start_iteration=fixed.training.iterations,
    )
    assert completed.model.num_gaussians == 0

    pruning = _config(
        adaptive_density_control=True,
        opacity_reset=False,
        iterations=2,
        densify_from=0,
        densify_until=2,
        densification_interval=1,
        gradient_threshold=1.0e9,
        prune_opacity_threshold=0.9,
    )
    model = _model()
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = _trainer(
        tmp_path / "event",
        pruning,
        model=model,
        statistics=statistics,
    )

    with pytest.raises(RuntimeError, match="pruned all Gaussians"):
        trainer.train_step(trainer.train_cameras[0], iteration=1)

    assert model.num_gaussians == 0
