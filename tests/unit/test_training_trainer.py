from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from gaussian_splatting.config import Config, load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.density_control import ScreenSpaceDensityStatistics
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[2]


def _tiny_config() -> Config:
    config = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        config,
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(
            config.training,
            iterations=3,
            log_interval=3,
            evaluation_interval=3,
            checkpoint_interval=3,
        ),
        output=replace(config.output, save_rendered_images=False),
    )


def _tiny_model() -> GaussianModel:
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


def _tiny_camera(
    image_value: float = 0.25,
    name: str = "tiny.png",
    center_x: float = 0.0,
) -> Camera:
    image = torch.full((3, 7, 7), image_value)
    image[0].add_(0.05)
    image[2].sub_(0.05)
    center = torch.tensor([center_x, 0.0, 0.0])
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


def test_train_step_has_finite_gradients_and_updates_parameters(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = _tiny_model()
    camera = _tiny_camera()
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    trainer = Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[camera],
        evaluation_cameras=[camera],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path,
        camera_order=[0],
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result = trainer.train_step(camera, iteration=1)

    assert torch.isfinite(result.loss.total)
    assert torch.isfinite(result.loss.l1)
    assert torch.isfinite(result.loss.dssim)
    assert torch.isfinite(result.psnr)
    assert torch.isfinite(result.render.image).all()
    assert not result.render.projected.means_screen.retains_grad
    assert result.position_learning_rate == scheduler._position_group()["lr"]
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.isfinite(parameter).all()
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
    )


def test_train_step_accumulates_optional_density_statistics(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    base_config = _tiny_config()
    config = replace(
        base_config,
        features=replace(
            base_config.features,
            adaptive_density_control=True,
        ),
    )
    model = _tiny_model()
    camera = _tiny_camera()
    second_camera = _tiny_camera(name="second.png", center_x=1.0)
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=config.training.iterations,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    statistics = ScreenSpaceDensityStatistics.for_model(model)
    trainer = Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[camera, second_camera],
        evaluation_cameras=[camera],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path,
        camera_order=[0, 1],
        density_statistics=statistics,
    )

    result = trainer.train_step(camera, iteration=1)

    indices = result.render.projected.original_indices
    screen_gradient = result.render.projected.means_screen.grad
    assert result.render.projected.means_screen.retains_grad
    assert screen_gradient is not None
    expected_accumulator = torch.zeros_like(
        statistics.position_gradient_accumulator
    )
    expected_accumulator[indices] = torch.linalg.vector_norm(
        screen_gradient, dim=-1
    )
    expected_denominator = torch.zeros_like(
        statistics.position_gradient_denominator
    )
    expected_denominator[indices] = 1
    expected_radius = torch.zeros_like(statistics.max_screen_radius)
    expected_radius[indices] = result.render.projected.radii
    torch.testing.assert_close(
        statistics.position_gradient_accumulator, expected_accumulator
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator, expected_denominator
    )
    torch.testing.assert_close(statistics.max_screen_radius, expected_radius)


def test_one_hundred_updates_remain_finite(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    config = replace(
        config,
        training=replace(config.training, iterations=100),
    )
    model = _tiny_model()
    camera = _tiny_camera()
    optimizer = create_optimizer(model, config)
    scheduler = PositionLearningRateScheduler(
        optimizer,
        total_iterations=100,
        initial_learning_rate=config.training.position_lr_initial,
        final_learning_rate=config.training.position_lr_final,
    )
    trainer = Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[camera],
        evaluation_cameras=[camera],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_directory=tmp_path,
        camera_order=[0],
    )

    for iteration in range(1, 101):
        result = trainer.train_step(camera, iteration)
        assert torch.isfinite(result.loss.total)
        assert torch.isfinite(result.render.image).all()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
