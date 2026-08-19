"""Deterministic single-view-at-a-time training orchestration."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from gaussian_splatting.config import Config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.evaluation.metrics import mean_psnr, psnr
from gaussian_splatting.io.checkpoint import save_checkpoint
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer, RenderResult
from gaussian_splatting.training.density_control import ScreenSpaceDensityStatistics
from gaussian_splatting.training.losses import LossResult, total_loss
from gaussian_splatting.training.schedules import PositionLearningRateScheduler


@dataclass(frozen=True)
class TrainStepResult:
    """Outputs and scalar diagnostics for one optimizer update."""

    iteration: int
    loss: LossResult
    psnr: Tensor
    position_learning_rate: float
    render: RenderResult


@dataclass(frozen=True)
class EvaluationResult:
    """Per-view and mean PSNR from a validation pass."""

    per_image_psnr: dict[str, float]
    mean_psnr: float


def camera_to(
    camera: Camera,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_image: bool = True,
) -> Camera:
    """Copy a camera and optionally its image to a runtime device and dtype."""

    image = camera.image
    if image is not None and include_image:
        image = image.to(device=device, dtype=dtype)
    elif not include_image:
        image = None
    return Camera(
        rotation_cw=camera.rotation_cw.to(device=device, dtype=dtype),
        translation_cw=camera.translation_cw.to(device=device, dtype=dtype),
        camera_center_world=camera.camera_center_world.to(device=device, dtype=dtype),
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        width=camera.width,
        height=camera.height,
        image=image,
        image_name=camera.image_name,
    )


def _save_image(image: Tensor, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rendered image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(pixels, mode="RGB").save(path)


class Trainer:
    """Train, validate, log, and checkpoint a fixed-size Gaussian model."""

    def __init__(
        self,
        *,
        model: GaussianModel,
        renderer: GaussianRenderer,
        train_cameras: list[Camera],
        evaluation_cameras: list[Camera],
        optimizer: torch.optim.Optimizer,
        scheduler: PositionLearningRateScheduler,
        config: Config,
        output_directory: str | Path,
        start_iteration: int = 0,
        camera_order: list[int] | None = None,
        camera_cursor: int = 0,
        best_mean_psnr: float | None = None,
        density_statistics: ScreenSpaceDensityStatistics | None = None,
    ) -> None:
        if not train_cameras or not evaluation_cameras:
            raise ValueError("training and evaluation camera sets must both be non-empty")
        if not 0 <= start_iteration <= config.training.iterations:
            raise ValueError("start_iteration is outside the configured training range")
        self.model = model
        self.renderer = renderer
        self.train_cameras = train_cameras
        self.evaluation_cameras = evaluation_cameras
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_directory = Path(output_directory)
        self.start_iteration = int(start_iteration)
        self.best_mean_psnr = best_mean_psnr
        if density_statistics is not None:
            density_statistics.validate_compatible(model)
        self.density_statistics = density_statistics
        self._started_at = time.monotonic()

        if camera_order is None:
            self.camera_order = list(range(len(train_cameras)))
            random.shuffle(self.camera_order)
            self.camera_cursor = 0
        else:
            if sorted(camera_order) != list(range(len(train_cameras))):
                raise ValueError("camera_order must be a permutation of training camera indices")
            if not 0 <= camera_cursor <= len(camera_order):
                raise ValueError("camera_cursor is outside camera_order")
            self.camera_order = list(camera_order)
            self.camera_cursor = int(camera_cursor)

    @property
    def _model_device(self) -> torch.device:
        return self.model.means_world.device

    @property
    def _model_dtype(self) -> torch.dtype:
        return self.model.means_world.dtype

    def _next_camera(self) -> tuple[int, Camera]:
        if self.camera_cursor == len(self.camera_order):
            random.shuffle(self.camera_order)
            self.camera_cursor = 0
        index = self.camera_order[self.camera_cursor]
        self.camera_cursor += 1
        return index, self.train_cameras[index]

    def _runtime_camera(self, camera: Camera) -> Camera:
        return camera_to(
            camera,
            device=self._model_device,
            dtype=self._model_dtype,
            include_image=True,
        )

    def _assert_finite_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"gradient for {name} contains NaN or Inf")

    def _assert_finite_parameters(self) -> None:
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"parameter {name} contains NaN or Inf")

    def train_step(self, camera: Camera, iteration: int) -> TrainStepResult:
        """Perform rendering, loss, autograd, and one Adam update."""

        runtime_camera = self._runtime_camera(camera)
        if runtime_camera.image is None:
            raise ValueError("a training camera must contain a target image")
        learning_rate = self.scheduler.step(iteration)
        self.optimizer.zero_grad(set_to_none=True)
        render = self.renderer(
            self.model,
            runtime_camera,
            retain_screen_grad=self.density_statistics is not None,
        )
        if render.image.shape != runtime_camera.image.shape:
            raise ValueError("rendered image and target image shapes differ")
        loss = total_loss(
            render.image,
            runtime_camera.image,
            lambda_dssim=self.config.loss.lambda_dssim,
            window_size=self.config.loss.ssim_window_size,
            sigma=self.config.loss.ssim_sigma,
            k1=self.config.loss.ssim_k1,
            k2=self.config.loss.ssim_k2,
        )
        loss.total.backward()
        self._assert_finite_gradients()
        if self.density_statistics is not None:
            self.density_statistics.accumulate(render)
        self.optimizer.step()
        self._assert_finite_parameters()
        with torch.no_grad():
            image_psnr = psnr(render.image.clamp(0.0, 1.0), runtime_camera.image)
        return TrainStepResult(
            iteration=int(iteration),
            loss=loss,
            psnr=image_psnr,
            position_learning_rate=learning_rate,
            render=render,
        )

    def validate(self, iteration: int) -> EvaluationResult:
        """Evaluate every held-out view and optionally save rendered PNGs."""

        was_training = self.model.training
        self.model.eval()
        per_image: dict[str, float] = {}
        values: list[Tensor] = []
        render_directory = self.output_directory / "renders" / "test"
        with torch.no_grad():
            for camera_index, camera in enumerate(self.evaluation_cameras):
                runtime_camera = self._runtime_camera(camera)
                if runtime_camera.image is None:
                    raise ValueError("an evaluation camera must contain a target image")
                rendered = self.renderer(self.model, runtime_camera).image.clamp(0.0, 1.0)
                value = psnr(rendered, runtime_camera.image)
                values.append(value)
                image_name = camera.image_name or f"view_{camera_index:03d}.png"
                per_image[image_name] = float(value.item())
                if (
                    self.config.output.save_rendered_images
                    and iteration == self.config.training.iterations
                ):
                    output_name = Path(image_name).with_suffix(".png").name
                    _save_image(rendered, render_directory / output_name)
        if was_training:
            self.model.train()
        average = float(mean_psnr(values).item())
        result = EvaluationResult(per_image_psnr=per_image, mean_psnr=average)
        metrics_path = (
            self.output_directory / "metrics" / f"validation_{iteration:08d}.json"
        )
        if metrics_path.exists():
            raise FileExistsError(
                f"refusing to overwrite validation metrics: {metrics_path}"
            )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"images": per_image, "mean_psnr": average}, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _checkpoint_arguments(self, iteration: int) -> dict[str, object]:
        return {
            "iteration": iteration,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "config": self.config,
            "camera_order": self.camera_order,
            "camera_cursor": self.camera_cursor,
            "best_mean_psnr": self.best_mean_psnr,
        }

    def save_checkpoint(self, iteration: int) -> None:
        """Save a numbered checkpoint and update ``latest.pt``."""

        checkpoint_directory = self.output_directory / "checkpoints"
        numbered = checkpoint_directory / f"iteration_{iteration:08d}.pt"
        if numbered.exists():
            raise FileExistsError(
                f"refusing to overwrite numbered checkpoint: {numbered}"
            )
        save_checkpoint(numbered, **self._checkpoint_arguments(iteration))
        shutil.copyfile(numbered, checkpoint_directory / "latest.pt")

    def _save_best_checkpoint(self, iteration: int) -> None:
        checkpoint_directory = self.output_directory / "checkpoints"
        save_checkpoint(
            checkpoint_directory / "best.pt",
            **self._checkpoint_arguments(iteration),
        )

    def _write_log(self, result: TrainStepResult) -> None:
        learning_rates = {
            str(group.get("name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }
        record: dict[str, int | float] = {
            "iteration": result.iteration,
            "loss_total": float(result.loss.total.detach().item()),
            "loss_l1": float(result.loss.l1.detach().item()),
            "loss_dssim": float(result.loss.dssim.detach().item()),
            "psnr": float(result.psnr.detach().item()),
            "gaussian_count": self.model.num_gaussians,
            "elapsed_seconds": time.monotonic() - self._started_at,
        }
        record.update(
            {f"lr_{group_name}": value for group_name, value in learning_rates.items()}
        )
        log_path = self.output_directory / "train_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def train(self) -> None:
        """Minimize reconstruction loss by repeated Adam updates.

        TeX: eq:optimization_problem.  Logging, evaluation, and checkpointing
        are forced at the final step even when their intervals do not divide it.
        """

        final_iteration = self.config.training.iterations
        self.model.train()
        for iteration in range(self.start_iteration + 1, final_iteration + 1):
            _, camera = self._next_camera()
            step_result = self.train_step(camera, iteration)
            is_final = iteration == final_iteration
            if iteration % self.config.training.log_interval == 0 or is_final:
                self._write_log(step_result)

            if iteration % self.config.training.evaluation_interval == 0 or is_final:
                evaluation = self.validate(iteration)
                if self.best_mean_psnr is None or evaluation.mean_psnr > self.best_mean_psnr:
                    self.best_mean_psnr = evaluation.mean_psnr
                    self._save_best_checkpoint(iteration)

            if iteration % self.config.training.checkpoint_interval == 0 or is_final:
                self.save_checkpoint(iteration)


__all__ = [
    "EvaluationResult",
    "TrainStepResult",
    "Trainer",
    "camera_to",
]
