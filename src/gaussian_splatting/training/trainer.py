"""Deterministic single-view-at-a-time training orchestration."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from gaussian_splatting.config import Config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.evaluation.metrics import mean_psnr, psnr
from gaussian_splatting.io.checkpoint import save_checkpoint
from gaussian_splatting.model.gaussian_model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer, RenderResult
from gaussian_splatting.training.density_control import (
    GaussianDensityControlResult,
    GaussianOpacityResetResult,
    ScreenSpaceDensityStatistics,
    reset_gaussian_opacity,
    run_density_control_event,
)
from gaussian_splatting.training.losses import LossResult, total_loss
from gaussian_splatting.training.profiling import TrainingProfiler
from gaussian_splatting.training.schedules import (
    PositionLearningRateScheduler,
    compute_scene_extent,
    density_control_event_parameters,
    density_control_schedule,
)
from gaussian_splatting.training.screen_radius_diagnostics import (
    ScreenRadiusDiagnostic,
)


@dataclass(frozen=True)
class TrainStepResult:
    """Outputs and scalar diagnostics for one optimizer update."""

    iteration: int
    loss: LossResult
    psnr: Tensor
    position_learning_rate: float
    render: RenderResult
    density_control_result: GaussianDensityControlResult | None = None
    opacity_reset_result: GaussianOpacityResetResult | None = None


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
    resolution_scale: float = 1.0,
) -> Camera:
    """Copy a camera and optionally its image to a runtime device and dtype."""

    if not 0.0 < resolution_scale <= 1.0:
        raise ValueError("resolution_scale must satisfy 0 < value <= 1")
    width = max(1, round(camera.width * resolution_scale))
    height = max(1, round(camera.height * resolution_scale))
    scale_x = width / camera.width
    scale_y = height / camera.height
    image = camera.image
    if image is not None and include_image:
        if (height, width) != (camera.height, camera.width):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        image = image.to(device=device, dtype=dtype)
    elif not include_image:
        image = None
    return Camera(
        rotation_cw=camera.rotation_cw.to(device=device, dtype=dtype),
        translation_cw=camera.translation_cw.to(device=device, dtype=dtype),
        camera_center_world=camera.camera_center_world.to(device=device, dtype=dtype),
        fx=camera.fx * scale_x,
        fy=camera.fy * scale_y,
        cx=camera.cx * scale_x,
        cy=camera.cy * scale_y,
        width=width,
        height=height,
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
    """Train, validate, log, and checkpoint a dynamically sized Gaussian model."""

    def __init__(
        self,
        *,
        model: GaussianModel,
        renderer: GaussianRenderer,
        train_cameras: list[Camera],
        train_cameras_quarter: list[Camera] | None = None,
        train_cameras_half: list[Camera] | None = None,
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
        screen_radius_diagnostic: ScreenRadiusDiagnostic | None = None,
        profile_training: bool = False,
        profile_warmup: int = 5,
        profile_steps: int = 10,
        milestone_iterations: tuple[int, ...] = (),
        stop_iteration: int | None = None,
    ) -> None:
        if not train_cameras:
            raise ValueError("training camera set must be non-empty")
        if not 0 <= start_iteration <= config.training.iterations:
            raise ValueError("start_iteration is outside the configured training range")
        self.model = model
        self.renderer = renderer
        self.train_cameras = train_cameras
        self.train_cameras_quarter = train_cameras_quarter
        self.train_cameras_half = train_cameras_half
        if config.features.resolution_warmup:
            for label, cameras in (
                ("quarter", train_cameras_quarter),
                ("half", train_cameras_half),
            ):
                if cameras is None or len(cameras) != len(train_cameras):
                    raise ValueError(
                        f"resolution warm-up requires an aligned {label}-resolution camera set"
                    )
        self.evaluation_cameras = evaluation_cameras
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_directory = Path(output_directory)
        self.start_iteration = int(start_iteration)
        self.best_mean_psnr = best_mean_psnr
        if (
            model.num_gaussians == 0
            and start_iteration < config.training.iterations
        ):
            raise RuntimeError(
                "cannot continue training with zero Gaussians"
            )
        if config.features.adaptive_density_control:
            if density_statistics is None:
                raise ValueError(
                    "density_statistics is required when adaptive density "
                    "control is enabled"
                )
            if not isinstance(
                density_statistics,
                ScreenSpaceDensityStatistics,
            ):
                raise TypeError(
                    "density_statistics must be ScreenSpaceDensityStatistics"
                )
            density_statistics.validate_compatible(model)
            self.scene_extent: float | None = compute_scene_extent(train_cameras)
        else:
            if density_statistics is not None:
                raise ValueError(
                    "density_statistics must be None when adaptive density "
                    "control is disabled"
                )
            self.scene_extent = None
        self.density_statistics = density_statistics
        self.screen_radius_diagnostic = screen_radius_diagnostic
        self._started_at = time.monotonic()
        self.training_profiler = (
            TrainingProfiler(
                output_directory=self.output_directory,
                device=self._model_device,
                dtype=self._model_dtype,
                render_backend=self.renderer.backend,
                warmup_steps=profile_warmup,
                profile_steps=profile_steps,
            )
            if profile_training
            else None
        )
        self.milestone_iterations = frozenset(int(value) for value in milestone_iterations)
        if any(value <= 0 or value > config.training.iterations for value in self.milestone_iterations):
            raise ValueError("milestone iterations must be within the training range")
        self.stop_iteration = (
            config.training.iterations
            if stop_iteration is None
            else int(stop_iteration)
        )
        if not start_iteration <= self.stop_iteration <= config.training.iterations:
            raise ValueError("stop_iteration is outside the remaining training range")
        self._telemetry_path = self.output_directory / "training_telemetry.json"
        self._prior_training_loop_seconds = 0.0
        self._prior_training_wall_seconds = 0.0
        self._prior_peak_allocated_mib = 0.0
        self._prior_peak_reserved_mib = 0.0
        self._milestone_telemetry: dict[str, dict[str, float | int]] = {}
        if self.start_iteration > 0 and self._telemetry_path.is_file():
            previous = json.loads(self._telemetry_path.read_text(encoding="utf-8"))
            self._prior_training_loop_seconds = float(previous.get("training_loop_seconds", 0.0))
            self._prior_training_wall_seconds = float(previous.get("training_wall_time_seconds", 0.0))
            self._prior_peak_allocated_mib = float(previous.get("peak_cuda_memory_allocated_MiB", 0.0))
            self._prior_peak_reserved_mib = float(previous.get("peak_cuda_memory_reserved_MiB", 0.0))
            self._milestone_telemetry = dict(previous.get("milestones", {}))
        self._training_loop_seconds = 0.0
        self._training_loop_started_at: float | None = None
        self._last_completed_iteration = self.start_iteration

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

    def _camera_for_iteration(self, index: int, iteration: int) -> Camera:
        if not self.config.features.resolution_warmup or iteration > 500:
            return self.train_cameras[index]
        if iteration <= 250:
            if self.train_cameras_quarter is None:  # pragma: no cover - constructor
                raise RuntimeError("quarter-resolution cameras are unavailable")
            return self.train_cameras_quarter[index]
        if self.train_cameras_half is None:  # pragma: no cover - constructor
            raise RuntimeError("half-resolution cameras are unavailable")
        return self.train_cameras_half[index]

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

        if self.model.num_gaussians == 0:
            raise RuntimeError("cannot train with zero Gaussians")
        profile_timer = (
            self.training_profiler.begin_iteration()
            if self.training_profiler is not None
            else None
        )
        gaussian_count = self.model.num_gaussians if profile_timer is not None else 0
        learning_rate = self.scheduler.step(iteration)
        if self.config.features.progressive_sh_degree and iteration % 1000 == 0:
            self.model.one_up_sh_degree()
        self.optimizer.zero_grad(set_to_none=True)
        decision = density_control_schedule(self.config, iteration)
        runtime_camera = self._runtime_camera(camera)
        if runtime_camera.image is None:
            raise ValueError("a training camera must contain a target image")
        render = self.renderer(
            self.model,
            runtime_camera,
            retain_screen_grad=decision.collect_statistics,
            profile_timer=profile_timer,
        )
        if render.image.shape != runtime_camera.image.shape:
            raise ValueError("rendered image and target image shapes differ")
        if profile_timer is None:
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
        else:
            with profile_timer.measure("loss"):
                loss = total_loss(
                    render.image,
                    runtime_camera.image,
                    lambda_dssim=self.config.loss.lambda_dssim,
                    window_size=self.config.loss.ssim_window_size,
                    sigma=self.config.loss.ssim_sigma,
                    k1=self.config.loss.ssim_k1,
                    k2=self.config.loss.ssim_k2,
                )
            with profile_timer.measure("backward"):
                loss.total.backward()
        self._assert_finite_gradients()
        if decision.collect_statistics:
            if self.density_statistics is None:  # pragma: no cover - constructor
                raise RuntimeError("density statistics are unavailable")
            self.density_statistics.accumulate(render)
            if self.screen_radius_diagnostic is not None:
                self.screen_radius_diagnostic.observe(
                    render,
                    iteration=iteration,
                    view_name=camera.image_name,
                    model=self.model,
                    statistics=self.density_statistics,
                )
        density_control_result: GaussianDensityControlResult | None = None
        if decision.run_density_control_event:
            if self.density_statistics is None or self.scene_extent is None:
                raise RuntimeError("density-control runtime state is unavailable")
            parameters = density_control_event_parameters(
                self.config,
                iteration,
                self.scene_extent,
            )
            density_control_result = run_density_control_event(
                self.model,
                self.optimizer,
                self.density_statistics,
                gradient_threshold=parameters.gradient_threshold,
                densify_world_scale_threshold=(
                    parameters.densify_world_scale_threshold
                ),
                prune_opacity_threshold=parameters.prune_opacity_threshold,
                prune_screen_radius_threshold=(
                    parameters.prune_screen_radius_threshold
                ),
                prune_world_scale_threshold=(
                    parameters.prune_world_scale_threshold
                ),
            )
            if self.model.num_gaussians == 0:
                raise RuntimeError(
                    "density-control event pruned all Gaussians"
                )
        opacity_reset_result: GaussianOpacityResetResult | None = None
        if decision.run_opacity_reset:
            opacity_reset_result = reset_gaussian_opacity(
                self.model,
                self.optimizer,
                maximum_opacity=(
                    self.config.density_control.opacity_reset_maximum
                ),
            )
        self._assert_finite_parameters()
        if profile_timer is None:
            self.optimizer.step()
        else:
            with profile_timer.measure("optimizer"):
                self.optimizer.step()
        self._assert_finite_parameters()
        with torch.no_grad():
            image_psnr = psnr(render.image.clamp(0.0, 1.0), runtime_camera.image)
        if profile_timer is not None and self.training_profiler is not None:
            _, stage_times = profile_timer.finish()
            self.training_profiler.record_iteration(
                iteration=iteration,
                gaussian_count=gaussian_count,
                visible_gaussian_count=int(render.visible_mask.sum().item()),
                width=runtime_camera.width,
                height=runtime_camera.height,
                density_control_event=decision.run_density_control_event,
                opacity_reset_event=decision.run_opacity_reset,
                stage_times=stage_times,
            )
        return TrainStepResult(
            iteration=int(iteration),
            loss=loss,
            psnr=image_psnr,
            position_learning_rate=learning_rate,
            render=render,
            density_control_result=density_control_result,
            opacity_reset_result=opacity_reset_result,
        )

    def validate(self, iteration: int) -> EvaluationResult:
        """Evaluate every held-out view and optionally save rendered PNGs."""

        if not self.evaluation_cameras:
            raise ValueError("evaluation camera set must be non-empty")

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
            "density_statistics": self.density_statistics,
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

    def save_recovery_checkpoint(self, iteration: int) -> None:
        """Overwrite one resumable checkpoint without accumulating large files."""
        checkpoint_directory = self.output_directory / "checkpoints"
        recovery = checkpoint_directory / "recovery.pt"
        save_checkpoint(recovery, **self._checkpoint_arguments(iteration))
        shutil.copyfile(recovery, checkpoint_directory / "latest.pt")

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
        record: dict[str, bool | int | float] = {
            "iteration": result.iteration,
            "loss_total": float(result.loss.total.detach().item()),
            "loss_l1": float(result.loss.l1.detach().item()),
            "loss_dssim": float(result.loss.dssim.detach().item()),
            "psnr": float(result.psnr.detach().item()),
            "gaussian_count": self.model.num_gaussians,
            "visible_gaussian_count": int(result.render.visible_mask.sum().item()),
            "elapsed_seconds": time.monotonic() - self._started_at,
        }
        if self._model_device.type == "cuda":
            record.update(
                {
                    "cuda_peak_allocated_MiB": (
                        torch.cuda.max_memory_allocated(self._model_device) / (1024**2)
                    ),
                    "cuda_peak_reserved_MiB": (
                        torch.cuda.max_memory_reserved(self._model_device) / (1024**2)
                    ),
                }
            )
        record.update(
            {f"lr_{group_name}": value for group_name, value in learning_rates.items()}
        )
        if result.density_control_result is not None:
            density = result.density_control_result
            record.update(
                {
                    "density_control_event": True,
                    "density_num_gaussians_before": density.num_gaussians_before,
                    "density_num_gaussians_after": density.num_gaussians_after,
                    "density_num_observed": density.num_observed,
                    "density_num_high_gradient": density.num_high_gradient,
                    "density_num_cloned": density.num_cloned,
                    "density_num_split_parents": density.num_split_parents,
                    "density_num_children_created": density.num_children_created,
                    "density_num_pruned_total": density.num_pruned_total,
                    "density_num_low_opacity": density.num_low_opacity,
                    "density_num_large_screen": density.num_large_screen,
                    "density_num_large_world": density.num_large_world,
                }
            )
        if result.opacity_reset_result is not None:
            opacity = result.opacity_reset_result
            record.update(
                {
                    "opacity_reset": True,
                    "opacity_num_clamped": opacity.num_clamped,
                    "opacity_reset_maximum": opacity.maximum_opacity,
                    "opacity_before_mean": opacity.before_mean,
                    "opacity_before_median": opacity.before_median,
                    "opacity_before_min": opacity.before_min,
                    "opacity_before_max": opacity.before_max,
                    "opacity_after_mean": opacity.after_mean,
                    "opacity_after_median": opacity.after_median,
                    "opacity_after_min": opacity.after_min,
                    "opacity_after_max": opacity.after_max,
                }
            )
        log_path = self.output_directory / "train_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def write_training_telemetry(self, *, status: str = "RUNNING") -> None:
        """Persist cumulative training-only/wall clocks and allocator peaks."""
        current_wall = 0.0
        if self._training_loop_started_at is not None:
            current_wall = time.monotonic() - self._training_loop_started_at
        allocated = reserved = 0.0
        if self._model_device.type == "cuda":
            torch.cuda.synchronize(self._model_device)
            allocated = torch.cuda.max_memory_allocated(self._model_device) / (1024**2)
            reserved = torch.cuda.max_memory_reserved(self._model_device) / (1024**2)
        payload = {
            "status": status,
            "iteration": self._last_completed_iteration,
            "training_loop_seconds": self._prior_training_loop_seconds + self._training_loop_seconds,
            "training_wall_time_seconds": self._prior_training_wall_seconds + current_wall,
            "peak_cuda_memory_allocated_MiB": max(self._prior_peak_allocated_mib, allocated),
            "peak_cuda_memory_reserved_MiB": max(self._prior_peak_reserved_mib, reserved),
            "gaussian_count": self.model.num_gaussians,
            "model_parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ),
            "active_sh_degree": self.model.active_sh_degree,
            "milestones": self._milestone_telemetry,
        }
        temporary = self._telemetry_path.with_suffix(".json.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self._telemetry_path)

    def train(self) -> None:
        """Minimize reconstruction loss by repeated Adam updates.

        TeX: eq:optimization_problem.  Logging, evaluation, and checkpointing
        are forced at the final step even when their intervals do not divide it.
        """

        final_iteration = self.stop_iteration
        self.model.train()
        self._training_loop_started_at = time.monotonic()
        for iteration in range(self.start_iteration + 1, final_iteration + 1):
            camera_index, _ = self._next_camera()
            camera = self._camera_for_iteration(camera_index, iteration)
            step_started_at = time.monotonic()
            step_result = self.train_step(camera, iteration)
            self._training_loop_seconds += time.monotonic() - step_started_at
            self._last_completed_iteration = iteration
            is_final = iteration == final_iteration
            is_milestone = iteration in self.milestone_iterations
            if is_milestone or is_final:
                current_wall = self._prior_training_wall_seconds + (
                    time.monotonic() - self._training_loop_started_at
                )
                self._milestone_telemetry[str(iteration)] = {
                    "training_loop_seconds": self._prior_training_loop_seconds + self._training_loop_seconds,
                    "training_wall_time_seconds": current_wall,
                    "gaussian_count": self.model.num_gaussians,
                    "model_parameter_bytes": sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in self.model.parameters()
                    ),
                }
            force_event_log = (
                step_result.density_control_result is not None
                or step_result.opacity_reset_result is not None
            )
            if (
                iteration % self.config.training.log_interval == 0
                or force_event_log
                or is_final
            ):
                self._write_log(step_result)

            if self.evaluation_cameras and (
                iteration % self.config.training.evaluation_interval == 0 or is_final
            ):
                evaluation = self.validate(iteration)
                if self.best_mean_psnr is None or evaluation.mean_psnr > self.best_mean_psnr:
                    self.best_mean_psnr = evaluation.mean_psnr
                    self._save_best_checkpoint(iteration)

            if is_milestone or is_final:
                self.save_checkpoint(iteration)
            elif iteration % self.config.training.checkpoint_interval == 0:
                if self.milestone_iterations:
                    self.save_recovery_checkpoint(iteration)
                else:
                    self.save_checkpoint(iteration)
            if is_milestone or is_final or iteration % self.config.training.checkpoint_interval == 0:
                self.write_training_telemetry(status="RUNNING")
        if self.training_profiler is not None:
            self.training_profiler.finalize()
        self.write_training_telemetry(status="COMPLETED")


__all__ = [
    "EvaluationResult",
    "TrainStepResult",
    "Trainer",
    "camera_to",
]
