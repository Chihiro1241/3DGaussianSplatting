"""The Dynamic 3D Gaussians priors layered on the frame-to-frame hand-off.

Three properties matter: a configuration that says nothing about the priors
trains exactly as before; each prior vanishes on the motion it is meant to
allow (a rigid motion of the whole scene) and grows on the motion it is meant
to penalize; and a regularized 4D run can be restarted mid-sequence without
changing what the restarted frame sees.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from gaussian_splatting.config import (
    DEFAULT_DYNAMIC_REGULARIZATION,
    ConfigError,
    config_from_mapping,
    load_config,
)
from gaussian_splatting.io.checkpoint import read_checkpoint
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer.renderer import GaussianRenderer
from gaussian_splatting.training.losses import RegularizationLoss
from gaussian_splatting.training.optimizer import create_optimizer
from gaussian_splatting.training.schedules import PositionLearningRateScheduler
from gaussian_splatting.training.trainer import Trainer

from dynamic_regularization import (
    DynamicRegularizer,
    FrameMotionState,
    build_neighbor_graph,
    extrapolate_handoff,
    freeze_parameter_groups,
    load_motion_origin,
    quaternion_conjugate,
    quaternion_multiply,
    save_motion_origin,
    unit_quaternions,
    validate_regularization_config,
)
from trainer_4d import FrameHandoff

from test_training_trainer import _tiny_camera, _tiny_config, _tiny_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
REGULARIZED_CONFIG = PROJECT_ROOT / "configs" / "neu3d" / "dynamic_regularization.yaml"


def _regularization(**overrides: object):
    return replace(DEFAULT_DYNAMIC_REGULARIZATION, enabled=True, **overrides)


def _random_scene(count: int = 40, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "means_world": torch.rand((count, 3), generator=generator, dtype=torch.float64),
        "raw_quaternions": torch.randn((count, 4), generator=generator, dtype=torch.float64),
        "raw_scales": torch.full((count, 3), -3.0, dtype=torch.float64),
        "raw_opacities": torch.zeros((count, 1), dtype=torch.float64),
        "sh_dc": torch.rand((count, 1, 3), generator=generator, dtype=torch.float64),
        "sh_rest": torch.zeros((count, 15, 3), dtype=torch.float64),
    }


def _model(parameters: dict[str, torch.Tensor]) -> GaussianModel:
    return GaussianModel(**parameters, epsilon_q=1e-8, sh_degree=3)


def _axis_angle_quaternion(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    direction = torch.tensor(axis, dtype=torch.float64)
    direction = direction / direction.norm()
    return torch.cat(
        (torch.tensor([math.cos(angle / 2.0)], dtype=torch.float64),
         math.sin(angle / 2.0) * direction)
    )


def _regularizer_for(
    parameters: dict[str, torch.Tensor], **overrides: object
) -> DynamicRegularizer:
    handoff = FrameHandoff(parameters=parameters, active_sh_degree=3, source_frame=1)
    graph = build_neighbor_graph(
        parameters["means_world"], num_neighbors=6, weight_lambda=5.0, reference_frame=1
    )
    previous = FrameMotionState.from_handoff(handoff, epsilon_q=1e-8)
    return DynamicRegularizer(graph, previous, _regularization(**overrides))


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_omitted_section_leaves_regularization_disabled() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.dynamic_regularization == DEFAULT_DYNAMIC_REGULARIZATION
    assert not config.dynamic_regularization.enabled


def test_shipped_regularized_configuration_matches_the_paper_weights() -> None:
    config = load_config(REGULARIZED_CONFIG)
    regularization = config.dynamic_regularization
    assert regularization.enabled
    assert (regularization.lambda_rigid, regularization.lambda_rotation,
            regularization.lambda_isometry, regularization.lambda_color) == (
        4.0, 4.0, 2.0, 0.01)
    assert regularization.num_neighbors == 20
    validate_regularization_config(config)


def test_section_is_checked_strictly() -> None:
    mapping = asdict(load_config(DEFAULT_CONFIG))
    mapping["dynamic_regularization"] = asdict(_regularization())
    assert config_from_mapping(mapping).dynamic_regularization.enabled

    mapping["dynamic_regularization"]["unexpected"] = 1.0
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_mapping(mapping)
    del mapping["dynamic_regularization"]["unexpected"]
    del mapping["dynamic_regularization"]["lambda_rigid"]
    with pytest.raises(ConfigError, match="missing keys"):
        config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_neighbors", 0),
        ("neighbor_weight_lambda", -1.0),
        ("lambda_rigid", -0.1),
        ("lambda_isometry", float("inf")),
    ],
)
def test_section_values_are_validated(field: str, value: object) -> None:
    mapping = asdict(load_config(DEFAULT_CONFIG))
    mapping["dynamic_regularization"] = {**asdict(_regularization()), field: value}
    with pytest.raises(ConfigError, match=field):
        config_from_mapping(mapping)


def test_priors_refuse_features_that_change_the_gaussian_set() -> None:
    base = load_config(REGULARIZED_CONFIG)
    with pytest.raises(ValueError, match="adaptive_density_control"):
        validate_regularization_config(
            replace(base, features=replace(base.features, adaptive_density_control=True))
        )
    with pytest.raises(ValueError, match="opacity_reset"):
        validate_regularization_config(
            replace(base, features=replace(base.features, opacity_reset=True))
        )
    # Disabled priors impose nothing.
    validate_regularization_config(load_config(DEFAULT_CONFIG))


# --------------------------------------------------------------------------
# neighbour graph
# --------------------------------------------------------------------------


def test_graph_matches_a_brute_force_search() -> None:
    means = _random_scene(30)["means_world"]
    graph = build_neighbor_graph(means, num_neighbors=5, weight_lambda=3.0, reference_frame=1)

    distances = torch.cdist(means, means)
    distances.fill_diagonal_(float("inf"))
    expected, _ = distances.topk(5, largest=False)
    torch.testing.assert_close(graph.distances, expected)
    torch.testing.assert_close(
        graph.distances, (means[graph.indices] - means[:, None]).norm(dim=-1)
    )
    torch.testing.assert_close(graph.weights, torch.exp(-3.0 * expected.square()))
    assert not (graph.indices == torch.arange(30)[:, None]).any()


def test_coincident_centres_never_make_a_gaussian_its_own_neighbour() -> None:
    means = torch.zeros((6, 3), dtype=torch.float64)
    means[5] = 1.0
    graph = build_neighbor_graph(means, num_neighbors=3, weight_lambda=1.0, reference_frame=1)
    assert not (graph.indices == torch.arange(6)[:, None]).any()
    assert graph.indices.shape == (6, 3)


def test_graph_needs_more_gaussians_than_neighbours() -> None:
    with pytest.raises(ValueError, match="more than 4"):
        build_neighbor_graph(torch.zeros((4, 3)), num_neighbors=4,
                             weight_lambda=1.0, reference_frame=1)


# --------------------------------------------------------------------------
# the priors
# --------------------------------------------------------------------------


def test_quaternion_product_matches_rotation_composition() -> None:
    from gaussian_splatting.math.covariance import quaternion_rotation_matrix

    first = unit_quaternions(torch.randn(8, 4, dtype=torch.float64), 1e-8)
    second = unit_quaternions(torch.randn(8, 4, dtype=torch.float64), 1e-8)
    torch.testing.assert_close(
        quaternion_rotation_matrix(quaternion_multiply(first, second)),
        quaternion_rotation_matrix(first) @ quaternion_rotation_matrix(second),
    )
    identity = quaternion_multiply(first, quaternion_conjugate(first))
    torch.testing.assert_close(identity, torch.tensor([1.0, 0, 0, 0], dtype=torch.float64).expand(8, 4))


def test_every_prior_vanishes_when_nothing_moves() -> None:
    parameters = _random_scene()
    result = _regularizer_for(parameters)(_model(parameters))
    assert set(result.terms) == {"rigid", "rotation", "isometry", "color"}
    for value in result.terms.values():
        assert float(value) < 1e-9


def test_rigid_motion_of_the_whole_scene_is_not_penalized() -> None:
    parameters = _random_scene()
    regularizer = _regularizer_for(parameters)
    rotation_q = _axis_angle_quaternion((0.3, -1.0, 0.5), 0.7)
    from gaussian_splatting.math.covariance import quaternion_rotation_matrix

    rotation = quaternion_rotation_matrix(rotation_q[None])[0]
    moved = dict(parameters)
    moved["means_world"] = parameters["means_world"] @ rotation.T + torch.tensor(
        [0.4, -0.2, 1.5], dtype=torch.float64
    )
    moved["raw_quaternions"] = quaternion_multiply(
        rotation_q.expand(parameters["raw_quaternions"].shape[0], 4),
        unit_quaternions(parameters["raw_quaternions"], 1e-8),
    )
    result = regularizer(_model(moved))
    for name in ("rigid", "rotation", "isometry"):
        assert float(result.terms[name]) < 1e-7, name


def test_non_rigid_motion_is_penalized_and_has_gradients() -> None:
    parameters = _random_scene()
    regularizer = _regularizer_for(parameters)
    moved = dict(parameters)
    # Stretch the scene along x and rotate every Gaussian differently.
    moved["means_world"] = parameters["means_world"] * torch.tensor(
        [1.5, 1.0, 1.0], dtype=torch.float64
    )
    moved["raw_quaternions"] = parameters["raw_quaternions"] + 0.3 * torch.randn(
        parameters["raw_quaternions"].shape, generator=torch.Generator().manual_seed(1),
        dtype=torch.float64,
    )
    moved["sh_dc"] = parameters["sh_dc"] + 0.1
    model = _model(moved)
    result = regularizer(model)
    for name in ("rigid", "rotation", "isometry"):
        assert float(result.terms[name]) > 1e-3, name
    torch.testing.assert_close(result.terms["color"], torch.tensor(0.3, dtype=torch.float64))
    expected_total = (4.0 * result.terms["rigid"] + 4.0 * result.terms["rotation"]
                      + 2.0 * result.terms["isometry"] + 0.01 * result.terms["color"])
    torch.testing.assert_close(result.total, expected_total)

    result.total.backward()
    for name in ("means_world", "raw_quaternions", "sh_dc"):
        gradient = getattr(model, name).grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.0
    assert model.raw_opacities.grad is None and model.raw_scales.grad is None


def test_isometry_is_measured_against_the_reference_frame() -> None:
    # Only the isometry term remembers the reference frame; a rigid step from
    # a stretched previous frame still costs isometry but not rigidity.
    parameters = _random_scene()
    handoff = FrameHandoff(parameters=parameters, active_sh_degree=3, source_frame=1)
    graph = build_neighbor_graph(parameters["means_world"], num_neighbors=6,
                                 weight_lambda=5.0, reference_frame=1)
    stretched = dict(parameters)
    stretched["means_world"] = parameters["means_world"] * 1.3
    previous = FrameMotionState.from_handoff(
        replace(handoff, parameters=stretched, source_frame=2), epsilon_q=1e-8
    )
    result = DynamicRegularizer(graph, previous, _regularization())(_model(stretched))
    assert float(result.terms["rigid"]) < 1e-9
    assert float(result.terms["isometry"]) > 1e-3


def test_zero_weights_skip_their_terms() -> None:
    parameters = _random_scene()
    regularizer = _regularizer_for(parameters, lambda_rigid=0.0, lambda_color=0.0)
    assert set(regularizer(_model(parameters)).terms) == {"rotation", "isometry"}


def test_a_changed_gaussian_count_is_refused() -> None:
    parameters = _random_scene()
    regularizer = _regularizer_for(parameters)
    smaller = {name: value[:-1] for name, value in parameters.items()}
    with pytest.raises(RuntimeError, match="neighbour graph"):
        regularizer(_model(smaller))


# --------------------------------------------------------------------------
# velocity initialization and freezing
# --------------------------------------------------------------------------


def test_velocity_initialization_continues_the_last_motion() -> None:
    parameters = _random_scene()
    origin_handoff = FrameHandoff(parameters=parameters, active_sh_degree=3, source_frame=1)
    origin = FrameMotionState.from_handoff(origin_handoff, epsilon_q=1e-8)
    step = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    turn = _axis_angle_quaternion((0.0, 0.0, 1.0), 0.1)
    moved = dict(parameters)
    moved["means_world"] = parameters["means_world"] + step
    moved["raw_quaternions"] = quaternion_multiply(
        turn.expand(40, 4), unit_quaternions(parameters["raw_quaternions"], 1e-8)
    )
    handoff = FrameHandoff(parameters=moved, active_sh_degree=3, source_frame=2)

    advanced = extrapolate_handoff(handoff, origin, epsilon_q=1e-8)
    torch.testing.assert_close(advanced.parameters["means_world"],
                               parameters["means_world"] + 2.0 * step)
    q = unit_quaternions(moved["raw_quaternions"], 1e-8)
    torch.testing.assert_close(
        advanced.parameters["raw_quaternions"],
        unit_quaternions(2.0 * q - origin.quaternions, 1e-8),
    )
    for name in ("raw_scales", "raw_opacities", "sh_dc", "sh_rest"):
        assert advanced.parameters[name] is handoff.parameters[name]
    assert extrapolate_handoff(handoff, None, epsilon_q=1e-8) is handoff
    with pytest.raises(ValueError, match="needs frame 2"):
        extrapolate_handoff(replace(handoff, source_frame=3), origin, epsilon_q=1e-8)


def test_freezing_zeroes_only_opacity_and_scale() -> None:
    config = _tiny_config()
    model = _tiny_model()
    optimizer = create_optimizer(model, config)
    before = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert sorted(freeze_parameter_groups(optimizer)) == ["raw_opacities", "raw_scales"]
    for group in optimizer.param_groups:
        expected = 0.0 if group["name"] in ("raw_opacities", "raw_scales") else before[group["name"]]
        assert group["lr"] == expected


def test_a_motion_origin_for_another_frame_is_refused(tmp_path: Path) -> None:
    parameters = _random_scene()
    origin = FrameMotionState.from_handoff(
        FrameHandoff(parameters=parameters, active_sh_degree=3, source_frame=4),
        epsilon_q=1e-8,
    )
    save_motion_origin(tmp_path, origin, next_frame=6)
    found, loaded = load_motion_origin(tmp_path, next_frame=6)
    assert found and loaded.frame == 4
    torch.testing.assert_close(loaded.means, origin.means)
    with pytest.raises(ValueError, match="not for frame 5"):
        load_motion_origin(tmp_path, next_frame=5)
    assert load_motion_origin(tmp_path / "absent", next_frame=5) == (False, None)


# --------------------------------------------------------------------------
# trainer hook
# --------------------------------------------------------------------------


def _trainer(tmp_path: Path, regularizer=None) -> Trainer:
    config = _tiny_config()
    model = _tiny_model()
    optimizer = create_optimizer(model, config)
    return Trainer(
        model=model,
        renderer=GaussianRenderer(config.rendering),
        train_cameras=[_tiny_camera()],
        evaluation_cameras=[],
        optimizer=optimizer,
        scheduler=PositionLearningRateScheduler(
            optimizer,
            total_iterations=config.training.iterations,
            initial_learning_rate=config.training.position_lr_initial,
            final_learning_rate=config.training.position_lr_final,
        ),
        config=config,
        output_directory=tmp_path,
        regularizer=regularizer,
    )


def test_trainer_adds_the_regularizer_to_the_objective(tmp_path: Path) -> None:
    def pull_to_origin(model: GaussianModel) -> RegularizationLoss:
        value = model.means_world.square().sum()
        return RegularizationLoss(total=10.0 * value, terms={"pull": value})

    def step_and_read_gradient(trainer: Trainer):
        # Read the position gradient just before Adam consumes it.
        captured = {}
        original_step = trainer.optimizer.step

        def step(*args, **kwargs):
            captured["means"] = trainer.model.means_world.grad.clone()
            return original_step(*args, **kwargs)

        trainer.optimizer.step = step
        result = trainer.train_step(_tiny_camera(), 1)
        return result, captured["means"]

    _, plain_gradient = step_and_read_gradient(_trainer(tmp_path / "plain"))
    regularized = _trainer(tmp_path / "regularized", regularizer=pull_to_origin)
    result, regularized_gradient = step_and_read_gradient(regularized)
    torch.testing.assert_close(
        regularized_gradient - plain_gradient, 20.0 * _tiny_model().means_world
    )
    assert result.regularization is not None
    regularized._write_log(result)
    record = json.loads((tmp_path / "regularized" / "train_log.jsonl").read_text().splitlines()[-1])
    assert record["loss_reg_pull"] == pytest.approx(float(_tiny_model().means_world.square().sum()))
    assert record["loss_regularization"] == pytest.approx(10.0 * record["loss_reg_pull"])
    # loss_total keeps meaning the image loss alone.
    assert record["loss_total"] == pytest.approx(float(result.loss.total))


def test_trainer_without_a_regularizer_logs_no_regularization(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    result = trainer.train_step(_tiny_camera(), 1)
    assert result.regularization is None
    trainer._write_log(result)
    record = json.loads((tmp_path / "train_log.jsonl").read_text())
    assert not any(key.startswith("loss_reg") for key in record)


# --------------------------------------------------------------------------
# end to end through scripts/4DGS/train_4d_regularized.py
# --------------------------------------------------------------------------


def _write_frame(root: Path, shift: float) -> None:
    """A two-view NeRF-synthetic frame whose bright square moves with ``shift``."""

    for split, views in (("train", (-0.3, 0.3)), ("val", (0.0,)), ("test", (0.0,))):
        directory = root / split
        directory.mkdir(parents=True)
        frames = []
        for index, x in enumerate(views):
            pixels = np.zeros((8, 8, 4), dtype=np.uint8)
            pixels[..., 3] = 255
            column = int(2 + round(shift * 4))
            pixels[2:6, column:column + 3, :3] = 220
            Image.fromarray(pixels).save(directory / f"r_{index}.png")
            frames.append({
                "file_path": f"./{split}/r_{index}",
                "transform_matrix": [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 3.0], [0, 0, 0, 1]],
            })
        (root / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": math.pi / 3.0, "frames": frames}), encoding="utf-8"
        )


def _write_config(path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config = replace(
        config,
        runtime=replace(config.runtime, device="cpu"),
        initialization=replace(config.initialization, num_gaussians=48),
        loss=replace(config.loss, ssim_window_size=3, ssim_sigma=0.8),
        training=replace(config.training, iterations=2, log_interval=1,
                         evaluation_interval=2, checkpoint_interval=2),
        features=replace(config.features, adaptive_density_control=False,
                         opacity_reset=False),
        output=replace(config.output, save_rendered_images=False),
        dynamic_regularization=_regularization(num_neighbors=4, neighbor_weight_lambda=1.0),
    )
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")


@pytest.fixture()
def sequence(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    for frame, shift in enumerate((0.0, 0.25, 0.5), start=1):
        _write_frame(data / f"frame_{frame:04d}", shift)
    config = tmp_path / "config.yaml"
    _write_config(config)
    return data, config


def _arguments(data: Path, config: Path, output: Path, *extra: str) -> list[str]:
    return ["--data", str(data), "--config", str(config), "--output", str(output),
            "--disable-training-evaluation", *extra]


def _run(data: Path, config: Path, output: Path, *extra: str) -> None:
    import train_4d_regularized

    assert train_4d_regularized.main(_arguments(data, config, output, *extra)) == 0


@pytest.fixture(autouse=True)
def _scripts_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts" / "4DGS"))


def _final_model(output: Path, frame: int):
    path = output / f"frame_{frame:04d}" / "checkpoints" / "iteration_00000002.pt"
    return read_checkpoint(path, map_location="cpu")["model_state_dict"]


def test_regularized_sequence_end_to_end(sequence: tuple[Path, Path], tmp_path: Path) -> None:
    data, config = sequence
    output = tmp_path / "run"
    _run(data, config, output)

    manifest = json.loads((output / "frames_4d.json").read_text())
    frames = {record["frame"]: record for record in manifest["frames"]}
    assert [frames[f]["dynamic_regularization"] for f in (1, 2, 3)] == [False, True, True]
    assert [frames[f]["velocity_initialized"] for f in (1, 2, 3)] == [False, False, True]
    assert frames[2]["frozen_parameter_groups"] == ["raw_opacities", "raw_scales"]
    assert (output / "dynamic_regularization" / "neighbor_graph.pt").is_file()

    first_log = (output / "frame_0001" / "train_log.jsonl").read_text()
    assert "loss_reg_rigid" not in first_log
    record = json.loads((output / "frame_0002" / "train_log.jsonl").read_text().splitlines()[0])
    for name in ("rigid", "rotation", "isometry", "color"):
        assert f"loss_reg_{name}" in record
    assert record["lr_raw_opacities"] == 0.0 and record["lr_raw_scales"] == 0.0

    # Frozen opacity and scale survive every regularized frame unchanged.
    reference = _final_model(output, 1)
    for frame in (2, 3):
        state = _final_model(output, frame)
        for name in ("raw_opacities", "raw_scales"):
            torch.testing.assert_close(state[name], reference[name])


def test_a_regularized_sequence_restarts_where_it_stopped(
    sequence: tuple[Path, Path], tmp_path: Path
) -> None:
    data, config = sequence
    unbroken = tmp_path / "unbroken"
    _run(data, config, unbroken)

    restarted = tmp_path / "restarted"
    _run(data, config, restarted, "--end-frame", "2")
    checkpoint = restarted / "frame_0002" / "checkpoints" / "iteration_00000002.pt"
    _run(data, config, restarted, "--start-frame", "3",
         "--carry-over-checkpoint", str(checkpoint))

    manifest = json.loads((restarted / "frames_4d.json").read_text())
    assert manifest["frames"][-1]["velocity_initialized"] is True
    for name, value in _final_model(unbroken, 3).items():
        torch.testing.assert_close(_final_model(restarted, 3)[name], value, msg=name)

    # A finished run's velocity origin belongs to frame 4, not frame 3.
    with pytest.raises(ValueError, match="not for frame 3"):
        _run(data, config, unbroken, "--start-frame", "3", "--carry-over-checkpoint",
             str(unbroken / "frame_0002" / "checkpoints" / "iteration_00000002.pt"))


def test_each_entry_point_refuses_the_other_ones_configuration(
    sequence: tuple[Path, Path], tmp_path: Path
) -> None:
    import train_4d
    import train_4d_regularized

    data, regularized_config = sequence
    with pytest.raises(ValueError, match="train_4d_regularized.py"):
        train_4d.main(_arguments(data, regularized_config, tmp_path / "plain"))

    plain_config = tmp_path / "plain.yaml"
    mapping = yaml.safe_load(regularized_config.read_text())
    mapping["dynamic_regularization"]["enabled"] = False
    plain_config.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="does not enable"):
        train_4d_regularized.main(_arguments(data, plain_config, tmp_path / "reg"))
