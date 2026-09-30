"""Point tracking and tracking metrics of Dynamic 3D Gaussians.

Tracking is checked on synthetic scenes whose motion is known exactly
(identity, pure translation, pure rotation), and the metrics on errors
whose values are known in closed form.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gaussian_splatting.config import load_config
from gaussian_splatting.data.camera import Camera
from gaussian_splatting.math.covariance import quaternion_rotation_matrix
from gaussian_splatting.model import GaussianModel

from point_tracking import (
    NO_GAUSSIAN,
    FrameGaussians,
    ReferenceGaussians,
    anchor_points,
    anchor_to_gaussians,
    compact_anchors,
    frame_subset,
    max_influence,
    median_depth_gaussians,
    project_points,
    propagate,
    track_points,
    unproject_pixels,
)
from tracking_metrics import (
    errors_2d,
    errors_3d,
    in_image_mask,
    scene_mean,
    track_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _random_rotations(count: int, generator: torch.Generator) -> torch.Tensor:
    quaternions = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    return quaternion_rotation_matrix(quaternions / quaternions.norm(dim=-1, keepdim=True))


def _axis_rotation(axis: tuple[float, float, float], degrees: float) -> torch.Tensor:
    half = math.radians(degrees) / 2
    x, y, z = (torch.tensor(axis, dtype=torch.float64) / math.sqrt(sum(a * a for a in axis)))
    quaternion = torch.tensor([[math.cos(half), *(math.sin(half) * torch.stack((x, y, z)))]],
                              dtype=torch.float64)
    return quaternion_rotation_matrix(quaternion)[0]


@pytest.fixture
def scene() -> ReferenceGaussians:
    generator = torch.Generator().manual_seed(0)
    count = 6
    return ReferenceGaussians(
        means=torch.randn(count, 3, generator=generator, dtype=torch.float64) * 3.0,
        rotations=_random_rotations(count, generator),
        scales=torch.rand(count, 3, generator=generator, dtype=torch.float64) * 0.4 + 0.2,
        opacities=torch.full((count,), 0.95, dtype=torch.float64),
    )


def _queries_near(scene: ReferenceGaussians) -> torch.Tensor:
    """One point just off each Gaussian's centre, well inside f >= 0.5."""

    offsets = torch.einsum("nij,nj->ni", scene.rotations, scene.scales * 0.1)
    return scene.means + offsets


# --------------------------------------------------------------------------
# anchoring
# --------------------------------------------------------------------------


def test_max_influence_matches_dense_evaluation_in_chunks(scene: ReferenceGaussians) -> None:
    points = torch.randn(40, 3, generator=torch.Generator().manual_seed(1),
                         dtype=torch.float64) * 3.0
    indices, influences = max_influence(points, scene, query_chunk=7, gaussian_chunk=4)

    covariances = scene.rotations @ torch.diag_embed(scene.scales.square()) \
        @ scene.rotations.transpose(1, 2)
    offsets = points[:, None] - scene.means[None]
    mahalanobis = torch.einsum("qni,nij,qnj->qn", offsets, torch.linalg.inv(covariances), offsets)
    dense = scene.opacities[None] * torch.exp(-0.5 * mahalanobis)
    assert torch.equal(indices, dense.argmax(dim=1))
    assert torch.allclose(influences, dense.max(dim=1).values, rtol=1e-9, atol=1e-300)


def test_opacity_breaks_a_tie_in_distance() -> None:
    gaussians = ReferenceGaussians(
        means=torch.tensor([[-1.0, 0, 0], [1.0, 0, 0]]),
        rotations=torch.eye(3).expand(2, 3, 3),
        scales=torch.ones(2, 3),
        opacities=torch.tensor([0.6, 0.9]),
    )
    indices, influences = max_influence(torch.zeros(1, 3), gaussians)
    assert indices.tolist() == [1]
    assert influences.item() == pytest.approx(0.9 * math.exp(-0.5))


def test_points_far_from_every_gaussian_are_static_background(scene: ReferenceGaussians) -> None:
    far = torch.tensor([[100.0, -100.0, 50.0]], dtype=torch.float64)
    anchors = anchor_points(far, scene)
    assert anchors.gaussian_indices.tolist() == [NO_GAUSSIAN]

    moved = FrameGaussians(means=scene.means + 5.0, rotations=scene.rotations)
    assert torch.equal(propagate(anchors, moved), far)
    # With the threshold off, the same point follows its most influential Gaussian.
    followed = anchor_points(far, scene, background_threshold=0.0)
    assert not bool(followed.is_background.any())
    assert torch.allclose(propagate(followed, moved), far + 5.0)


# --------------------------------------------------------------------------
# propagation under known motion
# --------------------------------------------------------------------------


def test_identity_motion_keeps_points_fixed(scene: ReferenceGaussians) -> None:
    points = _queries_near(scene)
    anchors = anchor_points(points, scene)
    assert not bool(anchors.is_background.any())

    tracks = track_points(anchors, [scene.motion] * 4)
    assert tracks.shape == (len(points), 4, 3)
    assert torch.allclose(tracks, points[:, None].expand(-1, 4, -1), atol=1e-12)


def test_pure_translation_moves_points_by_the_same_vector(scene: ReferenceGaussians) -> None:
    points = _queries_near(scene)
    anchors = anchor_points(points, scene)
    shifts = [torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64) * t for t in range(5)]
    frames = [FrameGaussians(means=scene.means + s, rotations=scene.rotations) for s in shifts]

    tracks = track_points(anchors, frames)
    expected = torch.stack([points + s for s in shifts], dim=1)
    assert torch.allclose(tracks, expected, atol=1e-12)


def test_pure_rotation_rotates_points_with_the_scene(scene: ReferenceGaussians) -> None:
    points = _queries_near(scene)
    anchors = anchor_points(points, scene)
    rotation = _axis_rotation((1.0, 2.0, -0.5), 73.0)
    frame = FrameGaussians(
        means=scene.means @ rotation.T,
        rotations=rotation @ scene.rotations,
    )

    assert torch.allclose(propagate(anchors, frame), points @ rotation.T, atol=1e-12)


def test_rotation_of_one_gaussian_turns_its_point_around_its_centre() -> None:
    gaussians = ReferenceGaussians(
        means=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        rotations=torch.eye(3, dtype=torch.float64)[None],
        scales=torch.ones(1, 3, dtype=torch.float64),
        opacities=torch.ones(1, dtype=torch.float64),
    )
    point = torch.tensor([[1.1, 2.0, 3.0]], dtype=torch.float64)
    anchors = anchor_points(point, gaussians)
    quarter_turn = _axis_rotation((0.0, 0.0, 1.0), 90.0)
    frame = FrameGaussians(means=gaussians.means, rotations=quarter_turn[None])

    assert torch.allclose(propagate(anchors, frame),
                          torch.tensor([[1.0, 2.1, 3.0]], dtype=torch.float64), atol=1e-12)


def test_compacted_anchors_track_the_same_as_the_full_model() -> None:
    generator = torch.Generator().manual_seed(2)
    count = 8
    model = GaussianModel(
        means_world=torch.randn(count, 3, generator=generator) * 2.0,
        raw_quaternions=torch.randn(count, 4, generator=generator),
        raw_scales=torch.log(torch.full((count, 3), 0.4)),
        raw_opacities=torch.full((count, 1), 3.0),
        sh_dc=torch.zeros(count, 1, 3),
        sh_rest=torch.zeros(count, 15, 3),
    )
    reference = ReferenceGaussians.from_model(model)
    points = torch.cat((reference.means[[1, 4, 4]] + 0.01, torch.tensor([[50.0, 50, 50]])))
    anchors = anchor_points(points, reference)
    used, compact = compact_anchors(anchors)
    assert used.tolist() == [1, 4]

    with torch.no_grad():
        model.means_world.add_(torch.tensor([0.3, 0.0, -0.2]))
    full = propagate(anchors, FrameGaussians.from_model(model))
    assert torch.allclose(propagate(compact, frame_subset(model, used)), full)
    assert torch.equal(full[3], points[3])


# --------------------------------------------------------------------------
# cameras and median depth
# --------------------------------------------------------------------------


def _camera(size: int = 64) -> Camera:
    return Camera(
        rotation_cw=torch.eye(3),
        translation_cw=torch.zeros(3),
        camera_center_world=torch.zeros(3),
        fx=float(size), fy=float(size), cx=size / 2, cy=size / 2,
        width=size, height=size,
    )


def test_projection_and_unprojection_invert_each_other() -> None:
    rotation = _axis_rotation((0.2, 1.0, 0.1), 25.0).float()
    translation = torch.tensor([0.1, -0.2, 3.0])
    camera = Camera(
        rotation_cw=rotation,
        translation_cw=translation,
        camera_center_world=-rotation.T @ translation,
        fx=500.0, fy=480.0, cx=320.0, cy=240.0, width=640, height=480,
    )
    points = torch.tensor([[0.2, 0.1, 1.0], [-0.5, 0.3, 2.0]])
    pixels, depths = project_points(points, camera)
    assert torch.allclose(unproject_pixels(pixels, depths, camera), points, atol=1e-5)


def _two_layer_model(front_opacity: float, back_opacity: float) -> GaussianModel:
    def logit(p: float) -> float:
        return math.log(p / (1 - p))

    return GaussianModel(
        means_world=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
        raw_quaternions=torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        raw_scales=torch.log(torch.full((2, 3), 0.05)),
        raw_opacities=torch.tensor([[logit(front_opacity)], [logit(back_opacity)]]),
        sh_dc=torch.zeros(2, 1, 3),
        sh_rest=torch.zeros(2, 15, 3),
    )


@pytest.mark.parametrize(
    ("front", "back", "expected_index", "expected_depth"),
    [
        (0.8, 0.9, 0, 2.0),  # the front layer alone takes T below 0.5
        (0.3, 0.9, 1, 4.0),  # T = 0.7 after the front layer, 0.07 after the back
    ],
)
def test_median_depth_is_where_transmittance_crosses_one_half(
    front: float, back: float, expected_index: int, expected_depth: float
) -> None:
    rendering = load_config(DEFAULT_CONFIG).rendering
    camera = _camera()
    centre_and_corner = torch.tensor([[32.0, 32.0], [1.0, 1.0]])

    indices, depths = median_depth_gaussians(
        _two_layer_model(front, back), camera, rendering, centre_and_corner
    )
    assert indices[0].item() == expected_index
    assert depths[0].item() == pytest.approx(expected_depth)
    # The corner ray misses both small Gaussians.
    assert indices[1].item() == NO_GAUSSIAN
    assert math.isnan(depths[1].item())


def test_a_2d_query_follows_the_gaussian_that_set_its_depth() -> None:
    model = _two_layer_model(0.3, 0.9)
    camera = _camera()
    pixel = torch.tensor([[32.0, 32.0]])
    indices, depths = median_depth_gaussians(model, camera, load_config(DEFAULT_CONFIG).rendering, pixel)
    point = unproject_pixels(pixel, depths, camera)
    anchors = anchor_to_gaussians(point, indices, ReferenceGaussians.from_model(model))

    with torch.no_grad():
        model.means_world[1].add_(torch.tensor([0.5, 0.0, 0.0]))
    moved = propagate(anchors, FrameGaussians.from_model(model))
    pixels, _ = project_points(moved, camera)
    # 0.5 sideways at depth 4 with fx = 64 is 8 pixels.
    assert torch.allclose(pixels, torch.tensor([[40.0, 32.0]]), atol=1e-4)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _tracks(count: int = 3, frames: int = 5, dimension: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(count, frames, dimension)) * 10.0


def test_perfect_tracks_score_perfectly() -> None:
    truth = _tracks()
    metrics = track_metrics(errors_3d(truth, truth), survival_threshold=50.0)
    assert metrics.mte == 0.0
    assert metrics.delta_avg == 100.0
    assert metrics.survival == 100.0


def test_constant_translation_error() -> None:
    truth = _tracks()
    offset = np.array([0.0, 3.0, 4.0]) / 100.0  # 5 cm in metres
    predicted = truth + offset
    predicted[:, 0] = truth[:, 0]  # exact at the query frame

    metrics = track_metrics(errors_3d(predicted, truth, unit_to_cm=100.0),
                            survival_threshold=50.0)
    # MTE is the median over frames [0, 5, 5, 5, 5] of each track.
    assert metrics.mte == pytest.approx(5.0)
    # 5 cm is below 8 and 16 but not 1, 2 or 4; frame 0 does not count.
    assert metrics.delta == pytest.approx({1.0: 0.0, 2.0: 0.0, 4.0: 0.0, 8.0: 100.0, 16.0: 100.0})
    assert metrics.delta_avg == pytest.approx(40.0)
    assert metrics.survival == 100.0


def test_survival_counts_frames_until_the_first_failure() -> None:
    errors = np.array([
        [0.0, 10.0, 60.0, 10.0],  # fails at frame 2: survives 2 of 4 frames
        [0.0, 10.0, 10.0, 10.0],  # never fails
    ])
    assert track_metrics(errors, survival_threshold=50.0).survival == pytest.approx(75.0)

    # An invalid frame never fails, and the median ignores it.
    valid = np.array([[True, True, False, True], [True, True, True, True]])
    metrics = track_metrics(errors, valid, survival_threshold=50.0)
    assert metrics.survival == 100.0
    assert metrics.mte == pytest.approx(10.0)


def test_mte_is_the_mean_of_per_track_medians() -> None:
    errors = np.array([[0.0, 1.0, 2.0, 9.0], [0.0, 4.0, 4.0, 4.0]])
    # medians 1.5 (even count: the mean of the middle two) and 4.0
    assert track_metrics(errors).mte == pytest.approx((1.5 + 4.0) / 2)


def test_two_d_errors_are_normalized_to_256_pixels_per_axis() -> None:
    truth = np.zeros((2, 2, 2)) + 100.0
    predicted = truth.copy()
    predicted[0, 1] += [2.0, 0.0]  # 2 px of 512 wide -> 1 normalized px
    predicted[1, 1] += [0.0, 1.0]  # 1 px of 256 high -> 1 normalized px
    errors = errors_2d(predicted, truth, np.array([512, 256]))
    assert np.allclose(errors, [[0.0, 1.0], [0.0, 1.0]])

    per_track = errors_2d(predicted, truth, np.array([[512, 256], [256, 128]]))
    assert np.allclose(per_track[:, 1], [1.0, 2.0])


def test_in_image_mask_drops_the_border_and_undefined_points() -> None:
    truth = np.array([[[1.0, 1.0], [0.5, 5.0], [8.0, 6.0], [np.nan, 3.0]]])
    assert in_image_mask(truth, np.array([10, 8])).tolist() == [[True, False, True, False]]


def test_scene_mean_averages_scenes_not_tracks() -> None:
    first = track_metrics(np.array([[0.0, 2.0]]))
    second = track_metrics(np.array([[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]))
    mean = scene_mean([first, second])
    assert mean.mte == pytest.approx((1.0 + 2.0) / 2)
    assert mean.num_tracks == 4
