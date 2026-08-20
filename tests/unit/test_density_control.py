from __future__ import annotations

import pytest
import torch
from torch import Tensor

from gaussian_splatting.renderer.projection import ProjectedGaussians
from gaussian_splatting.renderer.renderer import RenderResult
from gaussian_splatting.training.density_control import ScreenSpaceDensityStatistics


def _render_result(
    *,
    num_gaussians: int,
    original_indices: list[int],
    radii: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> RenderResult:
    visible_count = len(original_indices)
    base_means = torch.arange(
        visible_count * 2, dtype=dtype, device=device
    ).reshape(visible_count, 2)
    base_means.requires_grad_()
    means_screen = base_means + base_means.new_zeros(())
    means_screen.retain_grad()
    indices = torch.tensor(original_indices, dtype=torch.int64, device=device)
    radius_tensor = torch.tensor(radii, dtype=torch.int64, device=device)
    visible_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
    visible_mask[indices] = True

    def floating_zeros(*shape: int) -> Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    projected = ProjectedGaussians(
        original_indices=indices,
        means_camera=floating_zeros(visible_count, 3),
        means_screen=means_screen,
        depths=floating_zeros(visible_count),
        covariances_3d=floating_zeros(visible_count, 3, 3),
        covariances_2d=floating_zeros(visible_count, 2, 2),
        inverse_covariances_2d=floating_zeros(visible_count, 2, 2),
        colors=floating_zeros(visible_count, 3),
        opacities=floating_zeros(visible_count, 1),
        radii=radius_tensor,
        rectangles=torch.zeros(
            (visible_count, 4), dtype=torch.int64, device=device
        ),
    )
    return RenderResult(
        image=floating_zeros(3, 1, 1),
        final_transmittance=floating_zeros(1, 1),
        projected=projected,
        visible_mask=visible_mask,
    )


def _backward_with_gradient(render: RenderResult, gradient: Tensor) -> None:
    (render.projected.means_screen * gradient).sum().backward()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accumulate_maps_sorted_rows_to_original_indices_and_visibility(
    dtype: torch.dtype,
) -> None:
    device = torch.device("cpu")
    statistics = ScreenSpaceDensityStatistics(
        4, dtype=dtype, device=device
    )
    render = _render_result(
        num_gaussians=4,
        original_indices=[2, 0],
        radii=[4, 7],
        dtype=dtype,
        device=device,
    )
    gradient = torch.tensor(
        [[3.0, 4.0], [5.0, 12.0]], dtype=dtype, device=device
    )
    _backward_with_gradient(render, gradient)

    statistics.accumulate(render)

    torch.testing.assert_close(
        statistics.position_gradient_accumulator,
        torch.tensor([13.0, 0.0, 5.0, 0.0], dtype=dtype),
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator,
        torch.tensor([1, 0, 1, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        statistics.max_screen_radius,
        torch.tensor([7, 0, 4, 0], dtype=torch.int64),
    )


def test_multiple_observations_mean_zero_observation_and_max_radius() -> None:
    statistics = ScreenSpaceDensityStatistics(
        3, dtype=torch.float64, device="cpu"
    )
    first = _render_result(
        num_gaussians=3,
        original_indices=[1],
        radii=[9],
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    _backward_with_gradient(first, torch.tensor([[3.0, 4.0]], dtype=torch.float64))
    statistics.accumulate(first)

    second = _render_result(
        num_gaussians=3,
        original_indices=[1],
        radii=[6],
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    _backward_with_gradient(second, torch.tensor([[0.0, 12.0]], dtype=torch.float64))
    statistics.accumulate(second)

    torch.testing.assert_close(
        statistics.position_gradient_accumulator,
        torch.tensor([0.0, 17.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator,
        torch.tensor([0, 2, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        statistics.mean_position_gradient(),
        torch.tensor([0.0, 8.5, 0.0], dtype=torch.float64),
    )
    assert torch.isfinite(statistics.mean_position_gradient()).all()
    torch.testing.assert_close(
        statistics.max_screen_radius,
        torch.tensor([0, 9, 0], dtype=torch.int64),
    )


def test_reset_preserves_shape_dtype_and_device() -> None:
    statistics = ScreenSpaceDensityStatistics(
        2, dtype=torch.float64, device="cpu"
    )
    statistics.position_gradient_accumulator.copy_(torch.tensor([1.0, 2.0]))
    statistics.position_gradient_denominator.copy_(torch.tensor([1, 2]))
    statistics.max_screen_radius.copy_(torch.tensor([3, 4]))
    metadata = [
        (value.shape, value.dtype, value.device)
        for value in (
            statistics.position_gradient_accumulator,
            statistics.position_gradient_denominator,
            statistics.max_screen_radius,
        )
    ]

    statistics.reset()

    for value, expected_metadata in zip(
        (
            statistics.position_gradient_accumulator,
            statistics.position_gradient_denominator,
            statistics.max_screen_radius,
        ),
        metadata,
        strict=True,
    ):
        assert (value.shape, value.dtype, value.device) == expected_metadata
        assert torch.count_nonzero(value).item() == 0


def test_statistics_state_dict_is_detached_and_does_not_alias_runtime() -> None:
    statistics = ScreenSpaceDensityStatistics(
        3, dtype=torch.float64, device="cpu"
    )
    statistics.position_gradient_accumulator.copy_(torch.tensor([1.0, 2.0, 3.0]))
    statistics.position_gradient_denominator.copy_(torch.tensor([4, 5, 6]))
    statistics.max_screen_radius.copy_(torch.tensor([7, 8, 9]))

    state = statistics.state_dict()
    statistics.reset()

    assert all(not value.requires_grad for value in state.values())
    torch.testing.assert_close(
        state["position_gradient_accumulator"],
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["position_gradient_denominator"],
        torch.tensor([4, 5, 6]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["max_screen_radius"],
        torch.tensor([7, 8, 9]),
        rtol=0.0,
        atol=0.0,
    )


def test_statistics_load_state_dict_preserves_tensor_identity() -> None:
    statistics = ScreenSpaceDensityStatistics(
        3, dtype=torch.float32, device="cpu"
    )
    references = (
        statistics.position_gradient_accumulator,
        statistics.position_gradient_denominator,
        statistics.max_screen_radius,
    )
    state = {
        "position_gradient_accumulator": torch.tensor([1.0, 2.0, 3.0]),
        "position_gradient_denominator": torch.tensor([4, 5, 6]),
        "max_screen_radius": torch.tensor([7, 8, 9]),
    }

    statistics.load_state_dict(state)

    assert statistics.position_gradient_accumulator is references[0]
    assert statistics.position_gradient_denominator is references[1]
    assert statistics.max_screen_radius is references[2]
    torch.testing.assert_close(
        statistics.position_gradient_accumulator,
        state["position_gradient_accumulator"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator,
        state["position_gradient_denominator"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        statistics.max_screen_radius,
        state["max_screen_radius"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        ("missing", ValueError, "missing keys"),
        ("unexpected", ValueError, "unexpected keys"),
        ("shape", ValueError, "shape"),
        ("dtype", TypeError, "dtype"),
        ("device", ValueError, "device"),
        ("nonfinite", ValueError, "finite"),
        ("negative_accumulator", ValueError, "non-negative"),
        ("negative_denominator", ValueError, "non-negative"),
        ("negative_radius", ValueError, "non-negative"),
    ],
)
def test_statistics_load_rejects_invalid_state_without_partial_mutation(
    mutation: str,
    exception: type[Exception],
    message: str,
) -> None:
    statistics = ScreenSpaceDensityStatistics(
        2, dtype=torch.float32, device="cpu"
    )
    statistics.position_gradient_accumulator.copy_(torch.tensor([10.0, 20.0]))
    statistics.position_gradient_denominator.copy_(torch.tensor([3, 4]))
    statistics.max_screen_radius.copy_(torch.tensor([5, 6]))
    before = statistics.state_dict()
    state: dict[str, Tensor] = {
        "position_gradient_accumulator": torch.tensor([1.0, 2.0]),
        "position_gradient_denominator": torch.tensor([1, 2]),
        "max_screen_radius": torch.tensor([3, 4]),
    }
    if mutation == "missing":
        del state["max_screen_radius"]
    elif mutation == "unexpected":
        state["unexpected"] = torch.zeros(2)
    elif mutation == "shape":
        state["position_gradient_accumulator"] = torch.zeros(3)
    elif mutation == "dtype":
        state["position_gradient_accumulator"] = torch.zeros(2, dtype=torch.float64)
    elif mutation == "device":
        state["position_gradient_accumulator"] = torch.zeros(2, device="meta")
    elif mutation == "nonfinite":
        state["position_gradient_accumulator"][0] = float("nan")
    elif mutation == "negative_accumulator":
        state["position_gradient_accumulator"][0] = -1.0
    elif mutation == "negative_denominator":
        state["position_gradient_denominator"][0] = -1
    else:
        state["max_screen_radius"][0] = -1

    with pytest.raises(exception, match=message):
        statistics.load_state_dict(state)

    for name, expected in before.items():
        torch.testing.assert_close(
            getattr(statistics, name), expected, rtol=0.0, atol=0.0
        )


def test_append_and_keep_preserve_gaussian_index_correspondence() -> None:
    statistics = ScreenSpaceDensityStatistics(
        3, dtype=torch.float32, device="cpu"
    )
    statistics.position_gradient_accumulator.copy_(torch.tensor([1.0, 2.0, 3.0]))
    statistics.position_gradient_denominator.copy_(torch.tensor([4, 5, 6]))
    statistics.max_screen_radius.copy_(torch.tensor([7, 8, 9]))

    statistics.append_zeros(2)
    statistics.keep(torch.tensor([True, False, True, False, True]))

    assert statistics.num_gaussians == 3
    torch.testing.assert_close(
        statistics.position_gradient_accumulator, torch.tensor([1.0, 3.0, 0.0])
    )
    torch.testing.assert_close(
        statistics.position_gradient_denominator, torch.tensor([4, 6, 0])
    )
    torch.testing.assert_close(
        statistics.max_screen_radius, torch.tensor([7, 9, 0])
    )


def test_accumulate_rejects_missing_screen_gradient() -> None:
    render = _render_result(
        num_gaussians=1,
        original_indices=[0],
        radii=[1],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    statistics = ScreenSpaceDensityStatistics(
        1, dtype=torch.float32, device="cpu"
    )

    with pytest.raises(RuntimeError, match="retain_screen_grad=True"):
        statistics.accumulate(render)


def test_accumulate_allows_no_visible_gaussians_without_gradient() -> None:
    statistics = ScreenSpaceDensityStatistics(
        2, dtype=torch.float32, device="cpu"
    )
    render = _render_result(
        num_gaussians=2,
        original_indices=[],
        radii=[],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    statistics.accumulate(render)

    assert torch.count_nonzero(statistics.position_gradient_accumulator).item() == 0
    assert torch.count_nonzero(statistics.position_gradient_denominator).item() == 0
    assert torch.count_nonzero(statistics.max_screen_radius).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_accumulate_on_cuda() -> None:
    device = torch.device("cuda")
    statistics = ScreenSpaceDensityStatistics(
        2, dtype=torch.float32, device=device
    )
    render = _render_result(
        num_gaussians=2,
        original_indices=[1],
        radii=[5],
        dtype=torch.float32,
        device=device,
    )
    _backward_with_gradient(
        render, torch.tensor([[6.0, 8.0]], device=device)
    )

    statistics.accumulate(render)

    torch.testing.assert_close(
        statistics.position_gradient_accumulator,
        torch.tensor([0.0, 10.0], device=device),
    )
