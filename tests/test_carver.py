"""Tests for the carving loop, run with stand-in seam finders.

They do not depend on the real algorithms, so they pass while those are still stubs.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from seam_carving.carver import CarvingCancelled, carve, plan_orientations
from seam_carving.energy import gradient_energy, sobel_energy
from seam_carving.seams import HORIZONTAL, VERTICAL, InvalidSeamError
from tests.helpers import leftmost_seam, random_walk_finder, rightmost_seam


def test_no_op_returns_copy(rgb_image):
    result = carve(rgb_image, 16, 12, leftmost_seam)
    np.testing.assert_array_equal(result.image, rgb_image)
    assert result.image is not rgb_image
    assert result.seams == []


def test_vertical_leftmost_removes_first_columns(rgb_image):
    result = carve(rgb_image, 10, 12, leftmost_seam)
    np.testing.assert_array_equal(result.image, rgb_image[:, 6:])
    # Seam k removed original column k from every row.
    for k, seam in enumerate(result.seams):
        assert seam.orientation == VERTICAL
        np.testing.assert_array_equal(seam.cols, k)
        np.testing.assert_array_equal(seam.rows, np.arange(12))


def test_vertical_rightmost_removes_last_columns(rgb_image):
    result = carve(rgb_image, 10, 12, rightmost_seam)
    np.testing.assert_array_equal(result.image, rgb_image[:, :10])


def test_horizontal_seams_use_transposed_image(rgb_image):
    """With the transposed energy map, "leftmost column" means "top row" of the image."""
    result = carve(rgb_image, 16, 7, leftmost_seam)
    np.testing.assert_array_equal(result.image, rgb_image[5:, :])
    for k, seam in enumerate(result.seams):
        assert seam.orientation == HORIZONTAL
        np.testing.assert_array_equal(seam.rows, k)
        np.testing.assert_array_equal(seam.cols, np.arange(16))


@pytest.mark.parametrize("order", ["width-first", "height-first", "interleaved"])
@pytest.mark.parametrize("seed", range(5))
def test_coordinate_tracking_with_random_seams(rgb_image, order, seed):
    """Every result pixel maps back to its original pixel, and removed + kept = all pixels."""
    result = carve(rgb_image, 9, 8, random_walk_finder(seed), order=order)
    assert result.image.shape == (8, 9, 3)
    assert (result.vertical_seams, result.horizontal_seams) == (7, 4)
    np.testing.assert_array_equal(rgb_image[result.source_rows, result.source_cols], result.image)

    removed = np.zeros(rgb_image.shape[:2], dtype=int)
    for seam in result.seams:
        removed[seam.rows, seam.cols] += 1
    kept = np.zeros_like(removed)
    kept[result.source_rows, result.source_cols] += 1
    assert removed.max() == 1, "a pixel was removed twice"
    np.testing.assert_array_equal(removed + kept, 1)


def test_vertical_only_result_equals_original_minus_removed_pixels(rgb_image):
    result = carve(rgb_image, 11, 12, random_walk_finder(7))
    mask = np.ones(rgb_image.shape[:2], dtype=bool)
    for seam in result.seams:
        mask[seam.rows, seam.cols] = False
    np.testing.assert_array_equal(result.image, rgb_image[mask].reshape(12, 11, 3))


def test_grayscale_images_are_supported(rng):
    gray = rng.integers(0, 256, size=(10, 10), dtype=np.uint8)
    result = carve(gray, 6, 7, random_walk_finder(1), gradient_energy)
    assert result.image.shape == (7, 6)


def test_down_to_one_pixel(rgb_image):
    result = carve(rgb_image, 1, 1, random_walk_finder(3), order="interleaved")
    assert result.image.shape == (1, 1, 3)
    assert len(result.seams) == (16 - 1) + (12 - 1)


def test_seam_costs_match_energy_of_removed_pixels(rgb_image):
    result = carve(rgb_image, 15, 12, leftmost_seam, sobel_energy)
    assert result.seams[0].cost == pytest.approx(sobel_energy(rgb_image)[:, 0].sum())
    assert result.total_cost == pytest.approx(result.seams[0].cost)


def test_timings_are_recorded(rgb_image):
    result = carve(rgb_image, 12, 10, random_walk_finder())
    assert result.total_time >= result.find_time >= 0
    assert result.total_time >= result.energy_time >= 0
    assert result.ms_per_seam >= 0


def test_on_step_called_once_per_seam_before_removal(rgb_image):
    steps = []
    carve(rgb_image, 13, 10, leftmost_seam, on_step=steps.append, order="interleaved")
    assert [s.index for s in steps] == list(range(1, 6))
    assert all(s.total == 5 for s in steps)
    first = steps[0]
    assert first.image.shape == rgb_image.shape  # the image *before* the first removal
    expected = (12,) if first.orientation == VERTICAL else (16,)
    assert first.seam.shape == expected


def test_finder_receives_read_only_energy(rgb_image):
    def mutating_finder(energy):
        energy[0, 0] = -1  # not allowed
        return leftmost_seam(energy)

    with pytest.raises(ValueError, match="read-only"):
        carve(rgb_image, 15, 12, mutating_finder)


def test_invalid_seam_is_reported(rgb_image):
    def broken_finder(energy):
        return 2 * np.arange(energy.shape[0])  # jumps two columns per row

    with pytest.raises(InvalidSeamError, match="broken_finder"):
        carve(rgb_image, 15, 12, broken_finder)


def test_not_implemented_propagates(rgb_image):
    def stub(energy):
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        carve(rgb_image, 15, 12, stub)


def test_cancel_event_stops_carving(rgb_image):
    event = threading.Event()

    def cancel_after_two(step):
        if step.index == 2:
            event.set()

    with pytest.raises(CarvingCancelled):
        carve(rgb_image, 8, 12, leftmost_seam, on_step=cancel_after_two, cancel_event=event)


@pytest.mark.parametrize("tw, th", [(0, 12), (17, 12), (16, 13), (16, 0)])
def test_invalid_targets(rgb_image, tw, th):
    with pytest.raises(ValueError):
        carve(rgb_image, tw, th, leftmost_seam)


def test_non_integer_target(rgb_image):
    with pytest.raises(TypeError):
        carve(rgb_image, 10.5, 12, leftmost_seam)


@pytest.mark.parametrize("nv, nh", [(0, 0), (5, 0), (0, 5), (100, 20), (3, 7)])
@pytest.mark.parametrize("order", ["width-first", "height-first", "interleaved"])
def test_plan_orientations_counts(nv, nh, order):
    plan = plan_orientations(nv, nh, order)
    assert plan.count(VERTICAL) == nv
    assert plan.count(HORIZONTAL) == nh


def test_plan_interleaved_is_evenly_spread():
    plan = plan_orientations(100, 20, "interleaved")
    positions = [i for i, o in enumerate(plan) if o == HORIZONTAL]
    gaps = np.diff(positions)
    assert gaps.min() >= 5 and gaps.max() <= 7


def test_plan_unknown_order():
    with pytest.raises(ValueError):
        plan_orientations(1, 1, "random")
