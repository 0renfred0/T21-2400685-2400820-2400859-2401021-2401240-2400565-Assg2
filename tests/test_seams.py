from __future__ import annotations

import numpy as np
import pytest

from seam_carving.seams import (
    InvalidSeamError,
    remove_vertical_seam,
    seam_cost,
    validate_vertical_seam,
)
from tests.helpers import all_vertical_seams


def test_remove_vertical_seam_2d():
    arr = np.arange(12).reshape(3, 4)
    out = remove_vertical_seam(arr, np.array([0, 1, 3]))
    np.testing.assert_array_equal(out, [[1, 2, 3], [4, 6, 7], [8, 9, 10]])


def test_remove_vertical_seam_3d_keeps_channels():
    arr = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    out = remove_vertical_seam(arr, np.array([2, 1]))
    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out[0], arr[0, :2])
    np.testing.assert_array_equal(out[1], arr[1, [0, 2]])


def test_remove_vertical_seam_does_not_modify_input():
    arr = np.arange(9).reshape(3, 3)
    before = arr.copy()
    remove_vertical_seam(arr, np.array([1, 1, 1]))
    np.testing.assert_array_equal(arr, before)


def test_seam_cost():
    energy = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    assert seam_cost(energy, np.array([0, 1, 0])) == 1 + 4 + 5


def test_validate_accepts_valid_seam_and_integral_floats():
    np.testing.assert_array_equal(validate_vertical_seam([0, 1, 2, 1], 4, 3), [0, 1, 2, 1])
    out = validate_vertical_seam(np.array([1.0, 0.0]), 2, 2)
    assert np.issubdtype(out.dtype, np.integer)


@pytest.mark.parametrize(
    "seam, match",
    [
        (None, "None"),
        ([0, 0], "shape"),  # too short for 3 rows
        ([[0], [0], [0]], "shape"),  # 2-D
        ([0.5, 0, 0], "integers"),
        ([True, False, True], "integers"),
        ([0, 0, 3], "outside"),
        ([-1, 0, 0], "outside"),
        ([0, 2, 2], "not connected"),
    ],
)
def test_validate_rejects_invalid_seams(seam, match):
    with pytest.raises(InvalidSeamError, match=match):
        validate_vertical_seam(seam, 3, 3)


@pytest.mark.parametrize("h, w, expected", [(1, 1, 1), (1, 4, 4), (2, 2, 4), (3, 3, 17), (4, 1, 1)])
def test_seam_enumeration_counts(h, w, expected):
    seams = list(all_vertical_seams(h, w))
    assert len(seams) == expected
    for seam in seams:
        validate_vertical_seam(seam, h, w)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_number_of_seams_grows_at_least_exponentially(n):
    """Item 2a-i: with n > 1 columns there are at least 2^(m-1) seams for m rows."""
    counts = [len(list(all_vertical_seams(m, n))) for m in range(1, 9)]
    for m, count in enumerate(counts, start=1):
        assert count >= 2 ** (m - 1)
    # Each extra row multiplies the count by at least 2 (for n = 2 it is exactly 2).
    assert all(b >= 2 * a for a, b in zip(counts, counts[1:], strict=False))
