"""Correctness tests for the three seam finders.

A finder that still raises ``NotImplementedError`` has its tests **skipped**, not failed, so
CI stays green while the work is in progress. ``pytest -ra`` lists the skipped tests at the
end, which doubles as a TODO list.

Run the tests for one algorithm with ``pytest tests/test_algorithms.py -k dp`` (or ``greedy``
or ``graphcut``).

What these tests establish (useful evidence for the report):
* every finder returns a *valid* seam for all shapes, including 1xN, Nx1 and 1x1;
* finders do not modify their input and accept read-only and integer energy maps;
* DP and Graph Cut are *optimal*: same cost as brute force over every possible seam;
* Graph Cut and DP agree on cost on larger random maps;
* Greedy follows its local rule, is never better than DP, and can be much worse;
* every algorithm works end to end through the carver.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam_carving.algorithms import ALGORITHMS, get_algorithm
from seam_carving.carver import carve
from seam_carving.seams import seam_cost, validate_vertical_seam
from tests.helpers import brute_force_min_cost, find_seam_or_skip

ALL = pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.key)
OPTIMAL = pytest.mark.parametrize(
    "algorithm", [a for a in ALGORITHMS if a.key in ("dp", "graphcut")], ids=lambda a: a.key
)
DP = get_algorithm("dp")
GREEDY = get_algorithm("greedy")
GRAPH_CUT = get_algorithm("graphcut")

SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (5, 5), (8, 3), (3, 8), (20, 30)]


def _read_only(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64)
    arr.flags.writeable = False
    return arr


# --- properties every algorithm must have -------------------------------------------------


@ALL
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_returns_valid_seam(algorithm, shape, rng):
    energy = _read_only(rng.random(shape))
    seam = find_seam_or_skip(algorithm, energy)
    validate_vertical_seam(seam, *shape, source=algorithm.label)


@ALL
def test_does_not_modify_input(algorithm, rng):
    energy = rng.random((6, 7))
    before = energy.copy()
    find_seam_or_skip(algorithm, energy)
    np.testing.assert_array_equal(energy, before)


@ALL
def test_accepts_integer_energy(algorithm):
    energy = np.array([[3, 1, 2], [2, 9, 1], [1, 1, 5]], dtype=np.int64)
    seam = find_seam_or_skip(algorithm, energy)
    validate_vertical_seam(seam, 3, 3)


@ALL
def test_single_column_is_the_only_seam(algorithm, rng):
    seam = find_seam_or_skip(algorithm, _read_only(rng.random((9, 1))))
    np.testing.assert_array_equal(seam, 0)


@ALL
def test_follows_a_zero_energy_valley(algorithm):
    """A connected zero-energy path through high energy: every algorithm should find it."""
    energy = np.full((6, 6), 10.0)
    path = [4, 3, 3, 2, 1, 2]
    energy[np.arange(6), path] = 0.0
    seam = find_seam_or_skip(algorithm, _read_only(energy))
    np.testing.assert_array_equal(seam, path)


# --- optimality (DP and Graph Cut) ---------------------------------------------------------


@OPTIMAL
@pytest.mark.parametrize("seed", range(30))
def test_matches_brute_force_optimum(algorithm, seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 6)), int(rng.integers(1, 7)))
    energy = _read_only(rng.random(shape))
    seam = find_seam_or_skip(algorithm, energy)
    validate_vertical_seam(seam, *shape)
    assert seam_cost(energy, seam) == pytest.approx(brute_force_min_cost(energy))


@OPTIMAL
@pytest.mark.parametrize("column", ["first", "last"])
def test_optimal_seam_on_the_border(algorithm, column):
    """The cheapest seam is the first or last column.

    Graph Cut: an INF edge from the last column to the sink makes this fail for "last". See
    the GOTCHA note in graph_cut.py.
    """
    energy = np.ones((5, 6))
    energy[:, 0 if column == "first" else -1] = 0.0
    seam = find_seam_or_skip(algorithm, _read_only(energy))
    assert seam_cost(energy, seam) == 0.0


@OPTIMAL
def test_uniform_energy(algorithm):
    energy = np.full((4, 5), 2.5)
    seam = find_seam_or_skip(algorithm, _read_only(energy))
    assert seam_cost(energy, seam) == pytest.approx(4 * 2.5)


@pytest.mark.parametrize("seed", range(3))
def test_graph_cut_matches_dp_on_larger_maps(seed):
    rng = np.random.default_rng(100 + seed)
    energy = _read_only(rng.random((15, 20)) * 255)
    dp_seam = find_seam_or_skip(DP, energy)
    gc_seam = find_seam_or_skip(GRAPH_CUT, energy)
    validate_vertical_seam(gc_seam, *energy.shape)
    assert seam_cost(energy, gc_seam) == pytest.approx(seam_cost(energy, dp_seam))


# --- Greedy ---------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_greedy_follows_local_rule(seed):
    rng = np.random.default_rng(seed)
    energy = _read_only(rng.random((10, 8)))  # random floats, so there are no ties
    seam = find_seam_or_skip(GREEDY, energy)
    assert seam[0] == np.argmin(energy[0])
    for i in range(1, energy.shape[0]):
        lo, hi = max(seam[i - 1] - 1, 0), min(seam[i - 1] + 1, energy.shape[1] - 1)
        assert seam[i] == lo + np.argmin(energy[i, lo : hi + 1]), f"row {i}"


def test_greedy_is_not_always_optimal():
    """Greedy starts at the cheapest top-row pixel (0) and gets trapped in an expensive column.
    DP starts at 1 and follows a zero-cost path."""
    energy = _read_only([[0.0, 1.0, 5.0], [9.0, 9.5, 0.0], [9.0, 9.5, 0.0]])
    greedy = find_seam_or_skip(GREEDY, energy)
    np.testing.assert_array_equal(greedy, [0, 0, 0])
    assert seam_cost(energy, greedy) == pytest.approx(18.0)
    dp = find_seam_or_skip(DP, energy)
    np.testing.assert_array_equal(dp, [1, 2, 2])
    assert seam_cost(energy, dp) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(10))
def test_greedy_never_beats_dp(seed):
    rng = np.random.default_rng(seed)
    energy = _read_only(rng.random((12, 12)))
    greedy = seam_cost(energy, find_seam_or_skip(GREEDY, energy))
    dp = seam_cost(energy, find_seam_or_skip(DP, energy))
    assert greedy >= dp - 1e-9


# --- end to end through the carver ----------------------------------------------------------


@ALL
def test_carves_an_image_end_to_end(algorithm, rgb_image):
    find_seam_or_skip(algorithm, _read_only(np.ones((3, 3))))  # skip stubs early
    result = carve(rgb_image, 11, 9, algorithm.find_seam, order="interleaved")
    assert result.image.shape == (9, 11, 3)
    np.testing.assert_array_equal(rgb_image[result.source_rows, result.source_cols], result.image)
