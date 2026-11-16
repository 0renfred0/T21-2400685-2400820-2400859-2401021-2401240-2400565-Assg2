"""Tie-heavy min-cut cases, including maps without an augmenting path."""

import itertools

import numpy as np
import pytest

from seam_carving.algorithms.graph_cut import find_vertical_seam_graph_cut
from seam_carving.seams import seam_cost, validate_vertical_seam
from tests.helpers import brute_force_min_cost


def test_binary_maps_match_brute_force():
    for pixels in itertools.product((0.0, 1.0), repeat=9):
        energy = np.array(pixels).reshape(3, 3)
        seam = find_vertical_seam_graph_cut(energy)
        validate_vertical_seam(seam, 3, 3)
        assert seam_cost(energy, seam) == brute_force_min_cost(energy)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_energy_scale_does_not_change_optimum(scale):
    energy = scale * np.array([[0, 1, 4], [5, 4, 0], [7, 2, 0]])
    seam = find_vertical_seam_graph_cut(energy)
    np.testing.assert_array_equal(seam, [1, 2, 2])
    assert seam_cost(energy, seam) == pytest.approx(scale, rel=1e-12, abs=0)
