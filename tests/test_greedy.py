"""Deterministic tie-breaking in the Greedy seam finder."""

import numpy as np

from seam_carving.algorithms import find_vertical_seam_greedy


def test_greedy_breaks_ties_to_the_left():
    energy = np.array([[9, 9, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]])
    seam = find_vertical_seam_greedy(energy)
    np.testing.assert_array_equal(seam, [2, 1, 0])
