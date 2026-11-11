"""Shared test helpers: a brute-force oracle and simple stand-in seam finders."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import pytest

from seam_carving.algorithms import Algorithm
from seam_carving.seams import seam_cost


def all_vertical_seams(height: int, width: int) -> Iterator[np.ndarray]:
    """Yield every valid vertical seam of a ``height x width`` image.

    There are Theta(W * 3^(H-1)) of them, which is why brute force only works on tiny images
    (assignment item 2a-i).
    """
    for start in range(width):
        for moves in itertools.product((-1, 0, 1), repeat=height - 1):
            seam = np.empty(height, dtype=np.intp)
            seam[0] = start
            ok = True
            for i, step in enumerate(moves, start=1):
                seam[i] = seam[i - 1] + step
                if not 0 <= seam[i] < width:
                    ok = False
                    break
            if ok:
                yield seam


def brute_force_min_cost(energy: np.ndarray) -> float:
    """The lowest seam cost, found by trying every seam. Used to check the optimal algorithms."""
    return min(seam_cost(energy, s) for s in all_vertical_seams(*energy.shape))


def find_seam_or_skip(algorithm: Algorithm, energy: np.ndarray) -> np.ndarray:
    """Run the finder, or skip the test if it is still a stub."""
    try:
        return algorithm.find_seam(energy)
    except NotImplementedError:
        pytest.skip(f"{algorithm.label} is not implemented yet ({algorithm.source})")


# Stand-in seam finders for testing the pipeline without the real algorithms --------------


def leftmost_seam(energy: np.ndarray) -> np.ndarray:
    """Always removes column 0."""
    return np.zeros(energy.shape[0], dtype=np.intp)


def rightmost_seam(energy: np.ndarray) -> np.ndarray:
    """Always removes the last column."""
    return np.full(energy.shape[0], energy.shape[1] - 1, dtype=np.intp)


def random_walk_finder(seed: int = 0):
    """Returns a finder that produces random but valid seams (tests coordinate tracking)."""
    rng = np.random.default_rng(seed)

    def finder(energy: np.ndarray) -> np.ndarray:
        h, w = energy.shape
        seam = np.empty(h, dtype=np.intp)
        seam[0] = rng.integers(w)
        for i in range(1, h):
            seam[i] = np.clip(seam[i - 1] + rng.integers(-1, 2), 0, w - 1)
        return seam

    return finder
