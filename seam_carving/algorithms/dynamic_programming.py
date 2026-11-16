"""Seam finding with Dynamic Programming (assignment items 1 and 2).

The interface contract is in ``seam_carving/algorithms/__init__.py``. Check your work with::

    pytest tests/test_algorithms.py -k dp -v

Points for the report (items 2 and 5):
* Why does brute force need exponential time? (``tests/helpers.py`` counts the seams.)
* What is the optimal substructure, i.e. how does ``M[i, j]`` depend on the subproblems?
* Time and space complexity of one seam and of removing ``k`` seams.
* Do we need to keep the whole DP table? (Only the previous row of ``M`` is needed for the
  costs. The back-pointers can be stored as int8 offsets in {-1, 0, +1}.)
* Does multithreading help? (The cells in one row do not depend on each other. Each row
  depends on the row above it.)
"""

from __future__ import annotations

import numpy as np


def find_vertical_seam_dp(energy: np.ndarray) -> np.ndarray:
    """Return the vertical seam with the lowest total energy, found by dynamic programming.

    Parameters
    ----------
    energy:
        ``(H, W)`` float array, read-only. ``energy[i, j]`` is the cost of removing pixel
        ``(i, j)``.

    Returns
    -------
    np.ndarray
        ``(H,)`` integer array. ``seam[i]`` is the column removed from row ``i``, and
        ``|seam[i] - seam[i-1]| <= 1``.

    Recurrence (Avidan & Shamir 2007), with ``M[i, j]`` the cheapest total energy of a seam
    that starts in the top row and ends at ``(i, j)``::

        M[0, j] = e[0, j]
        M[i, j] = e[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])   # ignore columns off the edge

    The cheapest seam ends at ``argmin_j M[H-1, j]``. Follow the back-pointers up to row 0 to
    recover it. O(H * W) time and O(H * W) space per seam. Costs use a rolling row;
    only int8 parent offsets are stored for the full image. Ties choose the leftmost parent.
    """
    h, w = energy.shape
    costs = energy[0].astype(np.float64)
    back = np.zeros((h, w), dtype=np.int8)
    columns = np.arange(w)
    for i in range(1, h):
        padded = np.pad(costs, 1, constant_values=np.inf)
        parents = np.stack((padded[:-2], padded[1:-1], padded[2:]))
        choices = np.argmin(parents, axis=0)
        back[i] = choices - 1
        costs = energy[i] + parents[choices, columns]

    seam = np.empty(h, dtype=np.intp)
    seam[-1] = np.argmin(costs)
    for i in range(h - 1, 0, -1):
        seam[i - 1] = seam[i] + back[i, seam[i]]
    return seam
