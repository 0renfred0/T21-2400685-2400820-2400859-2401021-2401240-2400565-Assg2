"""Seam finding with Dynamic Programming (assignment items 1 and 2).

Owner:  <name / SIT ID>
Status: TODO. The function below is a stub. Write your implementation in its body.

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
    recover it. Target: O(H * W) time per seam.
    """
    # TODO(team): implement the DP, then delete the `raise` at the bottom.
    #
    # Suggested steps:
    #   1. h, w = energy.shape
    #      M = energy.astype(np.float64)    # astype copies; `energy` itself is read-only
    #      back = np.zeros((h, w), dtype=np.intp)   # column of the best parent in row i-1
    #
    #   2. Forward pass, for i in 1 .. h-1:
    #        For every column j, look at the up to 3 parents M[i-1, j-1 .. j+1].
    #        Store the cheapest one in back[i, j] and set M[i, j] = energy[i, j] + that cost.
    #        Hint: avoid a Python loop over j. Pad M[i-1] with +inf on both sides,
    #        stack the left / middle / right shifts into a (3, w) array, then use
    #        np.argmin along axis 0. That keeps the whole pass O(H * W) and fast in NumPy.
    #
    #   3. Backtrack:
    #        seam = np.empty(h, dtype=np.intp)
    #        seam[-1] = np.argmin(M[-1])
    #        for i in range(h - 1, 0, -1): seam[i - 1] = back[i, seam[i]]
    #
    #   4. return seam
    #
    # Edge cases the tests check: w == 1 (the only seam is all zeros), h == 1, and integer
    # energies.
    raise NotImplementedError(
        "Dynamic Programming seam finder is not implemented yet "
        "(seam_carving/algorithms/dynamic_programming.py)"
    )
