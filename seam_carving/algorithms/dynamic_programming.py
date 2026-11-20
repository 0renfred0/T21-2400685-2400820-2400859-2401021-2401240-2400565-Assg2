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

    # Row 0 is the easy base case: a seam that only has one pixel so far costs exactly that
    # pixel's energy. astype() makes a copy, which matters because `energy` is read-only.
    costs = energy[0].astype(np.float64)

    # We only ever look at the previous row of costs, so there's no need to keep the whole
    # M table around. What we do have to remember is which way each cell came from, because
    # that's how we rebuild the seam afterwards. Each step is -1, 0 or +1 (left, straight,
    # right), so a tiny int8 is plenty.
    back = np.zeros((h, w), dtype=np.int8)
    columns = np.arange(w)

    # Forward pass: fill in one row at a time, top to bottom. Every cell in a row can be
    # worked out at the same time (they don't depend on each other), so we let NumPy do the
    # whole row in one go instead of looping over the columns in Python.
    for i in range(1, h):
        # Pad with +inf on both sides. That way a pixel on the edge "sees" an impossible
        # parent where the missing neighbour would be, and the min never picks it.
        padded = np.pad(costs, 1, constant_values=np.inf)

        # Stack the three candidates side by side: the first line of `parents` is the up-left
        # neighbour of every column, the second is straight up, the third is up-right.
        # So parents[:, j] holds the three options for column j.
        parents = np.stack((padded[:-2], padded[1:-1], padded[2:]))

        # Pick the cheapest candidate per column. (argmin returns the first one on a tie, so
        # ties go to the left.) choices is 0/1/2, so subtracting 1 turns it into a -1/0/+1 step.
        choices = np.argmin(parents, axis=0)
        back[i] = choices - 1

        # M[i, j] = e[i, j] + the cost of the best parent we just picked.
        costs = energy[i] + parents[choices, columns]

    # The cheapest seam finishes at the cheapest cell in the bottom row...
    seam = np.empty(h, dtype=np.intp)
    seam[-1] = np.argmin(costs)

    # ...then we walk back up, following the saved steps, to see where it came from.
    # back[i, j] says how far left or right the parent sits from column j, so adding it to
    # the current column lands us on the seam's column in the row above.
    for i in range(h - 1, 0, -1):
        seam[i - 1] = seam[i] + back[i, seam[i]]
    return seam
