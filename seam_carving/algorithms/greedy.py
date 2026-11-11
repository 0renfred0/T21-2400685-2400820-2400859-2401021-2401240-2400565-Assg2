"""Seam finding with a Greedy Algorithm (assignment item 3).

Owner:  <name / SIT ID>
Status: TODO. The function below is a stub. Write your implementation in its body.

The interface contract is in ``seam_carving/algorithms/__init__.py``. Check your work with::

    pytest tests/test_algorithms.py -k greedy -v

Points for the report (items 3 and 5c):
* Is the result visually comparable to DP? Run ``python -m seam_carving.benchmark compare``
  for side-by-side images, seam costs and timings.
* Greedy is fast per seam (O(H) once the energy map exists) but not optimal. The test
  ``test_greedy_is_not_always_optimal`` has a 3x3 example where it is much worse.
* Recomputing the energy map is O(H * W), so does greedy actually make *carving* faster?
* Possible variant: run greedy from every top-row column and keep the cheapest seam. That is
  O(H * W) per seam, so how does it compare with DP?
"""

from __future__ import annotations

import numpy as np


def find_vertical_seam_greedy(energy: np.ndarray) -> np.ndarray:
    """Return a vertical seam built greedily, one row at a time.

    Start at the lowest-energy pixel of the top row. In every following row, move to whichever
    of the (up to) three neighbours below (``j-1``, ``j``, ``j+1``) has the lowest energy.
    Each choice only looks at the next row, never at the rest of the image.

    Parameters
    ----------
    energy:
        ``(H, W)`` float array, read-only.

    Returns
    -------
    np.ndarray
        ``(H,)`` integer array. ``seam[i]`` is the column removed from row ``i``.

    Target: O(H) time and O(H) extra space per seam.
    """
    # TODO(team): implement the greedy walk, then delete the `raise` at the bottom.
    #
    # Suggested steps:
    #   1. h, w = energy.shape
    #      seam = np.empty(h, dtype=np.intp)
    #      seam[0] = np.argmin(energy[0])     # np.argmin breaks ties by taking the leftmost
    #
    #   2. for i in range(1, h):
    #          j = seam[i - 1]
    #          lo, hi = max(j - 1, 0), min(j + 1, w - 1)    # stay inside the image
    #          seam[i] = lo + np.argmin(energy[i, lo : hi + 1])
    #
    #   3. return seam
    #
    # The tests check that seam[0] is the top-row minimum and that every step picks the
    # cheapest of its neighbours. So implement exactly this rule, and describe any variant
    # separately in the report.
    raise NotImplementedError(
        "Greedy seam finder is not implemented yet (seam_carving/algorithms/greedy.py)"
    )
