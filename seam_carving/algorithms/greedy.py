"""Seam finding with a Greedy Algorithm (assignment item 3).

The interface contract is in ``seam_carving/algorithms/__init__.py``. Check your work with::

    pytest tests/test_algorithms.py -k greedy -v

Points for the report (items 3 and 5c):
* Is the result visually comparable to DP? Run ``python -m seam_carving.benchmark compare``
  for side-by-side images, seam costs and timings.
* Greedy takes O(W + H) time per seam once the energy map exists, but is not optimal. The test
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
    Each choice only looks at the next row, never at the rest of the image. Ties choose the
    leftmost allowed pixel.

    Parameters
    ----------
    energy:
        ``(H, W)`` float array, read-only.

    Returns
    -------
    np.ndarray
        ``(H,)`` integer array. ``seam[i]`` is the column removed from row ``i``.

    O(W + H) time: scan W pixels to start, then at most three per remaining row.
    O(H) space for the returned seam, with O(1) other extra space.
    """
    h, w = energy.shape
    seam = np.empty(h, dtype=np.intp)
    seam[0] = np.argmin(energy[0])
    for i in range(1, h):
        j = seam[i - 1]
        lo, hi = max(j - 1, 0), min(j + 2, w)
        seam[i] = lo + np.argmin(energy[i, lo:hi])
    return seam
