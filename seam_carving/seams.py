"""Seam helpers shared by every algorithm: validate a seam, measure it, remove it.

A **vertical seam** is an ``(H,)`` integer array. ``seam[i]`` is the column of the pixel
removed from row ``i``. A valid seam stays inside the image and moves at most one column
between neighbouring rows, so ``|seam[i] - seam[i-1]| <= 1``.

Horizontal seams are vertical seams of the transposed image, so this module only has to deal
with vertical ones.
"""

from __future__ import annotations

import numpy as np

VERTICAL = "vertical"
HORIZONTAL = "horizontal"


class InvalidSeamError(ValueError):
    """A seam finder returned something that is not a valid seam."""


def validate_vertical_seam(
    seam: np.ndarray, height: int, width: int, *, source: str = "seam finder"
) -> np.ndarray:
    """Check that ``seam`` is a valid vertical seam for an ``height x width`` image.

    Returns the seam as an ``np.intp`` array. Raises :class:`InvalidSeamError` with a message
    that names the first problem it finds.
    """
    if seam is None:
        raise InvalidSeamError(f"{source} returned None instead of a seam array")
    arr = np.asarray(seam)
    if arr.ndim != 1 or arr.shape[0] != height:
        raise InvalidSeamError(
            f"{source} returned an array of shape {arr.shape}; expected ({height},), "
            "one column index per row"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        integral_floats = (
            np.issubdtype(arr.dtype, np.floating)
            and np.all(np.isfinite(arr))
            and np.all(arr == np.round(arr))
        )
        if not integral_floats:
            raise InvalidSeamError(f"{source} returned dtype {arr.dtype}; seams must be integers")
    arr = arr.astype(np.intp, copy=False)

    out_of_bounds = (arr < 0) | (arr >= width)
    if out_of_bounds.any():
        row = int(np.argmax(out_of_bounds))
        raise InvalidSeamError(
            f"{source}: seam[{row}] = {arr[row]} is outside the image "
            f"(valid columns are 0..{width - 1})"
        )

    jumps = np.abs(np.diff(arr)) > 1
    if jumps.any():
        row = int(np.argmax(jumps))
        raise InvalidSeamError(
            f"{source}: seam is not connected between rows {row} and {row + 1} "
            f"(columns {arr[row]} -> {arr[row + 1]}); neighbouring rows may differ by at most 1"
        )
    return arr


def seam_cost(energy: np.ndarray, seam: np.ndarray) -> float:
    """The total energy of the pixels on ``seam``, which is what seam carving minimises."""
    seam = np.asarray(seam, dtype=np.intp)
    return float(np.asarray(energy)[np.arange(seam.shape[0]), seam].sum())


def remove_vertical_seam(array: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Return a copy of ``array`` with one pixel per row (the seam) removed.

    Works for ``(H, W)`` and ``(H, W, C)`` arrays. All other pixels keep their left-to-right
    order. Runs in O(H * W) time.
    """
    h, w = array.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return array[mask].reshape((h, w - 1) + array.shape[2:])
