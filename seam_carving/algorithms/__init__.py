"""Seam-finding algorithms and the list that feeds the GUI dropdown and the CLI.

Interface contract for every seam finder
----------------------------------------
``find_seam(energy) -> seam``

* ``energy``: ``(H, W)`` float64 array, ``energy[i, j] >= 0``. It is **read-only**: call
  ``energy.astype(np.float64)`` (which copies) if you need a table to write into.
* ``seam``: ``(H,)`` integer array. ``seam[i]`` is the column removed from row ``i``, with
  ``0 <= seam[i] < W`` and ``|seam[i] - seam[i-1]| <= 1``.
* Only **vertical** seams are needed. The carver finds horizontal seams by running the same
  function on the transposed energy map.
* An unfinished finder raises ``NotImplementedError``. The GUI then shows a "not implemented
  yet" message and the tests for it are skipped.

The carver checks every seam (:func:`seam_carving.seams.validate_vertical_seam`), so a bug
shows up as a clear error message rather than a garbled image.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .dynamic_programming import find_vertical_seam_dp
from .graph_cut import find_vertical_seam_graph_cut
from .greedy import find_vertical_seam_greedy

SeamFinder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Algorithm:
    """A seam finder as it appears in the GUI dropdown and on the CLI."""

    key: str  # name on the command line, e.g. "dp"
    label: str  # text in the GUI dropdown
    find_seam: SeamFinder
    source: str  # file to edit, shown in "not implemented yet" messages
    description: str


ALGORITHMS: tuple[Algorithm, ...] = (
    Algorithm(
        key="dp",
        label="Dynamic Programming",
        find_seam=find_vertical_seam_dp,
        source="seam_carving/algorithms/dynamic_programming.py",
        description="Optimal seam from a cumulative-energy table. Target: O(H*W) per seam.",
    ),
    Algorithm(
        key="greedy",
        label="Greedy Algorithm",
        find_seam=find_vertical_seam_greedy,
        source="seam_carving/algorithms/greedy.py",
        description="Takes the cheapest neighbour row by row. O(H) per seam, but not optimal.",
    ),
    Algorithm(
        key="graphcut",
        label="Graph Cut",
        find_seam=find_vertical_seam_graph_cut,
        source="seam_carving/algorithms/graph_cut.py",
        description="Optimal seam as a minimum s-t cut (max-flow). Slow on large images.",
    ),
)


def get_algorithm(name: str) -> Algorithm:
    """Look up an algorithm by CLI key or GUI label (case-insensitive)."""
    wanted = name.strip().lower()
    for algorithm in ALGORITHMS:
        if wanted in (algorithm.key.lower(), algorithm.label.lower()):
            return algorithm
    choices = ", ".join(a.key for a in ALGORITHMS)
    raise KeyError(f"Unknown algorithm {name!r}. Choose one of: {choices}")


def is_implemented(algorithm: Algorithm) -> bool:
    """Return False if the finder is still a stub, i.e. it raises ``NotImplementedError``.

    The check calls the finder on a tiny 3x3 read-only energy map. Any other exception still
    counts as "implemented": the real error is shown when you carve with it.
    """
    probe = np.ones((3, 3))
    probe.flags.writeable = False
    try:
        algorithm.find_seam(probe)
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "SeamFinder",
    "find_vertical_seam_dp",
    "find_vertical_seam_graph_cut",
    "find_vertical_seam_greedy",
    "get_algorithm",
    "is_implemented",
]
