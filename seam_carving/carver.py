"""The seam-carving loop: compute energy, find a seam, remove it, repeat.

:func:`carve` does not depend on any particular algorithm. You pass it a *seam finder*: a
function that takes an ``(H, W)`` energy map and returns the cheapest vertical seam (see
:mod:`seam_carving.algorithms`). The same finder handles horizontal seams too, because the
carver runs it on the transposed energy map. That way each algorithm is written once and still
covers both orientations.

Each step of the loop (assignment items 1a-1c):

1. **Energy map.** Recompute the energy of the current image. Removing a seam changes the
   neighbours of the pixels next to it, so the old map is out of date.
2. **Find the seam.** Call the seam finder, then check the seam it returns.
3. **Remove the seam.** Delete one pixel per row (or per column), and record where those
   pixels were in the *original* image so the GUI can draw every removed seam.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .energy import EnergyFunction, sobel_energy
from .seams import (
    HORIZONTAL,
    VERTICAL,
    remove_vertical_seam,
    seam_cost,
    validate_vertical_seam,
)

SeamFinder = Callable[[np.ndarray], np.ndarray]

SEAM_ORDERS = ("width-first", "height-first", "interleaved")


class CarvingCancelled(Exception):
    """Raised inside :func:`carve` when ``cancel_event`` is set (the GUI's Stop button)."""


@dataclass(frozen=True)
class SeamRecord:
    """One removed seam, in coordinates of the original image."""

    orientation: str  # VERTICAL or HORIZONTAL
    rows: np.ndarray  # original row of every removed pixel
    cols: np.ndarray  # original column of every removed pixel
    cost: float  # total energy of the seam when it was removed


@dataclass(frozen=True)
class CarveStep:
    """What ``on_step`` gets: the image just before a seam is removed, and that seam."""

    index: int  # 1-based step number
    total: int  # total number of seams that will be removed
    orientation: str
    image: np.ndarray  # current image (H, W[, C]); the carver never changes it later
    seam: np.ndarray  # local seam: a column per row (vertical) or a row per column (horizontal)
    record: SeamRecord
    elapsed: float  # seconds since carving started


@dataclass
class CarveResult:
    """The output of :func:`carve`."""

    image: np.ndarray
    seams: list[SeamRecord]
    original_shape: tuple[int, int]
    source_rows: np.ndarray  # result.image[i, j] == original[source_rows[i, j], source_cols[i, j]]
    source_cols: np.ndarray
    total_time: float  # wall-clock seconds for the whole carve
    find_time: float  # seconds spent inside the seam finder
    energy_time: float  # seconds spent computing energy maps

    @property
    def vertical_seams(self) -> int:
        return sum(1 for s in self.seams if s.orientation == VERTICAL)

    @property
    def horizontal_seams(self) -> int:
        return sum(1 for s in self.seams if s.orientation == HORIZONTAL)

    @property
    def total_cost(self) -> float:
        """Sum of the energy of all removed seams. Lower means less visible damage."""
        return float(sum(s.cost for s in self.seams))

    @property
    def ms_per_seam(self) -> float:
        return 1000.0 * self.find_time / len(self.seams) if self.seams else 0.0


def plan_orientations(n_vertical: int, n_horizontal: int, order: str = "width-first") -> list[str]:
    """Decide the order in which vertical and horizontal seams are removed.

    * ``width-first``: all vertical seams first, then all horizontal ones.
    * ``height-first``: the reverse.
    * ``interleaved``: mix them evenly, e.g. 100 vertical and 20 horizontal gives one
      horizontal seam after every 5 vertical ones.

    (Avidan & Shamir find the *optimal* order with yet another DP over a "transport map".
    That is a possible extension to discuss in the report.)
    """
    if order == "width-first":
        return [VERTICAL] * n_vertical + [HORIZONTAL] * n_horizontal
    if order == "height-first":
        return [HORIZONTAL] * n_horizontal + [VERTICAL] * n_vertical
    if order == "interleaved":
        plan: list[str] = []
        v = h = 0
        while v < n_vertical or h < n_horizontal:
            # Take whichever orientation is furthest behind its share of the work.
            if h >= n_horizontal or (
                v < n_vertical and (v + 0.5) / n_vertical <= (h + 0.5) / n_horizontal
            ):
                plan.append(VERTICAL)
                v += 1
            else:
                plan.append(HORIZONTAL)
                h += 1
        return plan
    raise ValueError(f"Unknown seam order {order!r}. Choose one of: {', '.join(SEAM_ORDERS)}")


def _check_target(target: int, current: int, name: str) -> int:
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)):
        raise TypeError(f"target {name} must be an integer, got {target!r}")
    if not 1 <= target <= current:
        raise ValueError(
            f"target {name} must be between 1 and the current {name} ({current}), got {target}. "
            "Only shrinking is supported."
        )
    return int(target)


def carve(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    seam_finder: SeamFinder,
    energy_fn: EnergyFunction = sobel_energy,
    *,
    order: str = "width-first",
    on_step: Callable[[CarveStep], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> CarveResult:
    """Shrink ``image`` to ``target_width`` x ``target_height`` by removing low-energy seams.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` RGB or ``(H, W)`` grey image. It is not modified.
    target_width, target_height:
        The size you want, with ``1 <= target <= current size``.
    seam_finder:
        ``seam_finder(energy) -> seam``. It gets a **read-only** ``(h, w)`` energy map and must
        return a valid vertical seam. Horizontal seams are found by calling it on the
        transposed energy map.
    energy_fn:
        Energy function, see :mod:`seam_carving.energy`.
    order:
        One of :data:`SEAM_ORDERS`, see :func:`plan_orientations`.
    on_step:
        Optional callback, called once per seam *before* it is removed (the GUI uses it for the
        step-by-step animation). It may raise :class:`CarvingCancelled` to stop.
    cancel_event:
        Optional :class:`threading.Event`. When it is set, carving stops with
        :class:`CarvingCancelled`.

    Raises
    ------
    NotImplementedError
        The seam finder is still a stub.
    InvalidSeamError
        The seam finder returned an invalid seam.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected an (H, W) or (H, W, C) image, got shape {img.shape}")
    height, width = img.shape[:2]
    target_width = _check_target(target_width, width, "width")
    target_height = _check_target(target_height, height, "height")
    plan = plan_orientations(width - target_width, height - target_height, order)
    finder_name = getattr(seam_finder, "__name__", repr(seam_finder))

    # source_rows/source_cols say which original pixel each current pixel came from.
    # They shrink together with the image.
    source_rows, source_cols = np.indices((height, width))
    seams: list[SeamRecord] = []
    find_time = energy_time = 0.0
    start = time.perf_counter()

    for index, orientation in enumerate(plan, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise CarvingCancelled(f"cancelled after {index - 1} of {len(plan)} seams")

        t = time.perf_counter()
        energy = np.asarray(energy_fn(img), dtype=np.float64)
        energy_time += time.perf_counter() - t

        # Horizontal seam = vertical seam of the transposed image.
        transpose = orientation == HORIZONTAL
        work_energy = np.ascontiguousarray(energy.T if transpose else energy)
        work_energy.flags.writeable = False  # seam finders must not change their input

        t = time.perf_counter()
        seam = seam_finder(work_energy)
        find_time += time.perf_counter() - t
        seam = validate_vertical_seam(seam, *work_energy.shape, source=finder_name)

        work_img = img.swapaxes(0, 1) if transpose else img
        work_rows = source_rows.T if transpose else source_rows
        work_cols = source_cols.T if transpose else source_cols
        along = np.arange(seam.shape[0])
        record = SeamRecord(
            orientation=orientation,
            rows=work_rows[along, seam].copy(),
            cols=work_cols[along, seam].copy(),
            cost=seam_cost(work_energy, seam),
        )
        if on_step is not None:
            on_step(
                CarveStep(
                    index=index,
                    total=len(plan),
                    orientation=orientation,
                    image=img,
                    seam=seam,
                    record=record,
                    elapsed=time.perf_counter() - start,
                )
            )

        work_img = remove_vertical_seam(work_img, seam)
        work_rows = remove_vertical_seam(work_rows, seam)
        work_cols = remove_vertical_seam(work_cols, seam)
        if transpose:
            img = np.ascontiguousarray(work_img.swapaxes(0, 1))
            source_rows = np.ascontiguousarray(work_rows.T)
            source_cols = np.ascontiguousarray(work_cols.T)
        else:
            img, source_rows, source_cols = work_img, work_rows, work_cols
        seams.append(record)

    return CarveResult(
        image=img.copy() if not plan else img,
        seams=seams,
        original_shape=(height, width),
        source_rows=source_rows,
        source_cols=source_cols,
        total_time=time.perf_counter() - start,
        find_time=find_time,
        energy_time=energy_time,
    )
