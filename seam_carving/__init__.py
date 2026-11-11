"""Seam carving (content-aware image resizing) for SIT Assignment 2, Team T21.

Package layout::

    energy.py         energy maps (gradient, Sobel)
    seams.py          seam checking, cost and removal helpers
    carver.py         the carving loop (energy -> seam -> remove -> repeat)
    algorithms/       seam finders: Dynamic Programming, Greedy, Graph Cut
    visualization.py  seam overlays, energy images, montages
    image_io.py       loading and saving images
    gui.py            Tkinter desktop app      (python main.py)
    cli.py            command-line interface   (python -m seam_carving.cli)
    benchmark.py      algorithm comparison     (python -m seam_carving.benchmark)
"""

from .algorithms import ALGORITHMS, get_algorithm, is_implemented
from .carver import CarveResult, CarvingCancelled, carve
from .energy import ENERGY_OPTIONS, get_energy
from .seams import InvalidSeamError

__version__ = "0.1.0"

__all__ = [
    "ALGORITHMS",
    "ENERGY_OPTIONS",
    "CarveResult",
    "CarvingCancelled",
    "InvalidSeamError",
    "carve",
    "get_algorithm",
    "get_energy",
    "is_implemented",
]
