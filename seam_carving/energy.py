"""Energy maps: how important each pixel is.

Every energy function has the signature ``energy_fn(image) -> energy`` where

* ``image`` is an ``(H, W, 3)`` RGB array (or an ``(H, W)`` grey image), any numeric dtype;
* ``energy`` is an ``(H, W)`` ``float64`` array with ``energy >= 0``.

Pixels on edges and in textured areas get high energy. Pixels in smooth areas get low
energy, so a seam through them can be removed without anyone noticing.

The filters are written with plain NumPy array slicing instead of OpenCV or SciPy. That keeps
the dependencies small, and the convolution is easy to explain in the report. Each one runs in
O(H * W) time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EnergyFunction = Callable[[np.ndarray], np.ndarray]

# ITU-R BT.601 luma weights (the same weights OpenCV uses for RGB -> grey).
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])

SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T.copy()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB (or already grey) image to a float64 intensity image."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0].astype(np.float64)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[..., :3].astype(np.float64) @ _LUMA_WEIGHTS
    raise ValueError(f"Expected an (H, W) or (H, W, 3) image, got shape {image.shape}")


def correlate3x3(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Filter ``gray`` with a 3x3 ``kernel``, repeating edge pixels at the border.

    The result is the sum of 9 shifted copies of the padded image, each multiplied by one
    kernel weight. That is O(H * W) work, with NumPy running the inner loops.

    Strictly speaking this is correlation, not convolution: the kernel is not flipped. For the
    Sobel kernels flipping only changes the sign, and the sign disappears in the magnitude.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 kernel, got shape {kernel.shape}")
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    out = np.zeros((h, w), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            weight = kernel[di, dj]
            if weight != 0.0:
                out += weight * padded[di : di + h, dj : dj + w]
    return out


def gradient_energy(image: np.ndarray) -> np.ndarray:
    """The e1 energy from Avidan & Shamir (2007): ``|dI/dx| + |dI/dy|``.

    The derivatives are central differences (``I[i, j+1] - I[i, j-1]``) on the grey image,
    with edge pixels repeated at the border.
    """
    gray = to_grayscale(image)
    padded = np.pad(gray, 1, mode="edge")
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    return np.abs(dx) + np.abs(dy)


def sobel_energy(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude ``sqrt(Gx^2 + Gy^2)`` from the 3x3 Sobel kernels.

    Sobel smooths across the edge direction, so it is less sensitive to noise than
    :func:`gradient_energy`.
    """
    gray = to_grayscale(image)
    gx = correlate3x3(gray, SOBEL_X)
    gy = correlate3x3(gray, SOBEL_Y)
    return np.hypot(gx, gy)


@dataclass(frozen=True)
class EnergyOption:
    """An energy function as it appears in the GUI dropdown and on the CLI."""

    key: str  # name on the command line, e.g. "sobel"
    label: str  # text in the GUI dropdown
    fn: EnergyFunction
    description: str


ENERGY_OPTIONS: tuple[EnergyOption, ...] = (
    EnergyOption(
        "sobel",
        "Sobel filter",
        sobel_energy,
        "Gradient magnitude from the 3x3 Sobel kernels. Smoother and more robust to noise.",
    ),
    EnergyOption(
        "gradient",
        "Gradient magnitude (L1)",
        gradient_energy,
        "|dI/dx| + |dI/dy| from central differences (e1 in Avidan & Shamir 2007).",
    ),
)
DEFAULT_ENERGY = ENERGY_OPTIONS[0]


def get_energy(name: str) -> EnergyOption:
    """Look up an energy option by CLI key or GUI label (case-insensitive)."""
    wanted = name.strip().lower()
    for option in ENERGY_OPTIONS:
        if wanted in (option.key.lower(), option.label.lower()):
            return option
    choices = ", ".join(option.key for option in ENERGY_OPTIONS)
    raise KeyError(f"Unknown energy function {name!r}. Choose one of: {choices}")
