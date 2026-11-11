"""Reading and writing images. Everything inside the package is an ``(H, W, 3)`` uint8 RGB array."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an ``(H, W, 3)`` uint8 RGB array.

    The EXIF orientation flag is applied, so phone photos come out the right way up.
    Transparency is dropped.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("RGB"))


def save_image(array: np.ndarray, path: str | Path) -> Path:
    """Save an RGB or grey array to ``path``. The format comes from the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def downscale_to_fit(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink ``image`` uniformly so that its longer side is at most ``max_side`` pixels.

    Seam carving re-runs the whole pipeline for every seam, so large photos are slow. The GUI
    offers to downscale big images before carving.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    factor = max_side / max(h, w)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return np.array(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
