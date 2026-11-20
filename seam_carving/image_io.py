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
        # Phones often store a photo sideways and add a "rotate me" flag. Apply it now so the
        # picture looks the way it did on the phone.
        img = ImageOps.exif_transpose(img)

        # Convert everything (greyscale, palette, RGBA...) to plain RGB so the rest of the code
        # only has to deal with one layout.
        return np.array(img.convert("RGB"))


def save_image(array: np.ndarray, path: str | Path) -> Path:
    """Save an RGB or grey array to ``path``. The format comes from the extension."""
    path = Path(path)

    # Create the output folder if it doesn't exist yet (the CLI writes into output/ by default).
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pillow wants 8-bit values. Clip first so that values outside 0..255 don't wrap around.
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

    # Already small enough: leave it alone.
    if max(h, w) <= max_side:
        return image

    # One scale factor for both sides keeps the aspect ratio. max(1, ...) stops a very thin image
    # from rounding down to zero pixels. (Pillow wants the size as (width, height).)
    factor = max_side / max(h, w)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))

    # LANCZOS is a high-quality filter for shrinking, so we start from a clean picture.
    return np.array(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
