"""Turn energy maps and seams into images. Used by the GUI, the CLI and the benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .carver import SeamRecord
from .seams import VERTICAL

Color = tuple[int, int, int]

VERTICAL_SEAM_COLOR: Color = (255, 40, 40)
HORIZONTAL_SEAM_COLOR: Color = (30, 144, 255)
CURRENT_SEAM_COLOR: Color = (255, 215, 0)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as an ``(H, W, 3)`` uint8 array. Float values are taken to be 0..255."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # A grey image becomes RGB by repeating the single channel three times.
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)

    # Already colour: keep the first three channels (this drops an alpha channel).
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    raise ValueError(f"Cannot display an array of shape {arr.shape}")


def energy_to_image(energy: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Draw an energy map as a grey RGB image (bright = high energy).

    Values are scaled by the given percentile rather than the maximum, so a few very strong
    edges do not make everything else black. A square-root curve also brightens faint
    texture.
    """
    e = np.asarray(energy, dtype=np.float64)

    # What counts as "full brightness"? The 99th percentile, so a handful of very strong edges
    # don't squash everything else into black.
    top = float(np.percentile(e, percentile)) if e.size else 0.0

    # If that is 0 (e.g. a mostly flat image), fall back to the true maximum, or to 1 if the
    # whole map is zero, so we never divide by zero.
    if top <= 0.0:
        top = float(e.max()) if e.size and e.max() > 0 else 1.0

    # Scale to 0..1, clip anything brighter than `top`, then take a square root to lift faint
    # detail, and stretch to 0..255.
    gray = (np.sqrt(np.clip(e / top, 0.0, 1.0)) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def draw_seams(
    original: np.ndarray,
    seams: Iterable[SeamRecord],
    *,
    vertical_color: Color = VERTICAL_SEAM_COLOR,
    horizontal_color: Color = HORIZONTAL_SEAM_COLOR,
) -> np.ndarray:
    """Return a copy of ``original`` with every removed seam drawn on it (bonus item 7a)."""
    # Work on a copy so the caller's original image stays untouched.
    out = ensure_rgb(original).copy()
    for seam in seams:
        color = vertical_color if seam.orientation == VERTICAL else horizontal_color

        # Each SeamRecord stores the seam's pixel positions in the *original* image, so they
        # line up with `out` even though the image kept shrinking while we carved.
        out[seam.rows, seam.cols] = color
    return out


def highlight_seam(
    image: np.ndarray,
    seam: np.ndarray,
    orientation: str,
    color: Color = CURRENT_SEAM_COLOR,
) -> np.ndarray:
    """Return a copy of the *current* image with the seam about to be removed drawn on it."""
    out = ensure_rgb(image).copy()
    along = np.arange(len(seam))

    # A vertical seam has one column per row, so row `along[i]` gets column `seam[i]`. A
    # horizontal seam is the other way round: one row per column.
    if orientation == VERTICAL:
        out[along, seam] = color
    else:
        out[seam, along] = color
    return out


def pad_to_shape(
    image: np.ndarray, height: int, width: int, fill: Color = (32, 32, 32)
) -> np.ndarray:
    """Place ``image`` in the top-left corner of a ``height x width`` canvas filled with ``fill``.

    GIF frames all need the same size, so the shrinking image is padded back to the original
    size.
    """
    img = ensure_rgb(image)

    # Start with a blank canvas of the full size...
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[...] = fill

    # ...then paste the image into the top-left corner. min() makes sure we never copy more than
    # fits, even if the image is bigger than the canvas.
    h, w = min(height, img.shape[0]), min(width, img.shape[1])
    canvas[:h, :w] = img[:h, :w]
    return canvas


def montage(
    grid: Sequence[Sequence[tuple[str, np.ndarray]]],
    *,
    padding: int = 12,
    zoom: int = 1,
    background: Color = (255, 255, 255),
) -> Image.Image:
    """Arrange labelled images in a grid, e.g. one row per algorithm for the report.

    ``zoom`` enlarges every image by a whole number with nearest-neighbour sampling, so the
    seams in small test images stay visible. Each column is as wide as its widest image or
    label.
    """
    font = ImageFont.load_default()
    label_h = 18  # height reserved above each image for its caption

    # A throwaway 1x1 image, used only so Pillow can measure how wide a piece of text is.
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # Turn every numpy array into a Pillow image, keeping its label alongside.
    cells = [[(label, Image.fromarray(ensure_rgb(img))) for label, img in row] for row in grid]

    # NEAREST keeps hard pixel edges when enlarging. Smooth scaling would blur thin seams away.
    if zoom > 1:
        cells = [
            [
                (label, im.resize((im.width * zoom, im.height * zoom), Image.Resampling.NEAREST))
                for label, im in row
            ]
            for row in cells
        ]
    # First pass: measure. Each column is as wide as its widest image or caption, and each row is
    # as tall as its tallest image plus the caption strip.
    n_cols = max(len(row) for row in cells)
    col_w = [0] * n_cols
    row_h = []
    for row in cells:
        row_h.append(max(im.height for _, im in row) + label_h)
        for c, (label, im) in enumerate(row):
            text_w = int(measure.textlength(label, font=font))
            col_w[c] = max(col_w[c], im.width, text_w)

    # Second pass: draw. There is padding between cells and around the outside edge, hence the +1.
    total_w = sum(col_w) + padding * (n_cols + 1)
    total_h = sum(row_h) + padding * (len(cells) + 1)
    sheet = Image.new("RGB", (total_w, total_h), background)
    draw = ImageDraw.Draw(sheet)
    y = padding
    for r, row in enumerate(cells):
        x = padding
        for c, (label, im) in enumerate(row):
            draw.text((x, y), label, fill=(20, 20, 20), font=font)
            sheet.paste(im, (x, y + label_h))
            x += col_w[c] + padding
        y += row_h[r] + padding
    return sheet
