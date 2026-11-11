"""Regenerate the synthetic sample images in ``samples/``.

    python scripts/make_samples.py

The images are drawn in code, so they have no licensing issues and come out the same every
time (fixed random seed). Each one shows a different seam-carving behaviour:

* castle.png        480x300: smooth sky, lightly textured grass, a stone tower and a person.
                    Seams should take out the sky and grass between the objects and leave
                    both objects intact.
* shapes.png        320x200: two textured objects on a smooth gradient. Seams go around them.
* small_shapes.png   64x40: a tiny version for slow algorithms (Graph Cut) and for CI.

Objects get texture on purpose. A flat-coloured shape has zero energy inside, so seams would
cut straight through it. Real photos rarely have that problem.

Add your own photos to ``samples/`` as well (and say where they came from in the report).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _vertical_gradient(w: int, h: int, top: tuple, bottom: tuple) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    grad = (1 - t) * np.array(top, float) + t * np.array(bottom, float)
    return np.repeat(grad, w, axis=1)


def _textured_fill(
    img: np.ndarray, mask: np.ndarray, color: tuple, rng: np.random.Generator, sigma: float
) -> None:
    """Paint ``color`` plus Gaussian noise into ``img`` wherever ``mask`` is set (in place)."""
    noise = rng.normal(0.0, sigma, mask.shape)[..., None]
    img[mask] = (np.array(color, float) + noise)[mask]


def _mask(w: int, h: int, draw_fn) -> np.ndarray:
    layer = Image.new("L", (w, h), 0)
    draw_fn(ImageDraw.Draw(layer))
    return np.asarray(layer) > 0


def castle(rng: np.random.Generator) -> Image.Image:
    w, h, horizon = 480, 300, 228
    img = _vertical_gradient(w, h, (40, 110, 200), (150, 200, 245))

    # Soft clouds: blurred white ellipses, so they have low energy.
    clouds = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(clouds)
    for cx, cy, rx, ry in [(70, 60, 60, 14), (120, 75, 45, 10), (200, 40, 70, 12)]:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=220)
    alpha = np.asarray(clouds.filter(ImageFilter.GaussianBlur(9)), float)[..., None] / 255.0
    img = img * (1 - alpha) + 255.0 * alpha

    # Grass: a gradient with light texture (some energy, but less than the tower).
    grass = _vertical_gradient(w, h - horizon, (110, 150, 60), (80, 120, 40))
    img[horizon:] = grass + rng.normal(0, 4, grass.shape[:2])[..., None]

    # Stone tower: three turrets with battlements, stone noise and mortar lines.
    turrets = [(330, 362, 70), (360, 400, 95), (396, 428, 62)]

    def tower_shape(d: ImageDraw.ImageDraw) -> None:
        for x0, x1, top in turrets:
            d.rectangle((x0, top, x1, horizon + 6), fill=255)
            for bx in range(x0, x1, 8):
                d.rectangle((bx, top - 8, bx + 4, top), fill=255)

    tower = _mask(w, h, tower_shape)
    _textured_fill(img, tower, (196, 170, 128), rng, sigma=14)
    yy, xx = np.mgrid[0:h, 0:w]
    mortar = tower & ((yy % 9 == 0) | ((xx + 8 * ((yy // 9) % 2)) % 16 == 0))
    img[mortar] *= 0.75

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    for x0, x1, top in turrets:
        draw.rectangle((x0, top, x1, horizon + 6), outline=(110, 92, 70))
        cx = (x0 + x1) // 2
        for wy in range(top + 25, horizon - 20, 40):
            draw.rectangle((cx - 3, wy, cx + 3, wy + 12), fill=(55, 45, 40))

    # A small person on the grass, far from the tower.
    draw.rectangle((58, horizon - 2, 62, horizon + 14), fill=(150, 30, 40))
    draw.ellipse((57, horizon - 8, 63, horizon - 2), fill=(230, 190, 160))
    return pil


def shapes(rng: np.random.Generator, w: int = 320, h: int = 200) -> Image.Image:
    img = _vertical_gradient(w, h, (235, 225, 200), (200, 210, 230))
    img += rng.normal(0, 1.0, (h, w))[..., None]  # slight noise, so the energy is never exactly 0
    s = w / 320
    circle = _mask(w, h, lambda d: d.ellipse((40 * s, 50 * s, 120 * s, 130 * s), fill=255))
    box = _mask(w, h, lambda d: d.rectangle((220 * s, 60 * s, 280 * s, 150 * s), fill=255))
    _textured_fill(img, circle, (220, 60, 60), rng, sigma=18)
    _textured_fill(img, box, (50, 110, 200), rng, sigma=18)

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    border = max(1, round(2 * s))
    draw.ellipse((40 * s, 50 * s, 120 * s, 130 * s), outline=(120, 20, 20), width=border)
    draw.rectangle((220 * s, 60 * s, 280 * s, 150 * s), outline=(20, 40, 90), width=border)
    draw.line((0, 170 * s, w, 170 * s), fill=(90, 90, 90), width=border)
    return pil


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    rng = np.random.default_rng(2400)
    outputs = {
        "castle.png": castle(rng),
        "shapes.png": shapes(rng),
        "small_shapes.png": shapes(rng, 64, 40),
    }
    for name, img in outputs.items():
        img.save(SAMPLES / name, optimize=True)
        print(f"wrote samples/{name} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
