from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from seam_carving.carver import carve
from seam_carving.image_io import downscale_to_fit, load_image, save_image
from seam_carving.seams import HORIZONTAL, VERTICAL
from seam_carving.visualization import (
    CURRENT_SEAM_COLOR,
    HORIZONTAL_SEAM_COLOR,
    VERTICAL_SEAM_COLOR,
    draw_seams,
    energy_to_image,
    ensure_rgb,
    highlight_seam,
    montage,
    pad_to_shape,
)
from tests.helpers import leftmost_seam


def test_load_and_save_roundtrip(tmp_path, rgb_image):
    path = save_image(rgb_image, tmp_path / "sub" / "img.png")
    np.testing.assert_array_equal(load_image(path), rgb_image)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_load_converts_to_rgb(tmp_path, mode):
    Image.new(mode, (5, 4)).save(tmp_path / "x.png")
    img = load_image(tmp_path / "x.png")
    assert img.shape == (4, 5, 3) and img.dtype == np.uint8


def test_downscale_to_fit():
    img = np.zeros((300, 1200, 3), dtype=np.uint8)
    assert downscale_to_fit(img, 600).shape == (150, 600, 3)
    assert downscale_to_fit(img, 2000) is img


def test_ensure_rgb_variants():
    assert ensure_rgb(np.zeros((2, 3))).shape == (2, 3, 3)
    assert ensure_rgb(np.zeros((2, 3, 1))).shape == (2, 3, 3)
    assert ensure_rgb(np.zeros((2, 3, 4), dtype=np.uint8)).shape == (2, 3, 3)
    assert ensure_rgb(np.full((1, 1), 300.0))[0, 0, 0] == 255


def test_energy_to_image():
    energy = np.array([[0.0, 1.0], [4.0, 100.0]])
    img = energy_to_image(energy, percentile=100)
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    assert img[0, 0, 0] == 0 and img[1, 1, 0] == 255
    assert energy_to_image(np.zeros((2, 2))).max() == 0


def test_draw_seams_colours_removed_pixels(rgb_image):
    result = carve(rgb_image, 14, 10, leftmost_seam, order="width-first")
    overlay = draw_seams(rgb_image, result.seams)
    assert (overlay[:, 0] == VERTICAL_SEAM_COLOR).all()
    assert (overlay[:, 1] == VERTICAL_SEAM_COLOR).all()
    # Horizontal seams removed the first two rows of the already narrowed image.
    assert (overlay[0, 2:] == HORIZONTAL_SEAM_COLOR).all()
    assert (overlay[1, 2:] == HORIZONTAL_SEAM_COLOR).all()
    np.testing.assert_array_equal(overlay[2:, 2:], rgb_image[2:, 2:])


@pytest.mark.parametrize("orientation", [VERTICAL, HORIZONTAL])
def test_highlight_seam(rgb_image, orientation):
    n = rgb_image.shape[0] if orientation == VERTICAL else rgb_image.shape[1]
    seam = np.full(n, 3)
    out = highlight_seam(rgb_image, seam, orientation)
    line = out[:, 3] if orientation == VERTICAL else out[3, :]
    assert (line == CURRENT_SEAM_COLOR).all()
    assert out is not rgb_image


def test_pad_to_shape():
    out = pad_to_shape(np.full((2, 3, 3), 200, dtype=np.uint8), 4, 5, fill=(1, 2, 3))
    assert out.shape == (4, 5, 3)
    assert (out[:2, :3] == 200).all() and (out[3, 4] == (1, 2, 3)).all()


def test_montage_size():
    a, b = np.zeros((10, 20, 3), np.uint8), np.zeros((5, 8, 3), np.uint8)
    sheet = montage([[("a", a), ("b", b)], [("c", b)]], padding=2)
    assert sheet.size == (2 + 20 + 2 + 8 + 2, 2 + (10 + 18) + 2 + (5 + 18) + 2)
    zoomed = montage([[("a", a)]], padding=0, zoom=3)
    assert zoomed.size == (60, 30 + 18)


def test_montage_columns_fit_long_labels():
    tiny = np.zeros((4, 4, 3), np.uint8)
    sheet = montage([[("a much longer label than the image", tiny)]], padding=0)
    assert sheet.width > 4
