from __future__ import annotations

import numpy as np
import pytest

from seam_carving.energy import (
    ENERGY_OPTIONS,
    correlate3x3,
    get_energy,
    gradient_energy,
    sobel_energy,
    to_grayscale,
)

ENERGY_FNS = pytest.mark.parametrize(
    "energy_fn", [o.fn for o in ENERGY_OPTIONS], ids=lambda f: f.__name__
)


def test_grayscale_uses_luma_weights():
    img = np.zeros((1, 3, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    img[0, 1] = (0, 255, 0)
    img[0, 2] = (0, 0, 255)
    np.testing.assert_allclose(to_grayscale(img)[0], [0.299 * 255, 0.587 * 255, 0.114 * 255])


def test_grayscale_accepts_grey_images():
    gray = np.arange(6, dtype=np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(to_grayscale(gray), gray.astype(float))
    np.testing.assert_array_equal(to_grayscale(gray[..., None]), gray.astype(float))


def test_grayscale_rejects_bad_shapes():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 2, 2)))


@ENERGY_FNS
def test_shape_dtype_and_non_negative(energy_fn, rgb_image):
    energy = energy_fn(rgb_image)
    assert energy.shape == rgb_image.shape[:2]
    assert energy.dtype == np.float64
    assert (energy >= 0).all()


@ENERGY_FNS
def test_constant_image_has_zero_energy(energy_fn):
    img = np.full((8, 9, 3), 123, dtype=np.uint8)
    np.testing.assert_array_equal(energy_fn(img), 0.0)


@ENERGY_FNS
def test_energy_is_high_only_near_an_edge(energy_fn):
    img = np.zeros((6, 10, 3), dtype=np.uint8)
    img[:, 5:] = 255  # vertical edge between columns 4 and 5
    energy = energy_fn(img)
    assert (energy[:, 4:6] > 0).all()
    np.testing.assert_array_equal(energy[:, :3], 0.0)
    np.testing.assert_array_equal(energy[:, 7:], 0.0)


@ENERGY_FNS
@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_degenerate_sizes(energy_fn, shape):
    energy = energy_fn(np.ones(shape + (3,), dtype=np.uint8))
    assert energy.shape == shape


@ENERGY_FNS
def test_energy_commutes_with_transpose(energy_fn, rgb_image):
    """Horizontal seams use the transposed energy map, so the energy must not prefer a direction."""
    np.testing.assert_allclose(energy_fn(rgb_image.swapaxes(0, 1)), energy_fn(rgb_image).T)


def test_gradient_energy_matches_hand_computation():
    gray = np.array([[0, 10, 30], [0, 10, 30], [0, 10, 30]], dtype=float)
    # Central differences with edge replication: row values are [10, 30, 20], dy = 0.
    np.testing.assert_allclose(gradient_energy(gray), [[10, 30, 20]] * 3)


def test_correlate3x3_identity_kernel():
    gray = np.arange(12, dtype=float).reshape(3, 4)
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1
    np.testing.assert_array_equal(correlate3x3(gray, kernel), gray)


def test_correlate3x3_rejects_bad_kernel():
    with pytest.raises(ValueError):
        correlate3x3(np.zeros((3, 3)), np.ones((2, 2)))


def test_sobel_matches_hand_computation():
    gray = np.zeros((3, 3))
    gray[:, 2] = 1.0  # step edge on the right
    # Centre pixel: Gx = (1 + 2 + 1) = 4, Gy = 0
    assert sobel_energy(gray)[1, 1] == pytest.approx(4.0)


def test_get_energy_by_key_and_label():
    for option in ENERGY_OPTIONS:
        assert get_energy(option.key) is option
        assert get_energy(option.label.upper()) is option
    with pytest.raises(KeyError):
        get_energy("does-not-exist")
