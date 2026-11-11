from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def rgb_image(rng: np.random.Generator) -> np.ndarray:
    """A small random RGB image (H=12, W=16)."""
    return rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
