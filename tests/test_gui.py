"""End-to-end check of the GUI: open an image, carve it, check the result.

Skipped when Tkinter or a display is not available, or when ``SKIP_GUI_TESTS=1`` is set.
CI runs it on Linux under ``xvfb-run``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

if os.environ.get("SKIP_GUI_TESTS") == "1":
    pytest.skip("SKIP_GUI_TESTS=1", allow_module_level=True)
tk = pytest.importorskip("tkinter")

from seam_carving import gui  # noqa: E402
from seam_carving.algorithms import Algorithm  # noqa: E402
from seam_carving.image_io import save_image  # noqa: E402
from tests.helpers import leftmost_seam  # noqa: E402

FAKE = Algorithm("fake", "Dynamic Programming", leftmost_seam, "tests/helpers.py", "stand-in")


@pytest.fixture
def app(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display available: {exc}")
    root.withdraw()
    monkeypatch.setattr(gui, "get_algorithm", lambda _name: FAKE)
    monkeypatch.setattr(gui, "is_implemented", lambda _alg: True)
    application = gui.SeamCarvingApp(root)
    yield application
    application.cancel_event.set()
    root.destroy()


def _wait_until(app, condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        app.root.update()
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the GUI")
        time.sleep(0.01)


def test_open_and_carve(app, tmp_path, rgb_image):
    path = save_image(rgb_image, tmp_path / "img.png")
    app.open_image(path, ask_downscale=False)
    app.root.update()
    assert app.original.shape == rgb_image.shape
    assert app.width_var.get() == 12  # 75% of 16 by default

    app.width_var.set(10)
    app.height_var.set(9)
    app.animate_var.set(True)
    app._on_animation_changed()
    app._delay_s = 0.0
    app.start_carving()
    _wait_until(app, lambda: app.result is not None)

    np.testing.assert_array_equal(app.result.image, rgb_image[3:, 6:])
    assert app.view_var.get() == gui.VIEW_SEAMS
    assert "Dynamic Programming" in app.stats_var.get()

    for view in (gui.VIEW_ORIGINAL, gui.VIEW_ENERGY, gui.VIEW_SEAMS):
        app.view_var.set(view)
        app.render()

    app.use_result_as_input()
    assert app.original.shape == (9, 10, 3) and app.result is None


def test_stop_cancels_a_run(app, tmp_path, rgb_image):
    app.open_image(save_image(rgb_image, tmp_path / "img.png"), ask_downscale=False)
    app.width_var.set(1)
    app._delay_s = 0.2  # slow animation, so there is time to press Stop
    app.start_carving()
    _wait_until(app, lambda: app.preview is not None)
    app.stop_carving()
    _wait_until(app, lambda: not app._is_running() and app.messages.empty())
    app.root.update()
    assert app.result is None
    assert "stopped" in app.status_var.get().lower()
