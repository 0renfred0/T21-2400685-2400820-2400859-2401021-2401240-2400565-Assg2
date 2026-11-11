"""Tests for the command-line tools, run with a stand-in algorithm so they do not need the stubs."""

from __future__ import annotations

import numpy as np
import pytest

from seam_carving import benchmark, cli
from seam_carving.algorithms import Algorithm
from seam_carving.energy import DEFAULT_ENERGY
from seam_carving.image_io import load_image, save_image
from tests.helpers import leftmost_seam


def _stub(energy):
    raise NotImplementedError


FAKE = Algorithm("dp", "Fake DP", leftmost_seam, "tests/helpers.py", "stand-in")
STUB = Algorithm("todo", "Todo", _stub, "nowhere.py", "stub")


@pytest.fixture
def image_path(tmp_path, rgb_image):
    return save_image(rgb_image, tmp_path / "in.png")


@pytest.mark.parametrize(
    "value, current, expected",
    [(None, 40, 40), ("30", 40, 30), ("50%", 40, 20), ("1%", 40, 1), ("100%", 7, 7)],
)
def test_parse_size(value, current, expected):
    assert cli.parse_size(value, current) == expected


@pytest.mark.parametrize("value", ["0%", "150%", "abc"])
def test_parse_size_invalid(value):
    with pytest.raises(ValueError):
        cli.parse_size(value, 40)


def test_cli_writes_all_outputs(monkeypatch, tmp_path, image_path, capsys):
    monkeypatch.setattr(cli, "get_algorithm", lambda _name: FAKE)
    out = tmp_path / "out.png"
    code = cli.main(
        [
            str(image_path),
            "-W", "75%",
            "-H", "10",
            "-o", str(out),
            "--seams", str(tmp_path / "seams.png"),
            "--energy-map", str(tmp_path / "energy.png"),
            "--gif", str(tmp_path / "steps.gif"),
        ]
    )  # fmt: skip
    assert code == 0
    assert load_image(out).shape == (10, 12, 3)
    for name in ("seams.png", "energy.png", "steps.gif"):
        assert (tmp_path / name).exists()
    assert "Fake DP" in capsys.readouterr().out


def test_cli_reports_not_implemented(monkeypatch, image_path, capsys):
    monkeypatch.setattr(cli, "get_algorithm", lambda _name: STUB)
    assert cli.main([str(image_path), "-W", "10"]) == cli.EXIT_NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err


def test_cli_rejects_enlarging(monkeypatch, image_path):
    monkeypatch.setattr(cli, "get_algorithm", lambda _name: FAKE)
    assert cli.main([str(image_path), "-W", "100"]) == 1


def test_benchmark_compare(tmp_path, rgb_image):
    rows = benchmark.compare(rgb_image, 12, 12, [FAKE], DEFAULT_ENERGY, out_dir=tmp_path, name="t")
    assert len(rows) == 1 and rows[0]["seams"] == 4
    assert rows[0]["first_cost_vs_dp"] == "1.000"  # FAKE has key "dp", so it is its own reference
    for name in ("t_dp_result.png", "t_dp_seams.png", "t_comparison.png", "t_summary.csv"):
        assert (tmp_path / name).exists()


def test_benchmark_scaling_is_linear_for_linear_algorithm():
    rows, exponents = benchmark.scaling([FAKE], [20, 40], repeat=1)
    assert [r["size"] for r in rows] == ["20x20", "40x40"]
    assert "dp" in exponents


def test_benchmark_main_skips_stubs(monkeypatch, capsys, image_path):
    monkeypatch.setattr(benchmark, "_resolve_algorithms", lambda _keys: [STUB])
    assert benchmark.main(["compare", str(image_path)]) == 0
    out = capsys.readouterr().out
    assert "skipping Todo" in out and "No implemented algorithms" in out


def test_benchmark_main_runs(monkeypatch, capsys, image_path, tmp_path):
    monkeypatch.setattr(benchmark, "_resolve_algorithms", lambda _keys: [FAKE, STUB])
    code = benchmark.main(["compare", str(image_path), "-W", "90%", "--out-dir", str(tmp_path)])
    assert code == 0
    assert "Fake DP" in capsys.readouterr().out
    assert np.any(load_image(tmp_path / "in_comparison.png"))
