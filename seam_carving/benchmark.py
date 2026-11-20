"""Benchmarks that produce the comparison results for the report (items 3, 4 and 5).

``compare``: carve one image with every implemented algorithm and report running time, total
removed energy, and how each algorithm's first seam compares with Dynamic Programming's on
the same energy map. Saves the outputs and a side-by-side montage::

    python -m seam_carving.benchmark compare samples/castle.png --width 75%

``scaling``: time a single seam on random energy maps of growing size and fit the empirical
exponent of time vs. pixel count (about 1.0 means linear, i.e. O(H*W))::

    python -m seam_carving.benchmark scaling --sizes 16 32 64

Algorithms that are still stubs are skipped and listed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .algorithms import ALGORITHMS, Algorithm, get_algorithm, is_implemented
from .carver import SEAM_ORDERS, carve, plan_orientations
from .cli import parse_size
from .energy import DEFAULT_ENERGY, ENERGY_OPTIONS, EnergyOption, get_energy
from .image_io import load_image, save_image
from .seams import HORIZONTAL, seam_cost, validate_vertical_seam
from .visualization import draw_seams, energy_to_image, montage


def _split_implemented(algorithms: Sequence[Algorithm]) -> tuple[list[Algorithm], list[Algorithm]]:
    # Split the list into the algorithms we can run and the ones that are still stubs, so the
    # benchmark can skip the unfinished ones instead of crashing.
    ready = [a for a in algorithms if is_implemented(a)]
    pending = [a for a in algorithms if a not in ready]
    return ready, pending


def _print_table(rows: list[dict], columns: list[str]) -> None:
    # Print a Markdown table. Each column is as wide as its longest entry (or its heading), so
    # everything lines up and the output can be pasted straight into the report.
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in columns) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) + " |")


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    algorithms: Sequence[Algorithm],
    energy: EnergyOption,
    *,
    order: str = "width-first",
    repeat: int = 1,
    out_dir: Path | None = None,
    name: str = "image",
) -> list[dict]:
    """Carve ``image`` with each algorithm. Returns one summary row per algorithm."""
    energy_map = energy.fn(image)
    h, w = image.shape[:2]

    # Compare the *first* seam each algorithm picks on exactly the same energy map. If the first
    # step is a horizontal seam, the finders work on the transposed map, just like in the carver.
    plan = plan_orientations(w - target_width, h - target_height, order)
    first_energy = energy_map.T if plan and plan[0] == HORIZONTAL else energy_map
    first_energy.flags.writeable = False  # the finders promise not to modify their input
    first_seams = {
        a.key: validate_vertical_seam(
            a.find_seam(first_energy), *first_energy.shape, source=a.label
        )
        for a in algorithms
    }
    dp_first = first_seams.get("dp")  # DP is optimal, so it's the yardstick for the others

    # `rows` collects one summary line per algorithm. `grid` is the picture layout for the
    # montage: the first row shows the original and its energy map.
    rows, grid = (
        [],
        [[("Original", image), (f"Energy ({energy.label})", energy_to_image(energy_map))]],
    )
    for algorithm in algorithms:
        total_times, find_times, per_seam_times = [], [], []
        result = None

        # Run the whole carve several times and report the median, so one slow run caused by
        # something else on the computer doesn't skew the result.
        for _ in range(max(1, repeat)):
            result = carve(
                image, target_width, target_height, algorithm.find_seam, energy.fn, order=order
            )
            total_times.append(result.total_time)
            find_times.append(result.find_time)
            per_seam_times.append(result.ms_per_seam)
        assert result is not None
        first = first_seams[algorithm.key]
        first_cost = seam_cost(first_energy, first)
        row = {
            "algorithm": algorithm.label,
            "seams": len(result.seams),
            "total_s": f"{statistics.median(total_times):.3f}",
            "find_s": f"{statistics.median(find_times):.3f}",
            "ms_per_seam": f"{statistics.median(per_seam_times):.2f}",
            "removed_energy": f"{result.total_cost:.1f}",
            "first_seam_cost": f"{first_cost:.1f}",
        }
        if dp_first is not None:
            # 1.000 means "as cheap as DP's seam"; bigger means worse. The column difference says
            # how far apart the two seams are, on average (0 = the very same pixels).
            dp_cost = seam_cost(first_energy, dp_first)
            row["first_cost_vs_dp"] = f"{first_cost / dp_cost:.3f}" if dp_cost > 0 else "n/a"
            row["mean_col_diff_vs_dp"] = f"{np.mean(np.abs(first - dp_first)):.2f}"
        rows.append(row)

        overlay = draw_seams(image, result.seams)
        grid.append(
            [
                (f"{algorithm.label}: removed seams", overlay),
                (f"{algorithm.label}: result", result.image),
            ]
        )
        if out_dir is not None:
            save_image(result.image, out_dir / f"{name}_{algorithm.key}_result.png")
            save_image(overlay, out_dir / f"{name}_{algorithm.key}_seams.png")

    # Zoom tiny test images up (to about 240 px wide) so the seams are visible in the montage.
    if out_dir is not None and rows:
        zoom = max(1, 240 // image.shape[1])
        montage(grid, zoom=zoom).save(out_dir / f"{name}_comparison.png")
        _write_csv(rows, out_dir / f"{name}_summary.csv")
    return rows


def scaling(
    algorithms: Sequence[Algorithm],
    sizes: Sequence[int],
    *,
    repeat: int = 3,
    seed: int = 0,
) -> tuple[list[dict], dict[str, float]]:
    """Time one seam on random ``n x n`` energy maps.

    Returns the timing rows and, for each algorithm, the fitted exponent ``k`` in
    ``time ~ (H*W)^k``.
    """
    # A fixed seed means every algorithm and every run sees the same random energy maps.
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    samples: dict[str, list[tuple[int, float]]] = {a.key: [] for a in algorithms}
    for n in sizes:
        energy = rng.random((n, n)) * 255.0
        energy.flags.writeable = False
        row: dict = {"size": f"{n}x{n}", "pixels": n * n}
        for algorithm in algorithms:
            # Time just the seam finder (no energy map, no removal), several times, and keep the
            # fastest run: it is the least disturbed by other programs.
            times = []
            for _ in range(max(1, repeat)):
                t = time.perf_counter()
                algorithm.find_seam(energy)
                times.append(time.perf_counter() - t)
            best = min(times)
            row[f"{algorithm.key}_ms"] = f"{best * 1000:.3f}"
            samples[algorithm.key].append((n * n, best))
        rows.append(row)

    # If time ~ pixels^k, then log(time) = k * log(pixels) + c, which is a straight line. So
    # fitting a line through the log-log points gives k as its slope: about 1 means linear,
    # about 2 means quadratic. We need at least two sizes, and a time of zero can't be logged.
    exponents = {}
    for key, pts in samples.items():
        pts = [(p, t) for p, t in pts if t > 0]
        if len(pts) >= 2:
            slope, _ = np.polyfit(np.log([p for p, _ in pts]), np.log([t for _, t in pts]), 1)
            exponents[key] = float(slope)
    return rows, exponents


def _resolve_algorithms(keys: Sequence[str] | None) -> list[Algorithm]:
    return [get_algorithm(k) for k in keys] if keys else list(ALGORITHMS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m seam_carving.benchmark", description="Compare seam-finding algorithms."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    algo_keys = [a.key for a in ALGORITHMS]

    p_cmp = sub.add_parser("compare", help="carve one image with every algorithm")
    p_cmp.add_argument("input", type=Path)
    p_cmp.add_argument(
        "-W", "--width", help="target width (pixels or %%), default 80%%", default="80%"
    )
    p_cmp.add_argument("-H", "--height", help="target height (pixels or %%), default unchanged")
    p_cmp.add_argument(
        "-e", "--energy", default=DEFAULT_ENERGY.key, choices=[e.key for e in ENERGY_OPTIONS]
    )
    p_cmp.add_argument("--order", default="width-first", choices=SEAM_ORDERS)
    p_cmp.add_argument("-a", "--algorithms", nargs="+", choices=algo_keys)
    p_cmp.add_argument("--repeat", type=int, default=1, help="runs per algorithm (median time)")
    p_cmp.add_argument("--out-dir", type=Path, default=Path("results"))

    p_sc = sub.add_parser("scaling", help="time one seam on growing random energy maps")
    p_sc.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    p_sc.add_argument("-a", "--algorithms", nargs="+", choices=algo_keys)
    p_sc.add_argument("--repeat", type=int, default=3, help="runs per size (best time)")
    p_sc.add_argument("--seed", type=int, default=0)
    p_sc.add_argument("--csv", type=Path, help="also write the table to a CSV file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Benchmark only what's finished and say so about the rest, so a half-done team project
    # still gives useful numbers.
    ready, pending = _split_implemented(_resolve_algorithms(args.algorithms))
    for algorithm in pending:
        print(f"skipping {algorithm.label}: not implemented yet ({algorithm.source})")
    if not ready:
        print("No implemented algorithms to benchmark yet.")
        return 0

    if args.command == "compare":
        image = load_image(args.input)
        h, w = image.shape[:2]
        tw, th = parse_size(args.width, w), parse_size(args.height, h)
        print(f"\n{args.input.name}: {w}x{h} -> {tw}x{th}\n")
        rows = compare(
            image,
            tw,
            th,
            ready,
            get_energy(args.energy),
            order=args.order,
            repeat=args.repeat,
            out_dir=args.out_dir,
            name=args.input.stem,
        )
        _print_table(rows, list(rows[0].keys()))
        print(f"\nOutputs saved in {args.out_dir}/")
    else:
        rows, exponents = scaling(ready, args.sizes, repeat=args.repeat, seed=args.seed)
        print()
        _print_table(rows, list(rows[0].keys()))
        print()
        for key, k in exponents.items():
            print(f"{get_algorithm(key).label}: time ~ (H*W)^{k:.2f}")
        if args.csv:
            _write_csv(rows, args.csv)
            print(f"Saved {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
