"""Command-line interface: carve one image without opening the GUI.

Examples::

    python -m seam_carving.cli samples/castle.png --width 75% -a dp
    python -m seam_carving.cli photo.jpg -W 400 -H 300 -a greedy -e gradient \\
        -o output/photo_small.png --seams output/photo_seams.png --gif output/photo.gif
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .algorithms import ALGORITHMS, get_algorithm
from .carver import SEAM_ORDERS, CarveStep, carve
from .energy import DEFAULT_ENERGY, ENERGY_OPTIONS, get_energy
from .image_io import load_image, save_image
from .seams import InvalidSeamError
from .visualization import draw_seams, energy_to_image, highlight_seam, pad_to_shape

DEFAULT_OUTPUT_DIR = Path("output")
EXIT_NOT_IMPLEMENTED = 2
EXIT_INVALID_SEAM = 3


def parse_size(value: str | None, current: int) -> int:
    """Convert ``"300"`` (pixels), ``"75%"`` (of ``current``) or ``None`` (unchanged) to pixels."""
    if value is None:
        return current
    text = str(value).strip()
    if text.endswith("%"):
        percent = float(text[:-1])
        if not 0 < percent <= 100:
            raise ValueError(f"percentage must be in (0, 100], got {text}")
        return max(1, round(current * percent / 100.0))
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m seam_carving.cli",
        description="Shrink an image with content-aware seam carving.",
    )
    parser.add_argument("input", type=Path, help="input image (png, jpg, bmp, ...)")
    parser.add_argument(
        "-W", "--width", help="target width in pixels or as a percentage, e.g. 300 or 75%%"
    )
    parser.add_argument(
        "-H", "--height", help="target height in pixels or as a percentage, e.g. 200 or 90%%"
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default=ALGORITHMS[0].key,
        choices=[a.key for a in ALGORITHMS],
        help="seam-finding algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--energy",
        default=DEFAULT_ENERGY.key,
        choices=[e.key for e in ENERGY_OPTIONS],
        help="energy function (default: %(default)s)",
    )
    parser.add_argument(
        "--order",
        default="width-first",
        choices=SEAM_ORDERS,
        help="seam order (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output path (default: output/<name>_<algo>_<W>x<H>.png)"
    )
    parser.add_argument(
        "--seams", type=Path, help="also save the original with every removed seam drawn"
    )
    parser.add_argument("--energy-map", type=Path, help="also save the energy map of the input")
    parser.add_argument("--gif", type=Path, help="also save a step-by-step animation as a GIF")
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=60,
        help="about how many frames the GIF has (default: %(default)s)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    algorithm = get_algorithm(args.algorithm)
    energy = get_energy(args.energy)

    image = load_image(args.input)
    height, width = image.shape[:2]
    try:
        target_w = parse_size(args.width, width)
        target_h = parse_size(args.height, height)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frames = []
    total_seams = (width - target_w) + (height - target_h)
    every = max(1, total_seams // max(1, args.gif_frames))

    def on_step(step: CarveStep) -> None:
        if args.gif and (step.index - 1) % every == 0:
            frame = highlight_seam(step.image, step.seam, step.orientation)
            frames.append(pad_to_shape(frame, height, width))

    try:
        result = carve(
            image,
            target_w,
            target_h,
            algorithm.find_seam,
            energy.fn,
            order=args.order,
            on_step=on_step if args.gif else None,
        )
    except NotImplementedError:
        print(
            f"error: {algorithm.label} is not implemented yet. "
            f"Complete {algorithm.find_seam.__name__}() in {algorithm.source}",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED
    except (InvalidSeamError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_SEAM if isinstance(exc, InvalidSeamError) else 1

    out_h, out_w = result.image.shape[:2]
    output = (
        args.output or DEFAULT_OUTPUT_DIR / f"{args.input.stem}_{algorithm.key}_{out_w}x{out_h}.png"
    )
    save_image(result.image, output)
    saved = [output]
    if args.seams:
        saved.append(save_image(draw_seams(image, result.seams), args.seams))
    if args.energy_map:
        saved.append(save_image(energy_to_image(energy.fn(image)), args.energy_map))
    if args.gif:
        from PIL import Image

        frames.append(pad_to_shape(result.image, height, width))
        pil_frames = [Image.fromarray(f) for f in frames]
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            args.gif, save_all=True, append_images=pil_frames[1:], duration=80, loop=0
        )
        saved.append(args.gif)

    if not args.quiet:
        print(f"Algorithm      : {algorithm.label}")
        print(f"Energy         : {energy.label}")
        print(f"Size           : {width}x{height} -> {out_w}x{out_h}")
        seams = f"{result.vertical_seams} vertical, {result.horizontal_seams} horizontal"
        print(f"Seams removed  : {seams}")
        print(f"Total time     : {result.total_time:.3f} s")
        print(f"  seam finding : {result.find_time:.3f} s ({result.ms_per_seam:.2f} ms/seam)")
        print(f"  energy maps  : {result.energy_time:.3f} s")
        print(f"Removed energy : {result.total_cost:.1f}")
        for path in saved:
            print(f"Saved          : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
