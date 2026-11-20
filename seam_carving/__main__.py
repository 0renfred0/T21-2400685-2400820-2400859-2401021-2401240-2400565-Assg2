"""``python -m seam_carving [image]`` starts the GUI."""

from __future__ import annotations

import sys


def run() -> None:
    # The GUI is imported here, not at the top of the file, so a missing Tkinter gives a friendly
    # message instead of a long traceback. The command-line tool doesn't need Tkinter at all.
    try:
        from .gui import main
    except ModuleNotFoundError as exc:
        if exc.name in ("tkinter", "_tkinter"):
            sys.exit(
                "Tkinter is not available in this Python installation.\n"
                "  Windows/macOS: reinstall Python from python.org (Tcl/Tk is included).\n"
                "  Debian/Ubuntu: sudo apt install python3-tk\n"
                "You can still use the command line: python -m seam_carving.cli --help"
            )
        # Some other module is missing, which is a real bug, so don't hide it.
        raise
    main()


if __name__ == "__main__":
    run()
