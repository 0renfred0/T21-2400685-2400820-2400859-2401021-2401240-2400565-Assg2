"""Tkinter desktop app for the seam-carving demo.

Run with ``python main.py [image]`` or ``python -m seam_carving [image]``.

Layout
------
* Left panel: algorithm and energy dropdowns, target size sliders, animation options.
* Centre: the input image. Switch between Original, Energy map and Removed seams.
* Right: the carved result, animated seam by seam while carving.

Carving runs in a background thread so the window stays responsive. Tk widgets may only be
used from the main thread, so the worker sends progress messages through a
:class:`queue.Queue`, and the main loop checks that queue every few milliseconds
(:meth:`SeamCarvingApp._poll_worker`).
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .algorithms import ALGORITHMS, Algorithm, get_algorithm, is_implemented
from .carver import CarveResult, CarveStep, CarvingCancelled, SeamRecord, carve
from .energy import DEFAULT_ENERGY, ENERGY_OPTIONS, EnergyOption, get_energy
from .image_io import SUPPORTED_EXTENSIONS, downscale_to_fit, load_image, save_image
from .seams import VERTICAL, InvalidSeamError
from .visualization import (
    HORIZONTAL_SEAM_COLOR,
    VERTICAL_SEAM_COLOR,
    draw_seams,
    energy_to_image,
    ensure_rgb,
    highlight_seam,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"

VIEW_ORIGINAL = "Original"
VIEW_ENERGY = "Energy map"
VIEW_SEAMS = "Removed seams"

ORDER_CHOICES = {
    "Width first (vertical seams first)": "width-first",
    "Height first (horizontal seams first)": "height-first",
    "Interleaved": "interleaved",
}

LARGE_IMAGE_SIDE = 1000  # offer to downscale images larger than this ...
DOWNSCALE_SIDE = 800  # ... to this size
MAX_UPSCALE = 4  # small images are enlarged (by whole numbers) so the seams stay visible
POLL_MS = 30
CANVAS_BG = "#2b2d31"
CANVAS_PAD = 8
FILETYPES = [
    ("Images", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)),
    ("All files", "*.*"),
]


class SeamCarvingApp:
    """Main window. Create it with a Tk root: ``SeamCarvingApp(tk.Tk())``."""

    def __init__(self, root: tk.Tk, initial_image: Path | None = None) -> None:
        self.root = root
        self.root.title("Seam Carving: DP vs Greedy vs Graph Cut (Team T21)")
        self.root.geometry("1280x760")
        self.root.minsize(980, 600)

        # Image and run state
        self.image_name = "image"
        self.original: np.ndarray | None = None
        self.result: CarveResult | None = None
        self.preview: np.ndarray | None = None  # current frame of the animation
        self.live_overlay: np.ndarray | None = None  # original + seams removed so far
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.messages: queue.Queue = queue.Queue()
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._energy_cache: tuple[tuple[int, str], np.ndarray] | None = None
        self._render_job: str | None = None
        # The worker thread reads these plain attributes. Tk variables are not thread-safe.
        self._animate = True
        self._delay_s = 0.02

        # Tk variables
        self.algorithm_var = tk.StringVar(value=ALGORITHMS[0].label)
        self.energy_var = tk.StringVar(value=DEFAULT_ENERGY.label)
        self.order_var = tk.StringVar(value=next(iter(ORDER_CHOICES)))
        self.width_var = tk.IntVar(value=1)
        self.height_var = tk.IntVar(value=1)
        self.animate_var = tk.BooleanVar(value=True)
        self.delay_var = tk.IntVar(value=20)
        self.view_var = tk.StringVar(value=VIEW_ORIGINAL)
        self.status_var = tk.StringVar(value="Open an image to begin (Ctrl+O), or try a sample.")
        self.plan_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="No run yet.")

        self._setup_style()
        self._build_ui()
        self._bind_events()
        self._on_algorithm_changed()
        self._on_energy_changed()
        self._update_controls()
        self.root.after(POLL_MS, self._poll_worker)
        if initial_image is not None:
            self.root.after(50, lambda: self.open_image(initial_image))

    # ------------------------------------------------------------------ UI construction
    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        if self.root.tk.call("tk", "windowingsystem") == "x11" and "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Ok.TLabel", foreground="#1e8e3e")
        style.configure("Todo.TLabel", foreground="#d93025")

    def _build_ui(self) -> None:
        # Toolbar
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Open image…", command=self.ask_open_image).pack(side=tk.LEFT)
        self.samples_button = ttk.Menubutton(toolbar, text="Open sample")
        self.samples_menu = tk.Menu(self.samples_button, tearoff=False)
        self.samples_button["menu"] = self.samples_menu
        self.samples_button.pack(side=tk.LEFT, padx=(6, 0))
        self._populate_samples_menu()
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.save_button = ttk.Button(toolbar, text="Save result…", command=self.save_result)
        self.save_button.pack(side=tk.LEFT)
        self.save_seams_button = ttk.Button(
            toolbar, text="Save seam overlay…", command=self.save_seam_overlay
        )
        self.save_seams_button.pack(side=tk.LEFT, padx=(6, 0))
        self.use_result_button = ttk.Button(
            toolbar, text="Use result as input", command=self.use_result_as_input
        )
        self.use_result_button.pack(side=tk.LEFT, padx=(6, 0))

        # Status bar (packed before the body so it stays visible when the window shrinks)
        status = ttk.Frame(self.root, padding=(8, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self.progress = ttk.Progressbar(status, length=260, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, padding=(8, 0, 8, 0))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._build_controls(body)

        panes = ttk.Frame(body)
        panes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        panes.columnconfigure(0, weight=1, uniform="pane")
        panes.columnconfigure(1, weight=1, uniform="pane")
        panes.rowconfigure(0, weight=1)

        self.input_frame = ttk.LabelFrame(panes, text="Input", padding=4)
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        views = ttk.Frame(self.input_frame)
        views.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(views, text="Show:").pack(side=tk.LEFT)
        for view in (VIEW_ORIGINAL, VIEW_ENERGY, VIEW_SEAMS):
            ttk.Radiobutton(
                views, text=view, value=view, variable=self.view_var, command=self.schedule_render
            ).pack(side=tk.LEFT, padx=(6, 0))
        self.input_canvas = tk.Canvas(self.input_frame, bg=CANVAS_BG, highlightthickness=0)
        self.input_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))

        self.output_frame = ttk.LabelFrame(panes, text="Result", padding=4)
        self.output_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        legend = ttk.Frame(self.output_frame)
        legend.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(
            legend,
            text="Same scale as the input, so you can see how much it shrank.",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT)
        # Statistics of the last run, under the result (packed first so it always has room).
        stats = ttk.LabelFrame(self.output_frame, text="Last run", padding=(8, 4))
        stats.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Label(stats, textvariable=self.stats_var, font="TkFixedFont", justify=tk.LEFT).pack(
            anchor="w"
        )
        self.output_canvas = tk.Canvas(self.output_frame, bg=CANVAS_BG, highlightthickness=0)
        self.output_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))

    def _build_controls(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, width=300)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        box = ttk.LabelFrame(panel, text="1. Seam-finding algorithm", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.algorithm_combo = ttk.Combobox(
            box,
            textvariable=self.algorithm_var,
            state="readonly",
            width=32,
            values=[a.label for a in ALGORITHMS],
        )
        self.algorithm_combo.pack(fill=tk.X)
        self.algorithm_status = ttk.Label(box, text="")
        self.algorithm_status.pack(anchor="w", pady=(4, 0))
        self.algorithm_hint = ttk.Label(box, text="", style="Hint.TLabel", wraplength=270)
        self.algorithm_hint.pack(anchor="w")

        box = ttk.LabelFrame(panel, text="2. Energy function", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.energy_combo = ttk.Combobox(
            box,
            textvariable=self.energy_var,
            state="readonly",
            width=32,
            values=[e.label for e in ENERGY_OPTIONS],
        )
        self.energy_combo.pack(fill=tk.X)
        self.energy_hint = ttk.Label(box, text="", style="Hint.TLabel", wraplength=270)
        self.energy_hint.pack(anchor="w", pady=(4, 0))

        box = ttk.LabelFrame(panel, text="3. Target size", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        box.columnconfigure(1, weight=1)
        self.width_scale, self.width_spin = self._size_row(box, 0, "Width", self.width_var)
        self.height_scale, self.height_spin = self._size_row(box, 1, "Height", self.height_var)
        ttk.Label(box, textvariable=self.plan_var, style="Hint.TLabel", wraplength=270).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        ttk.Label(box, text="Seam order").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.order_combo = ttk.Combobox(
            box, textvariable=self.order_var, state="readonly", values=list(ORDER_CHOICES)
        )
        self.order_combo.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        box = ttk.LabelFrame(panel, text="4. Visualisation", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            box,
            text="Animate seam removal step by step",
            variable=self.animate_var,
            command=self._on_animation_changed,
        ).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Delay per seam").pack(side=tk.LEFT)
        self.delay_label = ttk.Label(row, text="20 ms", width=7)
        self.delay_label.pack(side=tk.RIGHT)
        ttk.Scale(
            row,
            from_=0,
            to=300,
            orient=tk.HORIZONTAL,
            command=self._on_delay_moved,
            value=self.delay_var.get(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        buttons = ttk.Frame(panel)
        buttons.pack(fill=tk.X, pady=(0, 8))
        self.carve_button = ttk.Button(
            buttons, text="Carve ▶", style="Accent.TButton", command=self.start_carving
        )
        self.carve_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop_carving)
        self.stop_button.pack(side=tk.LEFT, padx=(6, 0))
        self.reset_button = ttk.Button(buttons, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=(6, 0))

    def _size_row(
        self, parent: ttk.Frame, row: int, label: str, var: tk.IntVar
    ) -> tuple[ttk.Scale, ttk.Spinbox]:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(
            parent,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            command=lambda value, v=var: self._on_scale_moved(v, value),
        )
        scale.grid(row=row, column=1, sticky="ew", padx=6)
        spin = ttk.Spinbox(parent, from_=1, to=1, textvariable=var, width=6)
        spin.grid(row=row, column=2)
        spin.bind("<Return>", lambda _e: self._clamp_sizes())
        spin.bind("<FocusOut>", lambda _e: self._clamp_sizes())
        var.trace_add("write", lambda *_: self._on_size_var_changed())
        return scale, spin

    def _populate_samples_menu(self) -> None:
        self.samples_menu.delete(0, tk.END)
        samples = (
            sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if SAMPLES_DIR.is_dir()
            else []
        )
        if not samples:
            self.samples_menu.add_command(label="(no images in samples/)", state=tk.DISABLED)
        for path in samples:
            self.samples_menu.add_command(
                label=path.name, command=lambda p=path: self.open_image(p, ask_downscale=False)
            )

    def _bind_events(self) -> None:
        mod = "Command" if self.root.tk.call("tk", "windowingsystem") == "aqua" else "Control"
        self.root.bind_all(f"<{mod}-o>", lambda _e: self.ask_open_image())
        self.root.bind_all(f"<{mod}-s>", lambda _e: self.save_result())
        self.root.bind_all(f"<{mod}-Return>", lambda _e: self.start_carving())
        self.root.bind_all("<F5>", lambda _e: self.start_carving())
        self.root.bind_all("<Escape>", lambda _e: self.stop_carving())
        self.algorithm_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_algorithm_changed())
        self.energy_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_energy_changed())
        for canvas in (self.input_canvas, self.output_canvas):
            canvas.bind("<Configure>", lambda _e: self.schedule_render())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------ control callbacks
    def _selected_algorithm(self) -> Algorithm:
        return get_algorithm(self.algorithm_var.get())

    def _selected_energy(self) -> EnergyOption:
        return get_energy(self.energy_var.get())

    def _on_algorithm_changed(self) -> None:
        algorithm = self._selected_algorithm()
        if is_implemented(algorithm):
            self.algorithm_status.configure(text="● Ready", style="Ok.TLabel")
        else:
            self.algorithm_status.configure(
                text=f"● Not implemented yet: {Path(algorithm.source).name}", style="Todo.TLabel"
            )
        self.algorithm_hint.configure(text=algorithm.description)

    def _on_energy_changed(self) -> None:
        self.energy_hint.configure(text=self._selected_energy().description)
        if self.view_var.get() == VIEW_ENERGY:
            self.schedule_render()

    def _on_animation_changed(self) -> None:
        self._animate = bool(self.animate_var.get())

    def _on_delay_moved(self, value: str) -> None:
        ms = int(round(float(value)))
        self.delay_var.set(ms)
        self._delay_s = ms / 1000.0
        self.delay_label.configure(text=f"{ms} ms")

    def _on_scale_moved(self, var: tk.IntVar, value: str) -> None:
        new = int(round(float(value)))
        try:
            if var.get() == new:
                return
        except tk.TclError:
            pass
        var.set(new)

    def _read_size(self, var: tk.IntVar, limit: int) -> int:
        try:
            value = int(var.get())
        except (tk.TclError, ValueError):
            value = limit
        return min(max(value, 1), limit)

    def _clamp_sizes(self) -> None:
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        for var, limit in ((self.width_var, w), (self.height_var, h)):
            value = self._read_size(var, limit)
            try:
                if var.get() != value:
                    var.set(value)
            except tk.TclError:
                var.set(value)

    def _on_size_var_changed(self) -> None:
        if self.original is None:
            self.plan_var.set("")
            return
        h, w = self.original.shape[:2]
        for var, scale in (
            (self.width_var, self.width_scale),
            (self.height_var, self.height_scale),
        ):
            try:
                value = int(var.get())
            except (tk.TclError, ValueError):
                continue  # the user is still typing
            if int(round(float(scale.get()))) != value:
                scale.set(value)
        tw, th = self._read_size(self.width_var, w), self._read_size(self.height_var, h)
        self.plan_var.set(
            f"Original {w} × {h} px. Will remove {w - tw} vertical + {h - th} horizontal "
            f"seams ({100 * tw / w:.0f}% × {100 * th / h:.0f}%)."
        )

    def _update_controls(self) -> None:
        running = self.worker is not None and self.worker.is_alive()
        has_image = self.original is not None
        has_result = self.result is not None
        idle = tk.NORMAL if not running else tk.DISABLED

        def state(enabled: bool) -> str:
            return tk.NORMAL if enabled else tk.DISABLED

        self.carve_button.configure(state=state(has_image and not running))
        self.stop_button.configure(state=state(running))
        self.reset_button.configure(state=state(has_image and not running))
        self.save_button.configure(state=state(has_result and not running))
        self.save_seams_button.configure(state=state(has_result and not running))
        self.use_result_button.configure(state=state(has_result and not running))
        self.samples_button.configure(state=idle)
        for combo in (self.algorithm_combo, self.energy_combo, self.order_combo):
            combo.configure(state="readonly" if not running else tk.DISABLED)
        for widget in (self.width_scale, self.height_scale, self.width_spin, self.height_spin):
            widget.configure(state=state(has_image and not running))

    # ------------------------------------------------------------------ file actions
    def ask_open_image(self) -> None:
        if self._is_running():
            return
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=FILETYPES,
            initialdir=SAMPLES_DIR if SAMPLES_DIR.is_dir() else None,
        )
        if path:
            self.open_image(Path(path))

    def open_image(self, path: Path | str, *, ask_downscale: bool = True) -> None:
        path = Path(path)
        try:
            image = load_image(path)
        except Exception as exc:  # unreadable or unsupported file
            messagebox.showerror("Could not open image", f"{path}\n\n{exc}", parent=self.root)
            return
        h, w = image.shape[:2]
        if (
            ask_downscale
            and max(h, w) > LARGE_IMAGE_SIDE
            and messagebox.askyesno(
                "Large image",
                f"This image is {w} × {h} px. Seam carving recomputes everything for every seam, "
                f"so large images are slow.\n\nDownscale it to at most {DOWNSCALE_SIDE} px first?",
                parent=self.root,
            )
        ):
            image = downscale_to_fit(image, DOWNSCALE_SIDE)
        self.image_name = path.stem
        self._set_original(image)
        h, w = image.shape[:2]
        self.status_var.set(f"Loaded {path.name} ({w} × {h} px).")

    def _set_original(self, image: np.ndarray) -> None:
        self.original = image
        self.result = None
        self.preview = None
        self.live_overlay = None
        self._energy_cache = None
        self.view_var.set(VIEW_ORIGINAL)
        h, w = image.shape[:2]
        self.width_scale.configure(to=w)
        self.height_scale.configure(to=h)
        self.width_spin.configure(to=w)
        self.height_spin.configure(to=h)
        self.width_var.set(max(1, round(w * 0.75)))  # sensible default: 75% of the width
        self.height_var.set(h)
        self.progress.configure(value=0)
        self.stats_var.set("No run yet.")
        self._update_controls()
        self.schedule_render()

    def _default_save_name(self, suffix: str) -> str:
        assert self.result is not None
        h, w = self.result.image.shape[:2]
        return f"{self.image_name}_{self._selected_algorithm().key}_{w}x{h}{suffix}.png"

    def _ask_save_path(self, title: str, initialfile: str) -> Path | None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=OUTPUT_DIR,
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
        )
        return Path(path) if path else None

    def save_result(self) -> None:
        if self.result is None or self._is_running():
            return
        path = self._ask_save_path("Save carved image", self._default_save_name(""))
        if path:
            save_image(self.result.image, path)
            self.status_var.set(f"Saved {path}")

    def save_seam_overlay(self) -> None:
        if self.result is None or self.original is None or self._is_running():
            return
        path = self._ask_save_path("Save seam overlay", self._default_save_name("_seams"))
        if path:
            save_image(draw_seams(self.original, self.result.seams), path)
            self.status_var.set(f"Saved {path}")

    def use_result_as_input(self) -> None:
        if self.result is None or self._is_running():
            return
        image = self.result.image
        self._set_original(image)
        h, w = image.shape[:2]
        self.status_var.set(f"Using the previous result ({w} × {h} px) as the new input.")

    def reset(self) -> None:
        if self.original is not None and not self._is_running():
            self._set_original(self.original)
            self.status_var.set("Reset.")

    def close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()

    # ------------------------------------------------------------------ carving
    def _is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start_carving(self) -> None:
        if self.original is None or self._is_running():
            return
        self._clamp_sizes()
        h, w = self.original.shape[:2]
        tw, th = self._read_size(self.width_var, w), self._read_size(self.height_var, h)
        if (tw, th) == (w, h):
            messagebox.showinfo(
                "Nothing to do", "Choose a smaller width or height first.", parent=self.root
            )
            return
        algorithm = self._selected_algorithm()
        if not is_implemented(algorithm):
            self._show_not_implemented(algorithm)
            return

        energy = self._selected_energy()
        order = ORDER_CHOICES[self.order_var.get()]
        self.cancel_event.clear()
        self.result = None
        self.preview = None
        self.live_overlay = ensure_rgb(self.original).copy()
        self._animate = bool(self.animate_var.get())
        self.progress.configure(value=0, maximum=(w - tw) + (h - th))
        self.status_var.set(f"Carving with {algorithm.label}…")
        self.worker = threading.Thread(
            target=self._run_carve,
            args=(self.original, tw, th, algorithm, energy, order),
            daemon=True,
        )
        self.worker.start()
        self._update_controls()

    def stop_carving(self) -> None:
        if self._is_running():
            self.cancel_event.set()
            self.status_var.set("Stopping…")

    def _run_carve(
        self,
        image: np.ndarray,
        tw: int,
        th: int,
        algorithm: Algorithm,
        energy: EnergyOption,
        order: str,
    ) -> None:
        """Runs in the worker thread. It talks to the UI only through ``self.messages``."""

        def on_step(step: CarveStep) -> None:
            if self._animate:
                self.messages.put(("step", step))
                if self._delay_s > 0 and self.cancel_event.wait(self._delay_s):
                    raise CarvingCancelled()
            else:
                self.messages.put(("progress", step.index, step.record))

        try:
            result = carve(
                image,
                tw,
                th,
                algorithm.find_seam,
                energy.fn,
                order=order,
                on_step=on_step,
                cancel_event=self.cancel_event,
            )
        except CarvingCancelled:
            self.messages.put(("cancelled",))
        except NotImplementedError:
            self.messages.put(("not_implemented", algorithm))
        except Exception as exc:
            self.messages.put(("error", exc, traceback.format_exc()))
        else:
            self.messages.put(("done", result, algorithm, energy))

    def _poll_worker(self) -> None:
        """Main thread: handle worker messages, then draw only the newest animation frame."""
        latest: CarveStep | None = None
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]
                if kind == "step":
                    latest = message[1]
                    self._add_live_seam(latest.record)
                elif kind == "progress":
                    self.progress.configure(value=message[1])
                    self._add_live_seam(message[2])
                elif kind == "done":
                    latest = None
                    self._on_done(*message[1:])
                elif kind == "cancelled":
                    latest = None
                    self._on_finished_without_result("Carving stopped.")
                elif kind == "not_implemented":
                    latest = None
                    self._on_finished_without_result("Algorithm not implemented yet.")
                    self._show_not_implemented(message[1])
                elif kind == "error":
                    latest = None
                    self._on_error(message[1], message[2])
        except queue.Empty:
            pass

        if latest is not None:
            self.preview = highlight_seam(latest.image, latest.seam, latest.orientation)
            self.progress.configure(value=latest.index)
            self.status_var.set(
                f"Removing seam {latest.index}/{latest.total} ({latest.orientation}), "
                f"cost {latest.record.cost:.1f}, {latest.elapsed:.1f} s elapsed"
            )
            self.render()
        self.root.after(POLL_MS, self._poll_worker)

    def _add_live_seam(self, record: SeamRecord) -> None:
        if self.live_overlay is not None:
            color = VERTICAL_SEAM_COLOR if record.orientation == VERTICAL else HORIZONTAL_SEAM_COLOR
            self.live_overlay[record.rows, record.cols] = color

    def _on_done(self, result: CarveResult, algorithm: Algorithm, energy: EnergyOption) -> None:
        self.result = result
        self.preview = None
        self.live_overlay = None
        self.progress.configure(value=self.progress["maximum"])
        (h0, w0), (h1, w1) = result.original_shape, result.image.shape[:2]
        self.stats_var.set(
            f"Algorithm  {algorithm.label} ({energy.label})\n"
            f"Size       {w0}×{h0} → {w1}×{h1}\n"
            f"Seams      {result.vertical_seams} vertical + {result.horizontal_seams} horizontal\n"
            f"Time       {result.total_time:.3f} s total\n"
            f"           {result.find_time:.3f} s seam finding ({result.ms_per_seam:.2f} ms/seam)\n"
            f"           {result.energy_time:.3f} s computing energy maps\n"
            f"Removed E  {result.total_cost:,.1f} (total seam energy)"
        )
        self.status_var.set(
            f"Done: {w0}×{h0} → {w1}×{h1} in {result.total_time:.2f} s with {algorithm.label}."
        )
        self.view_var.set(VIEW_SEAMS)
        self._update_controls()
        self.render()

    def _on_finished_without_result(self, message: str) -> None:
        self.preview = None
        self.live_overlay = None
        self.status_var.set(message)
        self._update_controls()
        self.render()

    def _on_error(self, exc: Exception, details: str) -> None:
        print(details, file=sys.stderr)
        self._on_finished_without_result(f"Error: {exc}")
        title = "Invalid seam" if isinstance(exc, InvalidSeamError) else "Carving failed"
        messagebox.showerror(title, f"{type(exc).__name__}: {exc}", parent=self.root)

    def _show_not_implemented(self, algorithm: Algorithm) -> None:
        messagebox.showinfo(
            "Not implemented yet",
            f"{algorithm.label} has not been implemented yet.\n\n"
            f"Complete {algorithm.find_seam.__name__}() in\n{algorithm.source}\n\n"
            f"then check it with:\npytest tests/test_algorithms.py -k {algorithm.key}",
            parent=self.root,
        )

    # ------------------------------------------------------------------ rendering
    def schedule_render(self) -> None:
        """Redraw soon. Many resize events in a row lead to a single redraw."""
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(15, self.render)

    def _input_view(self) -> np.ndarray | None:
        if self.original is None:
            return None
        view = self.view_var.get()
        if view == VIEW_ENERGY:
            energy = self._selected_energy()
            key = (id(self.original), energy.key)
            if self._energy_cache is None or self._energy_cache[0] != key:
                self._energy_cache = (key, energy_to_image(energy.fn(self.original)))
            return self._energy_cache[1]
        if view == VIEW_SEAMS:
            if self.live_overlay is not None:
                return self.live_overlay
            if self.result is not None:
                return draw_seams(self.original, self.result.seams)
        return self.original

    def _display_scale(self) -> float:
        assert self.original is not None
        h, w = self.original.shape[:2]
        cw = min(self.input_canvas.winfo_width(), self.output_canvas.winfo_width()) - 2 * CANVAS_PAD
        ch = (
            min(self.input_canvas.winfo_height(), self.output_canvas.winfo_height())
            - 2 * CANVAS_PAD
        )
        scale = min(max(cw, 1) / w, max(ch, 1) / h)
        if scale >= 1:  # whole-number zoom keeps single-pixel seams crisp
            scale = float(min(MAX_UPSCALE, math.floor(scale)))
        return scale

    def render(self) -> None:
        self._render_job = None
        if self.original is None:
            self._draw(
                self.input_canvas,
                None,
                1.0,
                "input",
                "Open an image to begin\n(Ctrl+O, or use 'Open sample')",
            )
            self._draw(self.output_canvas, None, 1.0, "output", "The carved image will appear here")
            self.input_frame.configure(text="Input")
            self.output_frame.configure(text="Result")
            return
        scale = self._display_scale()
        h, w = self.original.shape[:2]
        self.input_frame.configure(text=f"Input: {w} × {h} px")
        self._draw(self.input_canvas, self._input_view(), scale, "input", "")

        output = (
            self.preview
            if self.preview is not None
            else (self.result.image if self.result is not None else None)
        )
        if output is not None:
            oh, ow = output.shape[:2]
            self.output_frame.configure(text=f"Result: {ow} × {oh} px")
        else:
            self.output_frame.configure(text="Result")
        self._draw(self.output_canvas, output, scale, "output", "Press Carve ▶ to start")

    def _draw(
        self, canvas: tk.Canvas, array: np.ndarray | None, scale: float, key: str, placeholder: str
    ) -> None:
        canvas.delete("all")
        cw, ch = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        if array is None:
            canvas.create_text(cw / 2, ch / 2, text=placeholder, fill="#9aa0a6", justify=tk.CENTER)
            self._photos.pop(key, None)
            return
        h, w = array.shape[:2]
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        pil = Image.fromarray(ensure_rgb(array))
        if size != (w, h):
            resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.BILINEAR
            pil = pil.resize(size, resample)
        photo = ImageTk.PhotoImage(pil)
        self._photos[key] = photo  # keep a reference, otherwise Tk shows a blank image
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)


def _enable_windows_dpi_awareness() -> None:
    """Without this, Tk looks blurry on high-DPI Windows screens."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    """Start the GUI. Optional first argument: an image to open."""
    argv = sys.argv[1:] if argv is None else argv
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    SeamCarvingApp(root, initial_image=Path(argv[0]) if argv else None)
    root.mainloop()


if __name__ == "__main__":
    main()
