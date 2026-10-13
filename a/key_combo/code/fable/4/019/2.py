from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

try:
    import sys

    _SOLVER_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _spec = importlib.util.spec_from_file_location("arc_utils", str(_SOLVER_UTILS_PATH))
    if _spec is not None and _spec.loader is not None:
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)  # type: ignore[union-attr]
        Color = _module.Color
    else:
        raise RuntimeError("arc_utils import path resolution failed")
except Exception:  # pragma: no cover - fallback if local utils path changes
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

        ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def _bg_color(grid: np.ndarray) -> int:
    """Return most-common color id."""
    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(counts)])


def _ring_mask(size: int, ring: int) -> np.ndarray:
    center = size // 2
    coords = np.indices((size, size))
    dist = np.maximum(np.abs(coords[0] - center), np.abs(coords[1] - center))
    return dist == ring


def _paint_ring(grid: np.ndarray, ring: int, color: int) -> None:
    center = grid.shape[0] // 2
    lo = center - ring
    hi = center + ring + 1
    grid[lo, lo:hi] = color
    grid[hi - 1, lo:hi] = color
    grid[lo:hi, lo] = color
    grid[lo:hi, hi - 1] = color


def _extract_rings(cell: np.ndarray, bg: int) -> list[int]:
    n, m = map(int, cell.shape)
    assert n == m, "cells must be square"
    assert n % 2 == 1, "cells must have odd size"
    max_ring = n // 2
    rings: list[int] = []
    seen_bg_only = False
    for ring in range(max_ring + 1):
        mask = _ring_mask(n, ring)
        vals = cell[mask]
        colored = vals[vals != bg]
        if colored.size == 0:
            if not seen_bg_only:
                seen_bg_only = True
                continue
            break
        unique = np.unique(colored)
        if unique.size != 1:
            raise AssertionError("frame ring must be monochromatic")
        rings.append(int(unique[0]))
        seen_bg_only = False
    return rings


def _infer_cycle_period(values: list[int]) -> int:
    n = len(values)
    assert n > 0, "must have at least one observed frame value"
    max_period = min(9, n)
    for period in range(1, max_period + 1):
        for i in range(n - period):
            if values[i] != values[i + period]:
                break
        else:
            return period
    raise AssertionError("failed to infer cycle period from frame colors")


def _build_frame(rings: list[int], bg: int) -> np.ndarray:
    n = len(rings)
    size = 2 * n - 1
    center = n - 1
    out = np.full((size, size), bg, dtype=int)
    for ring, color in enumerate(rings):
        lo = center - ring
        hi = center + ring + 1
        out[lo, lo:hi] = color
        out[hi - 1, lo:hi] = color
        out[lo:hi, lo] = color
        out[lo:hi, hi - 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the missing final frame of a ring-growth time series from concentric patterns.

    Input structure:
    - A single horizontal lattice of `N` identical odd square cells.
    - The first `N-1` cells are consecutive frames of concentric hollow squares around the cell center, with frame `t` containing exactly `t` rings.
    - The final cell is completely empty (all background); at least 3 prior frames are visible.
    - Ring colors come from a periodic palette and shift together by one-step each frame.

    Rule:
    - Split the row into cells and recover each frame's inner-to-outer ring color sequence.
    - Check that frame counts grow as 1,2,...,N-1 and build the underlying circular color stream from adjacent-frame overlap.
    - Infer the palette period from the stream and extrapolate the next outer ring color.
    - Draw frame `N` with `N` concentric hollow rings on a background field.

    Key insight:
    - The two channels are independent: ring count grows by one per frame while ring colors form one global color stream observed as a sliding window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    assert H <= 30 and W <= 30, "input exceeds ARC bounds"

    bg = _bg_color(g)
    assert H > 0 and W > 0 and H <= 30 and W <= 30
    assert H <= W and W % H == 0, "cells must be a full-width square-row tiling"
    cell_size = H
    cell_count = W // H
    assert cell_count >= 4, "must contain at least three source frames plus one blank cell"

    cells = [g[:, c0 : c0 + cell_size] for c0 in range(0, W, cell_size)]
    rings_per_cell = [_extract_rings(cell, bg) for cell in cells]

    blank_cells = [i for i, rings in enumerate(rings_per_cell) if len(rings) == 0]
    assert len(blank_cells) == 1, "exactly one blank cell is required"
    assert blank_cells[0] == cell_count - 1, "the blank cell must be the last cell"

    frames = rings_per_cell[:-1]
    assert len(frames) >= 3, "at least three visible frames are required"
    for i, rings in enumerate(frames, start=1):
        assert len(rings) == i, "frame i must contain i rings"

    stream: list[int] = []
    for frame_idx, rings in enumerate(frames, start=0):
        for ring_idx, color in enumerate(rings):
            pos = frame_idx + ring_idx
            if pos < len(stream):
                assert stream[pos] == color, "inconsistent ring stream"
            else:
                stream.append(int(color))

    period = _infer_cycle_period(stream)
    next_outer = stream[len(stream) % period]

    next_frame = list(frames[-1][1:]) + [int(next_outer)]
    return _build_frame(next_frame, bg)


def _render_frame(size: int, rings: int, palette: np.ndarray, phase: int, bg: int) -> np.ndarray:
    cell = np.full((size, size), bg, dtype=int)
    for ring in range(rings):
        color = int(palette[(ring + rings + phase) % len(palette)])
        _paint_ring(cell, ring, color)
    return cell


def generate_input() -> np.ndarray:
    palette_pool = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
    bg_choices = [Color.BLACK] + palette_pool
    attempts = 500

    for _ in range(attempts):
        bg = int(random.choice(bg_choices))
        cell_size = int(random.choice([5, 7]))
        palette_colors = [c for c in palette_pool if c != bg]
        palette_len = int(random.randint(3, min(5, len(palette_colors))))
        palette = np.array(random.sample(palette_colors, palette_len), dtype=int)
        phase = int(random.randint(0, palette_len - 1))

        frames = [_render_frame(cell_size, rings=t, palette=palette, phase=phase, bg=bg) for t in (1, 2, 3)]
        blank = np.full((cell_size, cell_size), bg, dtype=int)
        grid = np.hstack(frames + [blank])

        # Ensure generator constraints hold and keep enough space for ARC bounds.
        if grid.shape[1] > 30:
            continue
        if np.all(grid == bg):
            continue
        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid 2.py input after retries")
