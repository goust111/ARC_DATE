from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg, paint_coords
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a teal L-shaped path from the red pixel to the green pixel, moving horizontally first and then vertically.

    Input structure
    - The grid has one background color, exactly one red marker, and exactly one green marker.
    - The red and green markers are in different rows and different columns.
    - Every other input cell is background.

    Rule
    - Start from a copy of the input.
    - Paint teal background cells across the red marker's row until reaching the green marker's column.
    - From that elbow, paint teal background cells along the green marker's column until just before the green marker.

    Key insight
    - The red marker supplies the starting row, and the green marker supplies the target column for a horizontal-then-vertical Manhattan path.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg not in (Color.RED, Color.GREEN, Color.TEAL), "background must be distinct from marker/path colors"
    assert set(map(int, np.unique(g))) == {bg, Color.RED, Color.GREEN}, "expected only background, red, and green"

    red = np.argwhere(g == Color.RED)
    green = np.argwhere(g == Color.GREEN)
    assert red.shape == (1, 2), "expected exactly one red marker"
    assert green.shape == (1, 2), "expected exactly one green marker"

    red_r, red_c = map(int, red[0])
    green_r, green_c = map(int, green[0])
    assert red_r != green_r and red_c != green_c, "markers must define a non-degenerate L path"

    c_step = 1 if green_c > red_c else -1
    r_step = 1 if green_r > red_r else -1
    path = [(red_r, c) for c in range(red_c + c_step, green_c + c_step, c_step)]
    path.extend((r, green_c) for r in range(red_r + r_step, green_r, r_step))
    assert path, "expected at least one path cell"
    assert all(int(g[r, c]) == bg for r, c in path), "path must pass only through background cells"

    out = g.copy()
    paint_coords(out, path, Color.TEAL)
    return out

# --------------- generator (o5x) ---------------

# Sampling bounds and retry budget.
GRID_ROWS_RANGE = (10, 15)
GRID_COLS_RANGE = (14, 16)
MARKER_MARGIN = 1
MANHATTAN_RANGE = (16, 18)
DIRECTION_SIGNS = (-1, 1)
MAX_ATTEMPTS = 300

def _l_path_coords(red: tuple[int, int], green: tuple[int, int]) -> list[tuple[int, int]]:
    """Return horizontal-first L path cells, excluding the red/green endpoints."""
    red_r, red_c = red
    green_r, green_c = green
    c_step = 1 if green_c > red_c else -1
    r_step = 1 if green_r > red_r else -1
    path = [(red_r, c) for c in range(red_c + c_step, green_c + c_step, c_step)]
    path.extend((r, green_c) for r in range(red_r + r_step, green_r, r_step))
    return path


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    red = tuple(map(int, np.argwhere(inp == Color.RED)[0]))
    green = tuple(map(int, np.argwhere(inp == Color.GREEN)[0]))
    paint_coords(out, _l_path_coords(red, green), Color.TEAL)
    return out


def _valid_offsets(h: int, w: int, manhattan: int) -> list[tuple[int, int]]:
    """Return marker deltas that fit both markers inside the margin."""
    max_dr, max_dc = h - 1 - 2 * MARKER_MARGIN, w - 1 - 2 * MARKER_MARGIN
    return [(dr, manhattan - dr) for dr in range(1, max_dr + 1) if 1 <= manhattan - dr <= max_dc]


def _choose_marker_pair(h: int, w: int, dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample red and green marker positions for a fixed row/column delta."""
    r_sign, c_sign = random.choice(DIRECTION_SIGNS), random.choice(DIRECTION_SIGNS)
    row_lo = MARKER_MARGIN if r_sign > 0 else MARKER_MARGIN + dr
    row_hi = h - MARKER_MARGIN - 1 - (dr if r_sign > 0 else 0)
    col_lo = MARKER_MARGIN if c_sign > 0 else MARKER_MARGIN + dc
    col_hi = w - MARKER_MARGIN - 1 - (dc if c_sign > 0 else 0)
    red_r, red_c = random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)
    return (red_r, red_c), (red_r + r_sign * dr, red_c + c_sign * dc)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    grid = np.zeros((h, w), dtype=int)

    manhattan = random.randint(*MANHATTAN_RANGE)
    offsets = _valid_offsets(h, w, manhattan)
    if not offsets:
        return None

    red, green = _choose_marker_pair(h, w, *random.choice(offsets))
    red_r, red_c = red
    green_r, green_c = green
    grid[red_r, red_c] = Color.RED
    grid[green_r, green_c] = Color.GREEN

    expected = _gen_expected_output(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a2fd1cf0 input")
