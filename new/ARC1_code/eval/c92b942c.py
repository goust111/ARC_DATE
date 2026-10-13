from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input 3 by 3, draw blue across each repeated foreground row, and draw green on nearby northwest-southeast diagonals.

    Input structure
    - The grid has a dominant background and exactly one non-background foreground color.
    - The foreground color may appear in one or more cells.
    - Blue and green are reserved trace colors, not source foreground colors.

    Rule
    - Repeat the whole input three times vertically and horizontally.
    - In the tiled canvas, every background cell in a row containing a repeated foreground pixel becomes blue.
    - Every still-background cell one step northwest or southeast of a repeated foreground pixel becomes green.

    Key insight
    The repeated foreground pixels define the output scaffold; horizontal row traces have priority, and diagonal marks only fill cells that remain background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    assert foreground_colors[0] not in (1, 3), "blue and green are reserved trace colors"

    out = np.tile(g, (3, 3))
    foreground_mask = out != bg
    foreground_coords = np.argwhere(foreground_mask)
    assert foreground_coords.size > 0, "expected at least one foreground pixel"

    for r in np.flatnonzero(np.any(foreground_mask, axis=1)):
        out[int(r), out[int(r)] == bg] = 1

    for r_raw, c_raw in foreground_coords:
        r = int(r_raw)
        c = int(c_raw)
        for delta in (-1, 1):
            nr = r + delta
            nc = c + delta
            if in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == bg:
                out[nr, nc] = 3

    return out

# --------------- generator (o5x) ---------------

# Canvas/density, foreground, and validation knobs.
MIN_H, MAX_H = 2, 10
MIN_W, MAX_W = 2, 10
MIN_FG_COUNT, MAX_FG_COUNT, MIN_FG_ROWS, MAX_FG_ROWS = 1, 3, 1, 2
MAX_FG_DENSITY, MAX_ATTEMPTS = 0.125, 300
BG_COLOR, TILE_FACTOR = 0, 3
FG_COLORS = (2, 4, 5, 6, 7, 8, 9)

VALID_SPECS = [
    (h, w, count)
    for h in range(MIN_H, MAX_H + 1)
    for w in range(MIN_W, MAX_W + 1)
    for count in range(MIN_FG_COUNT, MAX_FG_COUNT + 1)
    if count / (h * w) <= MAX_FG_DENSITY
]


def _gen_attempt() -> np.ndarray:
    h, w, fg_count = VALID_SPECS[int(np.random.randint(len(VALID_SPECS)))]

    grid = np.zeros((h, w), dtype=int)
    fg_color = int(np.random.choice(FG_COLORS))

    row_count_options = [
        n_rows
        for n_rows in range(MIN_FG_ROWS, min(MAX_FG_ROWS, fg_count, h) + 1)
        if fg_count <= n_rows * w
    ]
    row_count = row_count_options[int(np.random.randint(len(row_count_options)))]
    rows = np.random.choice(h, size=row_count, replace=False)
    row_loads = np.ones(row_count, dtype=int)
    for _ in range(fg_count - row_count):
        choices = np.flatnonzero(row_loads < w)
        row_loads[int(np.random.choice(choices))] += 1
    for r, n_cells in zip(rows, row_loads):
        cols = np.random.choice(w, size=int(n_cells), replace=False)
        grid[int(r), cols] = fg_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        try:
            out = main(inp)
        except AssertionError:
            continue
        if out.shape == (inp.shape[0] * TILE_FACTOR, inp.shape[1] * TILE_FACTOR):
            return inp
    raise RuntimeError("failed to generate a valid input")
