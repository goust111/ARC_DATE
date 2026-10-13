from __future__ import annotations

import random

import numpy as np


BG_COLOR = 7
FILL_COLOR = 9
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 8)

MIN_ACTIVE_ROWS = 2
MAX_ACTIVE_ROWS = 6
MIN_ACTIVE_COLS = 2
MAX_ACTIVE_COLS = 6
MIN_GRID_H = 6
MAX_GRID_H = 16
MIN_GRID_W = 6
MAX_GRID_W = 16
MAX_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress scattered colored pixels into their active row/column intersection table.

    Input structure
    - Background is orange.
    - Non-background cells are isolated marker pixels scattered across the grid.

    Rule
    - Collect every row that contains at least one non-background marker.
    - Collect every column that contains at least one non-background marker.
    - Build the compact matrix induced by those rows and columns.
    - Keep the original color where a marker exists at that intersection.
    - Fill missing intersections with maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    active_rows = np.flatnonzero(np.any(g != BG_COLOR, axis=1))
    active_cols = np.flatnonzero(np.any(g != BG_COLOR, axis=0))
    assert active_rows.size > 0 and active_cols.size > 0, "expected at least one marker pixel"

    out = np.full((active_rows.size, active_cols.size), FILL_COLOR, dtype=int)
    for out_r, src_r in enumerate(active_rows):
        row_vals = g[int(src_r), active_cols]
        present = row_vals != BG_COLOR
        out[int(out_r), present] = row_vals[present]
    return out


def _sample_positions(length: int, count: int, require_gap: bool) -> list[int]:
    for _ in range(60):
        pos = sorted(random.sample(range(length), count))
        if not require_gap or any(b - a > 1 for a, b in zip(pos, pos[1:])):
            return pos
    return sorted(random.sample(range(length), count))


def _base_presence_mask(nrows: int, ncols: int) -> np.ndarray:
    mask = np.zeros((nrows, ncols), dtype=bool)

    if nrows <= ncols:
        primary_cols = random.sample(range(ncols), nrows)
        random.shuffle(primary_cols)
        for r, c in enumerate(primary_cols):
            mask[r, c] = True

        remaining_cols = [c for c in range(ncols) if c not in primary_cols]
        row_order = list(range(nrows))
        random.shuffle(row_order)
        for i, c in enumerate(remaining_cols):
            mask[row_order[i % nrows], c] = True
    else:
        primary_rows = random.sample(range(nrows), ncols)
        random.shuffle(primary_rows)
        for c, r in enumerate(primary_rows):
            mask[r, c] = True

        remaining_rows = [r for r in range(nrows) if r not in primary_rows]
        col_order = list(range(ncols))
        random.shuffle(col_order)
        for i, r in enumerate(remaining_rows):
            mask[r, col_order[i % ncols]] = True

    return mask


def _sample_presence_mask(nrows: int, ncols: int) -> np.ndarray:
    total = nrows * ncols
    mask = _base_presence_mask(nrows, ncols)
    base_count = int(mask.sum())

    if random.random() < 0.5:
        lo = base_count
        hi = max(base_count, total // 2)
    else:
        lo = max(base_count, (total + 1) // 2)
        hi = total - 1
    target_count = random.randint(lo, hi)

    empty_cells = [(r, c) for r in range(nrows) for c in range(ncols) if not mask[r, c]]
    random.shuffle(empty_cells)
    for r, c in empty_cells[: max(0, target_count - base_count)]:
        mask[r, c] = True

    return mask


def _paint_compact(mask: np.ndarray) -> np.ndarray:
    nrows, ncols = mask.shape
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    count = len(cells)

    if count == 2:
        num_colors = 2
    else:
        num_colors = random.randint(2, min(4, count - 1))
    colors = random.sample(MARKER_COLORS, num_colors)

    compact = np.full((nrows, ncols), FILL_COLOR, dtype=int)
    mode = random.choice(("random", "row", "col", "checker"))

    if mode == "row":
        row_colors = [random.choice(colors) for _ in range(nrows)]
        for r, c in cells:
            compact[r, c] = row_colors[r]
    elif mode == "col":
        col_colors = [random.choice(colors) for _ in range(ncols)]
        for r, c in cells:
            compact[r, c] = col_colors[c]
    elif mode == "checker":
        stride = random.randint(1, 3)
        for r, c in cells:
            compact[r, c] = colors[(r + stride * c) % num_colors]
    else:
        for r, c in cells:
            compact[r, c] = random.choice(colors)

    random.shuffle(cells)
    for i, color in enumerate(colors):
        r, c = cells[i]
        compact[r, c] = color

    if len({int(compact[r, c]) for r, c in cells}) == 1 and count >= 2:
        r, c = cells[0]
        compact[r, c] = colors[1]

    return compact


def _gen_attempt() -> np.ndarray | None:
    nrows = random.randint(MIN_ACTIVE_ROWS, MAX_ACTIVE_ROWS)
    ncols = random.randint(MIN_ACTIVE_COLS, MAX_ACTIVE_COLS)

    height = random.randint(max(MIN_GRID_H, nrows + 2), MAX_GRID_H)
    width = random.randint(max(MIN_GRID_W, ncols + 2), MAX_GRID_W)

    rows = _sample_positions(height, nrows, require_gap=random.random() < 0.7)
    cols = _sample_positions(width, ncols, require_gap=random.random() < 0.7)

    mask = _sample_presence_mask(nrows, ncols)
    compact = _paint_compact(mask)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for rr, src_r in enumerate(rows):
        for cc, src_c in enumerate(cols):
            value = int(compact[rr, cc])
            if value != FILL_COLOR:
                grid[int(src_r), int(src_c)] = value

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out, compact):
        return None
    if not np.any(out == FILL_COLOR):
        return None

    input_colors = {int(v) for v in np.unique(grid) if int(v) != BG_COLOR}
    if len(input_colors) < 2:
        return None

    marker_rows = np.count_nonzero(grid != BG_COLOR, axis=1)
    marker_cols = np.count_nonzero(grid != BG_COLOR, axis=0)
    if marker_rows.max() < 2 and marker_cols.max() < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
