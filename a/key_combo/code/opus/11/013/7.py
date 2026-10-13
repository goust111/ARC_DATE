from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

CORNER_CYCLE = (RED, BLUE)
INTERIOR_CYCLE = (GREEN, YELLOW, PINK)
GRID_BANDS = 4
GEN_RETRIES = 300


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1, *lines, limit]
    regions = [(a + 1, b - 1) for a, b in zip(bounds, bounds[1:])]
    assert all(r0 <= r1 for r0, r1 in regions)
    return regions


def _triangular_offsets(width: int) -> list[int]:
    offsets: list[int] = []
    pos = 0
    step = 1
    while pos < width:
        offsets.append(pos)
        pos += step
        step += 1
    return offsets


def _separator_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(rows) == GRID_BANDS - 1
    assert len(cols) == GRID_BANDS - 1
    assert rows[0] > 0 and rows[-1] < grid.shape[0] - 1
    assert cols[0] > 0 and cols[-1] < grid.shape[1] - 1

    expected = np.zeros_like(grid, dtype=bool)
    expected[rows, :] = True
    expected[:, cols] = True
    assert np.array_equal(grid == GREY, expected)
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """Fill only corner and strictly internal table cells using triangular-position color cycles.

    Input structure:
    - The input is a black background with grey full-width and full-height separator lines.
    - Those separators split the canvas into a 4x4 table of rectangular black cells.
    - Every non-separator cell is initially black, and there are no stray grey pixels away from the full lines.

    Rule:
    - Preserve the grey separator grid.
    - Treat the four table corners as one class and the four strictly internal cells as another; all remaining edge cells stay black.
    - In each corner cell, write the cycle [red, blue] on that cell's top row at triangular column offsets 0, 1, 3, 6, ... that fit.
    - In each strictly internal cell, write the cycle [green, yellow, pink] on that cell's top row at the same triangular offsets.

    Key insight:
    - The table position decides whether a cell is filled at all and which color cycle it uses, while the within-cell marks follow triangular-number spacing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}

    full_rows, full_cols = _separator_lines(g)
    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) == GRID_BANDS
    assert len(col_regions) == GRID_BANDS

    out = g.copy()
    for row_i, (r0, r1) in enumerate(row_regions):
        for col_i, (c0, c1) in enumerate(col_regions):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(cell == BLACK)

            cycle: tuple[int, ...] | None = None
            if (row_i, col_i) in {(0, 0), (0, 3), (3, 0), (3, 3)}:
                cycle = CORNER_CYCLE
            elif (row_i, col_i) in {(1, 1), (1, 2), (2, 1), (2, 2)}:
                cycle = INTERIOR_CYCLE

            if cycle is None:
                continue

            for idx, offset in enumerate(_triangular_offsets(c1 - c0 + 1)):
                out[r0, c0 + offset] = cycle[idx % len(cycle)]

    return out


def _distribute_extra(parts: int, total: int, rng: np.random.Generator) -> list[int]:
    extra = [0] * parts
    for _ in range(total):
        extra[int(rng.integers(0, parts))] += 1
    return extra


def _gen_row_heights(rng: np.random.Generator) -> list[int]:
    return [int(rng.integers(3, 7)) for _ in range(GRID_BANDS)]


def _gen_col_widths(rng: np.random.Generator) -> list[int]:
    total_extra = int(rng.choice(np.array([0, 1, 2, 3]), p=[0.15, 0.35, 0.30, 0.20]))
    widths = [6 + delta for delta in _distribute_extra(GRID_BANDS, total_extra, rng)]
    assert 24 <= sum(widths) <= 27
    return widths


def _build_input(row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    height = sum(row_heights) + (GRID_BANDS - 1)
    width = sum(col_widths) + (GRID_BANDS - 1)
    grid = np.zeros((height, width), dtype=int)

    r = row_heights[0]
    for band_h in row_heights[1:]:
        grid[r, :] = GREY
        r += 1 + band_h

    c = col_widths[0]
    for band_w in col_widths[1:]:
        grid[:, c] = GREY
        c += 1 + band_w

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_heights = _gen_row_heights(rng)
    col_widths = _gen_col_widths(rng)
    grid = _build_input(row_heights, col_widths)

    try:
        out = main(grid)
    except Exception:
        return None

    assert grid.shape == out.shape
    assert max(grid.shape) <= 30

    # Keep the triangular pattern visible in at least one filled column band often,
    # while still allowing some narrower truncation examples.
    if sum(w > 6 for w in col_widths) == 0 and int(rng.integers(0, 4)) != 0:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
