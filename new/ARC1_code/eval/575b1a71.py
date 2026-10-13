from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the foreground marker pixels by the left-to-right order of the columns they occupy.

    Input structure:
    - The grid has one dominant background color and one foreground marker color.
    - Foreground pixels appear in exactly four distinct columns.

    Rule:
    - Keep all background pixels unchanged.
    - Sort the foreground-occupied columns from left to right.
    - Recolor foreground pixels in those columns blue, red, green, and yellow respectively.

    Key insight:
    - Connectivity is irrelevant; every foreground pixel's output color is determined by
      the rank of its column among the occupied foreground columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    palette = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
    assert bg not in palette, "background must be distinct from the output palette"

    marker_colors = colors_present(g, exclude={bg})
    assert len(marker_colors) == 1, "expected exactly one foreground marker color"
    marker = marker_colors[0]
    assert marker not in palette, "marker color must be distinct from the output palette"

    marker_mask = g == marker
    assert np.any(marker_mask), "expected foreground marker pixels"
    marker_cols = np.where(marker_mask)[1]

    occupied_cols = sorted({int(c) for c in marker_cols})
    assert len(occupied_cols) == len(palette), "expected exactly four marker columns"

    out = g.copy()
    col_index = np.arange(g.shape[1])
    for col, color in zip(occupied_cols, palette):
        out[marker_mask & (col_index == col)] = color

    return out

# --------------- generator (o5x) ---------------
from itertools import product

# Canvas and colors.
GRID_SHAPE = (10, 10)
BG_COLOR, MARKER_COLOR = 5, 0
OUTPUT_PALETTE = (1, 2, 3, 4)
COLUMN_COUNT = len(OUTPUT_PALETTE)
# Sampling ranges use numpy's high-exclusive convention.
TOTAL_PIXEL_RANGE, PER_COLUMN_COUNT_RANGE = range(8, 16), range(1, 5)
ROW_COUNT_RANGE, COLUMN_GAP_RANGE = range(6, 9), range(1, 5)
GENERATION_RETRIES = 300
COUNT_OPTIONS_BY_TOTAL = {
    total: [counts for counts in product(PER_COLUMN_COUNT_RANGE, repeat=COLUMN_COUNT) if sum(counts) == total]
    for total in TOTAL_PIXEL_RANGE
}
GAP_OPTIONS = [
    gaps
    for gaps in product(COLUMN_GAP_RANGE, repeat=COLUMN_COUNT - 1)
    if sum(gaps) <= GRID_SHAPE[1] - 1
]


def _sample_layout(rng) -> tuple[tuple[int, ...], list[int]]:
    """Return per-column counts and four occupied columns."""
    total = int(rng.integers(TOTAL_PIXEL_RANGE.start, TOTAL_PIXEL_RANGE.stop))
    options = COUNT_OPTIONS_BY_TOTAL[total]
    counts = options[int(rng.integers(len(options)))]
    gaps = GAP_OPTIONS[int(rng.integers(len(GAP_OPTIONS)))]
    start = int(rng.integers(0, GRID_SHAPE[1] - sum(gaps)))
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    return counts, cols


def _assign_rows_by_col(rng, counts) -> list[set[int]] | None:
    """Assign sampled row positions while preserving per-column counts."""
    row_count = int(rng.integers(ROW_COUNT_RANGE.start, ROW_COUNT_RANGE.stop))
    target_rows = rng.choice(GRID_SHAPE[0], size=row_count, replace=False).tolist()
    rng.shuffle(target_rows)

    slots = [col_i for col_i, n in enumerate(counts) for _ in range(n)]
    rng.shuffle(slots)
    rows_by_col = [set() for _ in range(COLUMN_COUNT)]

    for row, col_i in zip(target_rows, slots[:row_count]):
        rows_by_col[col_i].add(int(row))

    for col_i in slots[row_count:]:
        available = [row for row in target_rows if row not in rows_by_col[col_i]]
        if not available:
            return None
        rows_by_col[col_i].add(int(available[int(rng.integers(len(available)))]))

    return rows_by_col


def _render_validated(cols, rows_by_col) -> np.ndarray | None:
    """Render the grid and return it only when main() matches the oracle."""
    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    expected = grid.copy()
    for col_i, col in enumerate(cols):
        rows = list(rows_by_col[col_i])
        grid[rows, col] = MARKER_COLOR
        expected[rows, col] = OUTPUT_PALETTE[col_i]
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    counts, cols = _sample_layout(rng)
    rows_by_col = _assign_rows_by_col(rng, counts)
    if rows_by_col is None:
        return None
    return _render_validated(cols, rows_by_col)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
