from __future__ import annotations

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the horizontal red/black segments bracketed by blue and teal markers, stacking them in marker-row order.

    Input structure
    - The grid is a black/red field with blue and teal control pixels.
    - Each marked row has exactly one blue control pixel to the left of exactly one teal control pixel.
    - All marked rows have the same number of cells between the two control pixels.

    Rule
    - Read marked rows from top to bottom.
    - For each marked row, copy the cells strictly between the blue and teal markers into the next output row.
    - Drop the marker pixels and all unmarked rows.

    Key insight
    - Each blue/teal pair brackets one output row; the puzzle is a compact extraction of those bracketed row segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rows = []
    for r in range(g.shape[0]):
        left_cols = np.flatnonzero(g[r] == Color.BLUE)
        right_cols = np.flatnonzero(g[r] == Color.TEAL)
        if len(left_cols) == 0 and len(right_cols) == 0:
            continue

        assert len(left_cols) == 1 and len(right_cols) == 1, "each marked row must have one blue and one teal marker"
        left = int(left_cols[0])
        right = int(right_cols[0])
        assert left < right, "blue marker must be left of teal marker"

        segment = g[r, left + 1 : right]
        assert segment.size > 0, "marker pair must bracket at least one cell"
        assert not np.isin(segment, [Color.BLUE, Color.TEAL]).any(), "marker colors only delimit segments"
        rows.append(segment.copy())

    assert rows, "expected at least one marked row"
    width = int(rows[0].size)
    assert all(int(row.size) == width for row in rows), "all bracketed segments must have equal width"

    return np.vstack(rows)

# --------------- generator (o5x) ---------------
# Canvas and structural bounds.
MARKER_COUNT_RANGE = (1, 5)
SEGMENT_WIDTH_RANGE = (3, 7)
NON_MARKER_COUNT_RANGE = (3, 9)
GRID_WIDTH_RANGE = (10, 14)
# Marker placement constraints.
MARKER_GAP_RANGE = (1, 3)
BLUE_COL_RANGE = (0, 5)
TEAL_COL_MAX = 10
BOTTOM_NON_MARKER_ROWS = 1
# Red density ranges.
SEGMENT_RED_FRACTION_RANGE = (0.30, 0.75)
BACKGROUND_RED_FRACTION_RANGE = (0.30, 0.60)
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _gen_red_black_row(width: int, red_fraction_range: tuple[float, float]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    if width <= 0:
        return row

    min_frac, max_frac = red_fraction_range
    lo = int(np.ceil(min_frac * width))
    hi = int(np.floor(max_frac * width))
    lo = max(0, min(width, lo))
    hi = max(lo, min(width, hi))
    red_count = _gen_randint(lo, hi)
    if red_count:
        cols = np.random.choice(width, size=red_count, replace=False)
        row[cols] = Color.RED
    return row


def _gen_bounded_composition(n: int, total: int, lo: int, hi: int) -> list[int]:
    parts: list[int] = []
    remaining = total
    for i in range(n):
        left = n - i - 1
        min_here = max(lo, remaining - hi * left)
        max_here = min(hi, remaining - lo * left)
        value = _gen_randint(min_here, max_here)
        parts.append(value)
        remaining -= value
    return parts


def _gen_marker_rows(height: int, marker_count: int) -> list[int]:
    last_marker_row = height - 1 - BOTTOM_NON_MARKER_ROWS
    if marker_count == 1:
        return [_gen_randint(0, last_marker_row)]

    min_span = MARKER_GAP_RANGE[0] * (marker_count - 1)
    max_span = min(MARKER_GAP_RANGE[1] * (marker_count - 1), last_marker_row)
    span = _gen_randint(min_span, max_span)
    start = _gen_randint(0, last_marker_row - span)
    deltas = _gen_bounded_composition(marker_count - 1, span, *MARKER_GAP_RANGE)

    rows = [start]
    for delta in deltas:
        rows.append(rows[-1] + delta)
    return rows


def _gen_marker_cols(marker_count: int, segment_width: int, grid_width: int) -> list[int]:
    max_blue = min(
        BLUE_COL_RANGE[1],
        grid_width - segment_width - 2,
        TEAL_COL_MAX - segment_width - 1,
    )
    choices = list(range(BLUE_COL_RANGE[0], max_blue + 1))
    cols = [int(np.random.choice(choices)) for _ in range(marker_count)]

    if marker_count > 1 and len(set(cols)) == 1 and len(choices) > 1:
        replace_at = _gen_randint(0, marker_count - 1)
        alternatives = [c for c in choices if c != cols[replace_at]]
        cols[replace_at] = int(np.random.choice(alternatives))
    return cols


def _gen_attempt() -> np.ndarray | None:
    marker_count = _gen_randint(*MARKER_COUNT_RANGE)
    segment_width = _gen_randint(*SEGMENT_WIDTH_RANGE)
    output = np.vstack(
        [_gen_red_black_row(segment_width, SEGMENT_RED_FRACTION_RANGE) for _ in range(marker_count)]
    )

    non_marker_count = _gen_randint(*NON_MARKER_COUNT_RANGE)
    height = marker_count + non_marker_count
    grid_width = _gen_randint(*GRID_WIDTH_RANGE)
    grid = np.vstack(
        [_gen_red_black_row(grid_width, BACKGROUND_RED_FRACTION_RANGE) for _ in range(height)]
    )

    marker_rows = _gen_marker_rows(height, marker_count)
    blue_cols = _gen_marker_cols(marker_count, segment_width, grid_width)
    for out_r, (grid_r, blue_col) in enumerate(zip(marker_rows, blue_cols)):
        teal_col = blue_col + 1 + segment_width
        row = _gen_red_black_row(grid_width, BACKGROUND_RED_FRACTION_RANGE)
        row[blue_col] = Color.BLUE
        row[blue_col + 1 : teal_col] = output[out_r]
        row[teal_col] = Color.TEAL
        grid[grid_r] = row

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 505fff84 input")
