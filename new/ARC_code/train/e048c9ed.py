import numpy as np

from arc_utils import Color, get_bg
import random


_QUANTITY_TO_COLOR = {2: Color.BLUE, 3: Color.YELLOW, 4: Color.MAROON, 5: Color.PINK}


def main(grid: np.ndarray) -> np.ndarray:
    """Add a quantity-color label in the top marker column for each horizontal bar.

    Input structure
    - The grid has a black background and one non-background marker pixel in the top row.
    - Below the marker row, every non-empty source row contains one solid monochrome horizontal bar to the left of the marker column.
    - Source rows are staggered every other row.

    Rule
    - Copy the input grid.
    - For each source bar, read its quantity as the bar length.
    - Paint the quantity color in the marker column on the same row as the bar:
      2->1, 3->4, 4->9, 5->6.

    Key insight
    - The grey marker supplies the result column, while each bar supplies a small quantity with a fixed color code.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == 0, "expected black background for square-digit labels"

    marker_cols = np.flatnonzero(g[0] != bg)
    assert len(marker_cols) == 1, "expected exactly one top-row marker"
    marker_c = int(marker_cols[0])
    assert g[0, marker_c] == Color.GREY, "top-row marker must be grey"

    source_rows = [r for r in range(1, g.shape[0]) if np.any(g[r] != bg)]
    assert source_rows, "expected at least one source bar"
    first_source_row = source_rows[0]

    out = g.copy()
    for bar_order, r in enumerate(source_rows):
        assert (r - first_source_row) % 2 == 0, "source bars must follow the staggered row spacing"
        cols = np.flatnonzero(g[r] != bg)
        c0, c1 = int(cols[0]), int(cols[-1])
        expected_cols = np.arange(c0, c1 + 1)
        assert np.array_equal(cols, expected_cols), "source row must contain one contiguous bar"
        assert np.unique(g[r, c0 : c1 + 1]).size == 1, "source bar must be monochrome"
        assert c1 < marker_c, "source bar must lie to the left of the marker column"
        assert g[r, marker_c] == bg, "marker-column output cell must be empty"

        bar_length = c1 - c0 + 1
        assert 2 <= bar_length <= 5, "bar length quantity must be 2-5"
        assert bar_length in _QUANTITY_TO_COLOR, "quantity must be one of 2, 3, 4, or 5"
        out[r, marker_c] = _QUANTITY_TO_COLOR[bar_length]

    return out

# --------------- generator (o5x) ---------------

# Row/bar structure.
BAR_COUNT_RANGE, FIRST_ROWS_SINGLE, FIRST_ROWS_MULTI = (1, 4), (2,), (1, 2)
ROW_STRIDE, BAR_LENGTHS, START_COL_RANGE = 2, (2, 3, 4, 5), (0, 2)
# Marker/canvas placement.
MARKER_COL_RANGE, MARKER_WEIGHTS = (2, 9), (8, 1)
WIDTH_EXTRA_RANGE, MAX_WIDTH = (1, 8), 12
# Colors and retry budget.
COLOR_RANGE, MARKER_COLOR, BG_COLOR = (1, 9), Color.GREY, 0
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    num_bars = random.randint(*BAR_COUNT_RANGE)
    first_rows = FIRST_ROWS_SINGLE if num_bars == 1 else FIRST_ROWS_MULTI
    first_src_row = random.choice(first_rows)
    height = ROW_STRIDE * (first_src_row + num_bars - 1)

    bar_lengths = random.sample(BAR_LENGTHS, k=num_bars)
    start_cols = [random.randint(*START_COL_RANGE) for _ in range(num_bars)]
    max_bar_end = max(c + length - 1 for c, length in zip(start_cols, bar_lengths))

    marker_min = max(MARKER_COL_RANGE[0], max_bar_end + 1)
    if marker_min > MARKER_COL_RANGE[1]:
        return None
    marker_choices = list(range(marker_min, MARKER_COL_RANGE[1] + 1))
    marker_weights = [MARKER_WEIGHTS[0]] + [MARKER_WEIGHTS[1]] * (len(marker_choices) - 1)
    marker_col = random.choices(marker_choices, weights=marker_weights, k=1)[0]
    width = random.randint(marker_col + WIDTH_EXTRA_RANGE[0], min(MAX_WIDTH, marker_col + WIDTH_EXTRA_RANGE[1]))

    grid = np.zeros((height, width), dtype=int)
    grid[0, marker_col] = MARKER_COLOR

    for i, (start_col, bar_length) in enumerate(zip(start_cols, bar_lengths)):
        row = first_src_row + ROW_STRIDE * i
        grid[row, start_col : start_col + bar_length] = random.randint(*COLOR_RANGE)

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
