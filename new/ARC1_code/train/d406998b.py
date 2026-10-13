import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every second grey marker from the right edge to green.

    Input structure
    The grid has three rows, black background, and exactly one grey foreground
    pixel in each column.

    Rule
    Starting with the rightmost column, recolor the grey foreground pixel in
    every other column to green; leave the skipped grey pixels and background
    unchanged.

    Key insight
    The foreground pixels form a left-to-right sequence, and the output marks
    alternating sequence positions counted from the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == 3, "expected the task's three-row marker strip"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors == [Color.GREY], "expected grey as the only foreground color"

    marker_mask = g == Color.GREY
    assert int(marker_mask.sum()) == width, "expected one marker per column"
    assert np.all(marker_mask.sum(axis=0) == 1), "expected one marker per column"

    out = g.copy()
    for col in range(width - 1, -1, -2):
        rows = np.flatnonzero(marker_mask[:, col])
        assert rows.size == 1, "expected exactly one marker in selected column"
        out[int(rows[0]), col] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------
GRID_HEIGHT = 3  # Grid dimensions and valid marker rows.
MIN_WIDTH, MAX_WIDTH = 10, 17
ROW_VALUES = tuple(range(GRID_HEIGHT))
BG_COLOR, MARKER_COLOR, OUTPUT_COLOR = Color.BLACK, Color.GREY, Color.GREEN
MAX_GENERATION_ATTEMPTS = 300  # Generation retry budget.


def _sample_marker_rows(width: int) -> list[int]:
    """Return row indices with no adjacent repeats and full row coverage."""
    rows: list[int] = []
    for col in range(width):
        previous = rows[-1] if rows else None
        choices = [row for row in ROW_VALUES if row != previous]
        missing = set(ROW_VALUES) - set(rows)
        remaining_after = width - col - 1
        if len(missing) > remaining_after:
            choices = [row for row in missing if row != previous]
        rows.append(random.choice(choices))
    return rows


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    rows = _sample_marker_rows(width)

    grid = np.full((GRID_HEIGHT, width), BG_COLOR, dtype=int)
    grid[np.array(rows), np.arange(width)] = MARKER_COLOR

    expected = grid.copy()
    for col in range(width - 1, -1, -2):
        expected[rows[col], col] = OUTPUT_COLOR
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for d406998b")
