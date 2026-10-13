import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the top grey vertical bar into teal fill on its left and pink fill on its right.

    Input structure:
    - The grid has a black background and exactly one foreground object: a grey vertical bar.
    - The grey bar is one pixel wide, starts at the top edge, and has no gaps.
    - There is at least one column on each side of the bar.

    Rule:
    - Preserve the grey bar.
    - For each column d steps left of the bar, paint teal from the top for bar_height + 2*d cells.
    - For each column d steps right of the bar, paint pink from the top for bar_height - 2*d cells, stopping when this is nonpositive.
    - Clip all painted heights to the grid.

    Key insight:
    - Horizontal distance from the top bar controls the vertical extent: the left side grows by two cells per step, while the right side shrinks by two cells per step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    fg_colors = colors_present(g, exclude=[bg])
    assert fg_colors == [Color.GREY], "expected only a grey foreground bar"

    bar_coords = np.argwhere(g == Color.GREY)
    assert len(bar_coords) > 0, "expected a nonempty grey bar"
    bar_cols = np.unique(bar_coords[:, 1])
    assert len(bar_cols) == 1, "expected the grey bar to be vertical"
    bar_col = int(bar_cols[0])
    assert 0 < bar_col < width - 1, "expected space on both sides of the bar"

    bar_rows = np.sort(bar_coords[:, 0])
    bar_height = len(bar_rows)
    assert np.array_equal(bar_rows, np.arange(bar_height)), "expected a top-anchored contiguous bar"

    out = g.copy()

    for offset in range(1, bar_col + 1):
        paint_height = min(height, bar_height + 2 * offset)
        out[:paint_height, bar_col - offset] = Color.TEAL

    for offset in range(1, width - bar_col):
        paint_height = bar_height - 2 * offset
        if paint_height <= 0:
            break
        out[: min(height, paint_height), bar_col + offset] = Color.PINK

    out[bar_rows, bar_col] = Color.GREY
    return out

# --------------- generator (o5x) ---------------
# Canvas and bar bounds.
HEIGHT_RANGE = (11, 16)
WIDTH_RANGE = (7, 10)
BAR_COL_RANGE = (2, 5)
BAR_HEIGHT_RANGE = (4, 7)
# Margin, fill, and retry controls.
RIGHT_COLS_RANGE = (3, 5)
FILL_STEP = 2
GEN_MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*HEIGHT_RANGE))
    width = int(np.random.randint(*WIDTH_RANGE))
    valid_bar_cols = [
        c
        for c in range(*BAR_COL_RANGE)
        if BAR_COL_RANGE[0] <= c <= width - RIGHT_COLS_RANGE[0] - 1
        and RIGHT_COLS_RANGE[0] <= width - c - 1 <= RIGHT_COLS_RANGE[1]
    ]
    if not valid_bar_cols:
        return None
    bar_col = int(np.random.choice(valid_bar_cols))
    max_bar_height = min(height - 1, BAR_HEIGHT_RANGE[1])
    bar_height = int(np.random.randint(BAR_HEIGHT_RANGE[0], max_bar_height + 1))
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[:bar_height, bar_col] = Color.GREY
    expected = grid.copy()
    for offset in range(1, bar_col + 1):
        paint_height = min(height, bar_height + FILL_STEP * offset)
        expected[:paint_height, bar_col - offset] = Color.TEAL
    for offset in range(1, width - bar_col):
        paint_height = bar_height - FILL_STEP * offset
        if paint_height <= 0:
            break
        expected[: min(height, paint_height), bar_col + offset] = Color.PINK
    expected[:bar_height, bar_col] = Color.GREY
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 5207a7b5")
