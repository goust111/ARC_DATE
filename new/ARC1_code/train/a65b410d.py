import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill triangular green and blue wedges around a left-anchored red horizontal segment.

    Input structure:
    - The grid has an empty background and exactly one foreground object.
    - The foreground object is a red horizontal segment starting at the left edge.

    Rule:
    - Preserve the red segment.
    - Above the red segment, fill left-aligned rows green with width equal to the segment length plus distance above.
    - Below the red segment, fill left-aligned rows blue with width equal to the segment length minus distance below.

    Key insight:
    - The red segment is the shared base width; moving upward expands the fill by one cell per row, while moving downward shrinks it by one cell per row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert colors == [Color.RED], "expected only a red foreground segment"

    red = np.argwhere(g == Color.RED)
    assert red.size > 0, "expected red pixels"
    rows = np.unique(red[:, 0])
    assert len(rows) == 1, "red segment must occupy one row"

    bar_row = int(rows[0])
    bar_cols = np.sort(red[:, 1])
    bar_len = len(bar_cols)
    assert np.array_equal(bar_cols, np.arange(bar_len)), "red segment must start at the left edge and be contiguous"

    out = g.copy()
    for r in range(bar_row):
        fill_width = bar_len + (bar_row - r)
        assert fill_width <= width, "green wedge must fit inside the grid"
        out[r, :fill_width] = Color.GREEN

    for r in range(bar_row + 1, height):
        fill_width = bar_len - (r - bar_row)
        if fill_width <= 0:
            break
        out[r, :fill_width] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------
# Bounds, colors, and retry budget.
GRID_MIN, GRID_MAX_EXCLUSIVE = 7, 10
BAR_LEN_MIN, BAR_LEN_MAX = 2, 5
BAR_ROW_MIN, BAR_ROW_MAX = 2, 3
BACKGROUND_COLOR = 0
GEN_RETRY_LIMIT = 300

def _gen_expected_output(inp: np.ndarray, bar_row: int, bar_len: int) -> np.ndarray:
    out = inp.copy()
    for r in range(bar_row):
        fill_width = bar_len + (bar_row - r)
        out[r, :fill_width] = Color.GREEN
    for r in range(bar_row + 1, min(inp.shape[0], bar_row + bar_len)):
        fill_width = bar_len - (r - bar_row)
        out[r, :fill_width] = Color.BLUE
    return out


def _sample_bar(height: int, width: int) -> tuple[int, int] | None:
    max_bar_len = min(BAR_LEN_MAX, height - BAR_ROW_MIN, width - BAR_LEN_MIN)
    if max_bar_len < BAR_LEN_MIN:
        return None
    bar_len = int(np.random.randint(BAR_LEN_MIN, max_bar_len + 1))
    max_bar_row = min(height - bar_len, width - bar_len)
    if max_bar_row < BAR_ROW_MIN:
        return None
    max_bar_row = min(BAR_ROW_MAX, max_bar_row)
    return int(np.random.randint(BAR_ROW_MIN, max_bar_row + 1)), bar_len


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(GRID_MIN, GRID_MAX_EXCLUSIVE))
    width = int(np.random.randint(GRID_MIN, GRID_MAX_EXCLUSIVE))
    bar = _sample_bar(height, width)
    if bar is None:
        return None
    bar_row, bar_len = bar
    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    inp[bar_row, :bar_len] = Color.RED

    expected = _gen_expected_output(inp, bar_row, bar_len)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
