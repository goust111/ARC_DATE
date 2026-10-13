import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two equal-size input masks and mark cells that are background in both.

    Input structure
    The grid is a vertical stack of two same-width rectangular masks.  Each half
    contains the same background color plus exactly one foreground color.

    Rule
    Return one half's shape, filled with the background color, and recolor a
    cell red when the corresponding cell is background in both input halves.

    Key insight
    The output is the common empty-space mask of the two stacked patterns, not
    a crop or a transformation of either foreground object.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H % 2 == 0

    h = H // 2
    top = g[:h, :]
    bottom = g[h:, :]

    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    bg_colors = top_colors & bottom_colors
    assert len(bg_colors) == 1
    bg = bg_colors.pop()
    assert len(top_colors - {bg}) == 1
    assert len(bottom_colors - {bg}) == 1
    assert Color.RED not in top_colors | bottom_colors

    out = np.full((h, W), bg, dtype=int)
    out[(top == bg) & (bottom == bg)] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# CFG: half-grid geometry and colors.
HALF_H, HALF_W = 4, 4
BG_COLOR, TOP_FG_COLOR, BOTTOM_FG_COLOR, OUTPUT_MARK_COLOR = 0, 3, 1, Color.RED

# CFG: foreground/background overlap policy.
TOP_FG_COUNTS, BOTTOM_FG_COUNTS, BOTH_BG_COUNTS = range(4, 12), range(5, 13), range(1, 5)
MAX_GENERATION_ATTEMPTS = 300

CELLS = np.arange(HALF_H * HALF_W)
VALID_COUNT_COMBINATIONS = tuple(
    (top_count, bottom_count, both_bg_count, intersection)
    for top_count in TOP_FG_COUNTS
    for bottom_count in BOTTOM_FG_COUNTS
    for both_bg_count in BOTH_BG_COUNTS
    for intersection in (top_count + bottom_count + both_bg_count - CELLS.size,)
    if 0 <= intersection <= min(top_count, bottom_count)
)


def _paint_cells(cells, color):
    half = np.full((HALF_H, HALF_W), BG_COLOR, dtype=int)
    rows, cols = np.divmod(list(cells), HALF_W)
    half[rows, cols] = color
    return half


def _gen_attempt():
    rng = np.random.default_rng()
    count_index = int(rng.integers(len(VALID_COUNT_COMBINATIONS)))
    top_count, _bottom_count, both_bg_count, intersection = VALID_COUNT_COMBINATIONS[count_index]

    both_bg = set(map(int, rng.choice(CELLS, size=both_bg_count, replace=False)))
    union = np.array([cell for cell in CELLS if int(cell) not in both_bg], dtype=int)

    shared_fg = set(map(int, rng.choice(union, size=intersection, replace=False)))
    remaining = np.array([cell for cell in union if int(cell) not in shared_fg], dtype=int)
    top_only_count = top_count - intersection
    top_only = set(map(int, rng.choice(remaining, size=top_only_count, replace=False)))
    bottom_only = set(map(int, remaining)) - top_only

    top = _paint_cells(shared_fg | top_only, TOP_FG_COLOR)
    bottom = _paint_cells(shared_fg | bottom_only, BOTTOM_FG_COLOR)
    return np.vstack([top, bottom])


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        expected = np.full((HALF_H, HALF_W), BG_COLOR, dtype=int)
        expected[(inp[:HALF_H] == BG_COLOR) & (inp[HALF_H:] == BG_COLOR)] = OUTPUT_MARK_COLOR
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate valid input")
