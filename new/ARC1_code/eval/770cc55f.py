import numpy as np

from arc_utils import Color, Color as _gen_Color, get_bg
import random as _gen_random


def _edge_bar(row: np.ndarray, bg: int, separator: int) -> tuple[list[int], int]:
    cols = [c for c, value in enumerate(row) if int(value) not in {bg, separator}]
    assert cols, "expected a colored edge bar"
    assert cols == list(range(cols[0], cols[-1] + 1)), "edge bar must be contiguous"

    colors = {int(row[c]) for c in cols}
    assert len(colors) == 1, "edge bar must be monochrome"
    color = colors.pop()
    assert color != Color.YELLOW, "edge bar must be distinct from the fill color"
    return cols, color


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the gap on the longer edge-bar side in the columns where the two edge bars overlap.

    Input structure
    - The grid has a black dominant background.
    - One non-background internal row is a full-width red separator.
    - The first and last rows each contain one contiguous horizontal bar of the same non-red color.

    Rule
    - Find the column sets occupied by the top and bottom edge bars.
    - Compute their column overlap.
    - If the top bar is longer, paint yellow through the overlap columns from below the top row down to the separator.
    - If the bottom bar is longer, paint yellow through the overlap columns from below the separator down to above the bottom row.
    - Leave all original non-background pixels unchanged.

    Key insight
    - The side to fill is chosen by the longer edge bar, while the fill width comes from the shared horizontal projection of both bars.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= 3 and W >= 1, "expected room for two edge bars and a separator"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    separator_rows = [
        r
        for r in range(1, H - 1)
        if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    assert len(separator_rows) == 1, "expected exactly one internal full-width separator row"
    separator_r = separator_rows[0]
    separator = int(g[separator_r, 0])
    assert separator == Color.RED, "expected a red separator row"

    top_cols, top_color = _edge_bar(g[0], bg, separator)
    bottom_cols, bottom_color = _edge_bar(g[-1], bg, separator)
    assert top_color == bottom_color, "edge bars must have the same color"

    overlap = sorted(set(top_cols) & set(bottom_cols))
    assert overlap, "edge bars must overlap in at least one column"
    assert len(top_cols) != len(bottom_cols), "one edge bar must be longer"

    out = g.copy()
    if len(top_cols) > len(bottom_cols):
        fill_rows = range(1, separator_r)
    else:
        fill_rows = range(separator_r + 1, H - 1)

    for r in fill_rows:
        for c in overlap:
            assert out[r, c] == bg, "fill region must start as background"
            out[r, c] = Color.YELLOW

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
_GEN_HEIGHTS = [9, 11, 13]
_GEN_WIDTH_BOUNDS = (4, 6)

# Fixed colors.
_GEN_BACKGROUND_COLOR = _gen_Color.BLACK
_GEN_SEPARATOR_COLOR = _gen_Color.RED
_GEN_FILL_COLOR = _gen_Color.YELLOW
_GEN_BAR_COLORS = [
    _gen_Color.BLUE, _gen_Color.GREEN, _gen_Color.GREY,
    _gen_Color.PINK, _gen_Color.ORANGE, _gen_Color.MAROON,
]

# Edge bar and retry limits.
_GEN_MAX_BAR_LENGTH = 5
_GEN_RETRY_LIMIT = 300


def _gen_segment_start(width: int, length: int, required_col: int) -> int:
    """Sample a valid start so the segment includes required_col."""
    lo = max(0, required_col - length + 1)
    hi = min(required_col, width - length)
    assert lo <= hi
    return _gen_random.randint(lo, hi)


def _gen_edge_bars(width: int) -> tuple[list[int], list[int]]:
    """Return overlapping top/bottom bar columns with unequal lengths."""
    max_length = min(width, _GEN_MAX_BAR_LENGTH)
    length_pairs = [
        (long_length, short_length)
        for long_length in range(2, max_length + 1)
        for short_length in range(1, long_length)
    ]
    long_length, short_length = _gen_random.choice(length_pairs)

    overlap_col = _gen_random.randrange(width)
    long_start = _gen_segment_start(width, long_length, overlap_col)
    short_start = _gen_segment_start(width, short_length, overlap_col)

    long_cols = list(range(long_start, long_start + long_length))
    short_cols = list(range(short_start, short_start + short_length))
    if bool(_gen_random.getrandbits(1)):
        return long_cols, short_cols
    return short_cols, long_cols


def _gen_expected_output(grid, top_cols, bottom_cols, separator_r) -> np.ndarray:
    """Render the expected solver output for the sampled input."""
    expected = grid.copy()
    overlap = sorted(set(top_cols) & set(bottom_cols))
    if len(top_cols) > len(bottom_cols):
        fill_rows = slice(1, separator_r)
    else:
        fill_rows = slice(separator_r + 1, grid.shape[0] - 1)
    expected[fill_rows, overlap] = _GEN_FILL_COLOR
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = _gen_random.choice(_GEN_HEIGHTS)
    width = _gen_random.randint(*_GEN_WIDTH_BOUNDS)
    separator_r = height // 2

    grid = np.full((height, width), _GEN_BACKGROUND_COLOR, dtype=int)
    grid[separator_r, :] = _GEN_SEPARATOR_COLOR
    bar_color = _gen_random.choice(_GEN_BAR_COLORS)

    top_cols, bottom_cols = _gen_edge_bars(width)
    grid[0, top_cols] = bar_color
    grid[-1, bottom_cols] = bar_color
    expected = _gen_expected_output(grid, top_cols, bottom_cols, separator_r)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 770cc55f input")
