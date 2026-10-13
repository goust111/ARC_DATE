import numpy as np

from arc_utils import Color, get_bg


_SIGNAL_COLORS = {Color.RED, Color.GREY}


def _alternate(color: int, steps: int) -> int:
    assert color in _SIGNAL_COLORS, "source color must be red or grey"
    if steps % 2 == 0:
        return color
    return Color.GREY if color == Color.RED else Color.RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the background under a top-row red/grey span with a shrinking diagonal pattern that alternates red and grey.

    Input structure:
    - The grid has an orange background.
    - All foreground pixels are red or grey and lie on the top row.
    - The first and last top-row foreground pixels define an odd-width span.

    Rule:
    - Copy the input grid.
    - For row r below the top, fill columns span_left + r through span_right - r.
    - Each filled cell takes the top-row color r columns up-left, flipped red<->grey once per row step.

    Key insight:
    - The output is the top-row span propagated down-right along diagonals, while the allowable region shrinks inward from both ends each row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _ = map(int, g.shape)

    bg = get_bg(g)
    assert bg == Color.ORANGE, "expected orange background"

    foreground = np.argwhere(g != bg)
    assert foreground.size > 0, "expected top-row foreground pixels"
    assert np.all(foreground[:, 0] == 0), "all input foreground must be on the top row"
    assert all(int(g[r, c]) in _SIGNAL_COLORS for r, c in foreground), "foreground must be red or grey"

    top_cols = foreground[:, 1]
    left = int(top_cols.min())
    right = int(top_cols.max())
    span_width = right - left + 1
    assert span_width >= 3 and span_width % 2 == 1, "expected an odd top-row span of width at least 3"

    out = g.copy()
    for row in range(1, height):
        lo = left + row
        hi = right - row
        if lo > hi:
            break
        for col in range(lo, hi + 1):
            source = int(g[0, col - row])
            out[row, col] = _alternate(source, row)

    return out

# --------------- generator (att2) ---------------

# Generator configuration: colors, sampled span sizes, canvas padding, retries.
GEN_BG = Color.ORANGE
GEN_SIGNAL_COLORS = np.array([Color.RED, Color.GREY], dtype=int)
SPAN_WIDTHS = np.arange(3, 16, 2)
WIDTH_WITH_GAP = 3
EXTRA_HEIGHT_RANGE, EXTRA_WIDTH_RANGE = (0, 2), (0, 3)
MAX_GENERATION_ATTEMPTS = 300


def _expected_output(grid: np.ndarray, left: int, span_width: int) -> np.ndarray:
    expected = grid.copy()
    right = left + span_width - 1
    for row in range(1, grid.shape[0]):
        lo = left + row
        hi = right - row
        if lo > hi:
            break
        for col in range(lo, hi + 1):
            source = int(grid[0, col - row])
            expected[row, col] = source if row % 2 == 0 else Color.GREY if source == Color.RED else Color.RED
    return expected


def _gen_attempt() -> np.ndarray | None:
    span_width = int(np.random.choice(SPAN_WIDTHS))
    triangle_height = (span_width - 1) // 2 + 1
    height = triangle_height + int(np.random.randint(*EXTRA_HEIGHT_RANGE))
    width = span_width + int(np.random.randint(*EXTRA_WIDTH_RANGE))
    left = int(np.random.randint(0, width - span_width + 1))
    grid = np.full((height, width), GEN_BG, dtype=int)
    if span_width == WIDTH_WITH_GAP:
        grid[0, left] = int(np.random.choice(GEN_SIGNAL_COLORS))
        grid[0, left + 2] = int(np.random.choice(GEN_SIGNAL_COLORS))
    else:
        grid[0, left : left + span_width] = np.random.choice(GEN_SIGNAL_COLORS, size=span_width)
    expected = _expected_output(grid, left, span_width)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b5bb5719 input")
