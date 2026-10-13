import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize vertical two-color bars between paired marker columns around the marker rows.

    Input structure
    - The background is the dominant color.
    - A row of red marker pixels sits directly above a row of gray marker pixels
      in the same alternating columns.
    - The columns between adjacent markers contain vertical bars made from a
      contiguous teal run above a contiguous blue run.

    Rule
    - Keep the marker pixels and all non-bar background cells fixed.
    - Let h be the maximum teal-run height among the bar columns; paint teal in
      every bar column for h rows ending at the red marker row.
    - In each bar column, move its blue run to start at the gray marker row and
      fill all lower cells in that column with maroon.

    Key insight
    - The marker rows are a baseline: teal is normalized into a common upper
      band, while each blue run keeps its own height below the baseline.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    H = g.shape[0]

    blue = 1
    red = 2
    gray = 5
    teal = 8
    maroon = 9

    red_coords = np.argwhere(g == red)
    gray_coords = np.argwhere(g == gray)
    assert red_coords.size > 0 and gray_coords.size > 0

    red_rows = np.unique(red_coords[:, 0])
    gray_rows = np.unique(gray_coords[:, 0])
    assert len(red_rows) == 1 and len(gray_rows) == 1

    red_row = int(red_rows[0])
    gray_row = int(gray_rows[0])
    assert gray_row == red_row + 1

    marker_cols = np.where(g[red_row] == red)[0]
    gray_cols = np.where(g[gray_row] == gray)[0]
    assert marker_cols.size >= 2
    assert np.array_equal(marker_cols, gray_cols)
    assert np.all(np.diff(marker_cols) == 2)

    bar_cols = [int((left + right) // 2) for left, right in zip(marker_cols, marker_cols[1:])]
    assert bar_cols

    max_teal_height = 0
    blue_heights: list[int] = []

    for c in bar_cols:
        col = g[:, c]
        non_bg_rows = np.where(col != bg)[0]
        assert non_bg_rows.size > 0

        first = int(non_bg_rows[0])
        assert np.array_equal(non_bg_rows, np.arange(first, H))
        assert np.all(np.isin(col[first:], [blue, teal]))

        segment = col[first:]
        blue_positions = np.where(segment == blue)[0]
        teal_positions = np.where(segment == teal)[0]
        assert blue_positions.size > 0 and teal_positions.size > 0
        assert np.array_equal(teal_positions, np.arange(teal_positions.size))
        assert np.array_equal(
            blue_positions,
            np.arange(teal_positions.size, teal_positions.size + blue_positions.size),
        )

        max_teal_height = max(max_teal_height, int(teal_positions.size))
        blue_heights.append(int(blue_positions.size))

    top_row = red_row - max_teal_height + 1
    assert top_row >= 0

    out = g.copy()
    for c, blue_height in zip(bar_cols, blue_heights):
        assert gray_row + blue_height <= H
        out[:, c] = bg
        out[top_row : red_row + 1, c] = teal
        out[gray_row : gray_row + blue_height, c] = blue
        out[gray_row + blue_height :, c] = maroon

    return out

# --------------- generator (att2, o5x) ---------------

_RNG = np.random.default_rng()

# Colors.
BG, BLUE, RED, GRAY, TEAL, MAROON = 7, 1, 2, 5, 8, 9
# Canvas and marker layout.
MIN_MARKERS, MAX_MARKERS_EXCL = 5, 9
MARKER_STEP, MARKER_ROWS = 2, 2
MIN_RED_ROW, MAX_RED_ROW = 2, 6
# Bar height bounds.
MIN_MAX_TEAL, MAX_MAX_TEAL_EXCL = 2, 5
MIN_BLUE_HEIGHT, MIN_REQUIRED_BLUE_HEIGHT = 1, 2
MIN_ROWS_BELOW_GRAY, MIN_INPUT_TEAL_HEIGHT = 2, 2
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _sample_layout():
    """Sample marker geometry and the output-side bar heights."""
    n_markers = int(_RNG.integers(MIN_MARKERS, MAX_MARKERS_EXCL))
    size = MARKER_STEP * n_markers - 1
    marker_cols = np.arange(0, size, MARKER_STEP)
    bar_cols = np.arange(1, size, MARKER_STEP)

    max_teal = int(_RNG.integers(MIN_MAX_TEAL, MAX_MAX_TEAL_EXCL))
    red_min = max(MIN_RED_ROW, max_teal - 1)
    red_max = min(MAX_RED_ROW, size - MIN_ROWS_BELOW_GRAY - MARKER_ROWS)
    if red_min > red_max: return None
    red_row = int(_RNG.integers(red_min, red_max + 1))

    max_blue = size - red_row - MARKER_ROWS
    if max_blue < MIN_REQUIRED_BLUE_HEIGHT: return None
    blue_heights = _RNG.integers(MIN_BLUE_HEIGHT, max_blue + 1, len(bar_cols)).astype(int)
    if np.max(blue_heights) < MIN_REQUIRED_BLUE_HEIGHT:
        blue_heights[int(_RNG.integers(0, len(bar_cols)))] = int(
            _RNG.integers(MIN_REQUIRED_BLUE_HEIGHT, max_blue + 1)
        )
    return size, marker_cols, bar_cols, red_row, red_row + 1, max_teal, blue_heights


def _build_grids(layout):
    """Render the target grid, then bottom-align bars to derive the input."""
    size, marker_cols, bar_cols, red_row, gray_row, max_teal, blue_heights = layout

    out = np.full((size, size), BG, dtype=int)
    out[red_row, marker_cols] = RED
    out[gray_row, marker_cols] = GRAY
    teal_top = red_row - max_teal + 1
    for c, blue_h in zip(bar_cols, blue_heights):
        out[teal_top : red_row + 1, c] = TEAL
        out[gray_row : gray_row + blue_h, c] = BLUE
        out[gray_row + blue_h :, c] = MAROON

    inp = out.copy()
    min_teal = max(MIN_INPUT_TEAL_HEIGHT, max_teal - 1)
    teal_heights = _RNG.integers(min_teal, max_teal + 1, len(bar_cols)).astype(int)
    teal_heights[int(_RNG.integers(0, len(bar_cols)))] = max_teal

    for c, teal_h, blue_h in zip(bar_cols, teal_heights, blue_heights):
        inp[:, c] = BG
        start = size - teal_h - blue_h
        if start < 0: return None
        inp[start : start + teal_h, c] = TEAL
        inp[start + teal_h :, c] = BLUE

    inp[red_row, marker_cols] = RED
    inp[gray_row, marker_cols] = GRAY
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None: return None

    pair = _build_grids(layout)
    if pair is None: return None
    inp, out = pair
    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 41ace6b5 input")
