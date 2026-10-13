from collections import Counter

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw vertical bars for the colors whose bottom code-row frequency equals the marker count.

    Input structure
    The grid has a dominant background, one solid full-width separator row, a
    single contiguous marker run above the separator, and one non-background
    code row below the separator.

    Rule
    Count the marker pixels above the separator. In the code row, find every
    color that appears exactly that many times. For each occurrence of such a
    color, fill that column for the same number of cells immediately above the
    separator, using the code-row color.

    Key insight
    The marker run is the histogram target count, and the code row gives both
    the colors to test and the columns where matching bars must be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height = int(g.shape[0])
    bg = get_bg(g)

    separator_rows = [
        r
        for r in range(height)
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    assert len(separator_rows) == 1, "expected one solid non-background separator row"
    separator_row = separator_rows[0]

    marker_mask = g[:separator_row, :] != bg
    assert marker_mask.any(), "expected marker pixels above the separator"
    marker_rows = np.flatnonzero(marker_mask.any(axis=1))
    assert len(marker_rows) == 1, "marker pixels must occupy one row"
    marker_row = int(marker_rows[0])
    marker_cols = np.flatnonzero(marker_mask[marker_row])
    assert len(marker_cols) == int(marker_mask.sum()), "expected no other foreground above the separator"
    assert len(set(map(int, g[marker_row, marker_cols]))) == 1, "marker run must be monochromatic"
    assert np.all(np.diff(marker_cols) == 1), "marker pixels must form a contiguous run"
    marker_count = len(marker_cols)

    code_rows = [r for r in range(separator_row + 1, height) if np.all(g[r, :] != bg)]
    assert len(code_rows) == 1, "expected one non-background code row below the separator"
    code = g[code_rows[0], :]

    start_row = separator_row - marker_count
    assert start_row >= 0, "marker count must fit above the separator"

    code_counts = Counter(map(int, code))
    selected_cols = [c for c, color in enumerate(map(int, code)) if code_counts[color] == marker_count]
    assert selected_cols, "expected at least one code color with marker-count frequency"

    out = g.copy()
    for col in selected_cols:
        color = int(code[col])
        assert np.all(out[start_row:separator_row, col] == bg), "bar target cells must be background"
        out[start_row:separator_row, col] = color

    return out

# --------------- generator (o5x) ---------------
from functools import lru_cache
from itertools import product

# Grid layout.
GRID_SHAPE = (10, 10)
MARKER_ROW, SEPARATOR_ROW, CODE_ROW = 0, 6, 8
BACKGROUND_COLOR, SEPARATOR_COLOR, MARKER_COLOR = 0, 5, 8
CODE_WIDTH = GRID_SHAPE[1]

# Sampling bounds and retry policy.
MARKER_LEN_RANGE, COLOR_COUNT_RANGE, COLOR_VALUE_RANGE = (1, 5), (2, 6), (1, 10)
MAX_MATCHING_COLORS, MAX_SELECTED_COLUMNS, RETRY_LIMIT = 3, 6, 300
CODE_PALETTE = tuple(c for c in range(*COLOR_VALUE_RANGE) if c != SEPARATOR_COLOR)

@lru_cache(maxsize=None)
def _gen_count_options(marker_len: int, color_count: int, matching_count: int) -> list[tuple[int, ...]]:
    """Return ordered non-matching color counts that fill the code row."""
    other_count = color_count - matching_count
    other_total = CODE_WIDTH - marker_len * matching_count
    return [
        counts
        for counts in product(range(1, other_total + 1), repeat=other_count)
        if sum(counts) == other_total and marker_len not in counts
    ]


def _count_plans(marker_len: int) -> list[tuple[int, int, list[tuple[int, ...]]]]:
    """Return feasible color-count plans with cached non-matching counts."""
    plans = []
    for color_count in range(*COLOR_COUNT_RANGE):
        max_matching = min(MAX_MATCHING_COLORS, color_count, MAX_SELECTED_COLUMNS // marker_len)
        for matching_count in range(1, max_matching + 1):
            options = _gen_count_options(marker_len, color_count, matching_count)
            if options:
                plans.append((color_count, matching_count, options))
    return plans


def _sample_code_values(marker_len, color_count, matching_count, options):
    """Sample a shuffled code row and the colors expected to trigger bars."""
    other_counts = list(options[int(np.random.randint(len(options)))])
    colors = list(np.random.choice(CODE_PALETTE, size=color_count, replace=False))
    matching_colors = colors[:matching_count]
    counts = [marker_len] * matching_count + other_counts
    code_values = [int(color) for color, count in zip(colors, counts) for _ in range(int(count))]
    np.random.shuffle(code_values)
    return code_values, matching_colors


def _render_case(marker_len, code_values, matching_colors):
    """Build the input grid and its oracle output for the sampled code row."""
    code_row = np.array(code_values, dtype=int)
    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    grid[MARKER_ROW, :marker_len] = MARKER_COLOR
    grid[SEPARATOR_ROW, :] = SEPARATOR_COLOR
    grid[CODE_ROW, :] = code_row
    expected = grid.copy()
    matching_mask = np.isin(code_row, matching_colors)
    expected[SEPARATOR_ROW - marker_len : SEPARATOR_ROW, matching_mask] = code_row[matching_mask]
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    marker_len = int(np.random.randint(*MARKER_LEN_RANGE))

    plans = _count_plans(marker_len)
    if not plans:
        return None

    color_count, matching_count, options = plans[int(np.random.randint(len(plans)))]
    code_values, matching_colors = _sample_code_values(marker_len, color_count, matching_count, options)
    grid, expected = _render_case(marker_len, code_values, matching_colors)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
