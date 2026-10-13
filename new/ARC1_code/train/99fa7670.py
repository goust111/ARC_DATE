from __future__ import annotations

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored marker rightward to the grid edge, then downward on the right edge until the next marker row.

    Input structure
    - The grid is a non-empty rectangle with a dominant background color.
    - Every foreground cell is a single colored marker pixel.
    - Each active row contains exactly one marker.

    Rule
    - Sort the marker pixels from top to bottom.
    - For each marker, paint its row from the marker column through the right edge with the marker color.
    - Then paint the rightmost column from that marker row down to one row before the next lower marker, or to the bottom for the last marker.

    Key insight
    - The markers are row-start controls for right-turning paths, and the next marker below is the stopping boundary for the vertical edge segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H > 0 and W > 0, "expected a non-empty grid"

    bg = int(get_bg(g))
    marker_coords = np.argwhere(g != bg)
    assert len(marker_coords) > 0, "expected at least one marker pixel"

    markers = [(int(r), int(c), int(g[r, c])) for r, c in marker_coords]
    marker_rows = [r for r, _c, _color in markers]
    assert len(marker_rows) == len(set(marker_rows)), "expected exactly one marker per active row"
    markers.sort(key=lambda marker: marker[0])

    out = g.copy()
    for i, (r, c, color) in enumerate(markers):
        end_r = markers[i + 1][0] if i + 1 < len(markers) else H
        assert r < end_r <= H, "marker rows must be strictly increasing"

        out[r, c:W] = color
        out[r:end_r, W - 1] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry budget.
MIN_HEIGHT, MAX_HEIGHT, MIN_WIDTH, MAX_WIDTH = 3, 8, 3, 8
BACKGROUND_COLOR = 0
COLOR_POOL = np.arange(1, 10)
MAX_GENERATION_ATTEMPTS = 300

# Marker rows and L-path geometry.
MIN_MARKERS, MAX_MARKERS = 1, 3
ROW_GAPS, LAST_VERTICAL_LENGTHS = (2, 3), (2, 3)
MIN_HORIZONTAL_LEN, MAX_HORIZONTAL_LEN, MAX_L_PATH_CELLS = 2, 6, 7


def _gen_row_options(height: int, count: int) -> list[list[int]]:
    """Return all top-to-bottom marker row patterns for one height/count pair."""
    gap_patterns: list[list[int]] = [[]]
    for _ in range(count - 1):
        gap_patterns = [pattern + [gap] for pattern in gap_patterns for gap in ROW_GAPS]

    options: list[list[int]] = []
    for last_vertical_len in LAST_VERTICAL_LENGTHS:
        last_row = height - last_vertical_len
        if last_row < 0:
            continue
        for gaps in gap_patterns:
            rows = [last_row]
            for gap in reversed(gaps):
                rows.append(rows[-1] - gap)
            rows = list(reversed(rows))
            if rows[0] >= 0:
                options.append(rows)
    return options

def _choose_rows(height: int) -> list[int] | None:
    """Sample a feasible marker count, then one row pattern for it."""
    options_by_count = {count: _gen_row_options(height, count) for count in range(MIN_MARKERS, MAX_MARKERS + 1)}
    feasible_counts = [count for count, options in options_by_count.items() if options]
    if not feasible_counts:
        return None
    row_options = options_by_count[int(np.random.choice(feasible_counts))]
    return row_options[int(np.random.randint(len(row_options)))]

def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(MIN_HEIGHT, MAX_HEIGHT + 1))
    width = int(np.random.randint(MIN_WIDTH, MAX_WIDTH + 1))
    rows = _choose_rows(height)
    if rows is None:
        return None

    count = len(rows)
    colors = np.random.choice(COLOR_POOL, size=count, replace=False)

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    expected = np.full_like(grid, BACKGROUND_COLOR)

    for i, (row, color) in enumerate(zip(rows, colors)):
        next_row = rows[i + 1] if i + 1 < count else height
        vertical_len = next_row - row
        max_horizontal_len = min(width - 1, MAX_HORIZONTAL_LEN, MAX_L_PATH_CELLS + 1 - vertical_len)
        if max_horizontal_len < MIN_HORIZONTAL_LEN:
            return None

        horizontal_len = int(np.random.randint(MIN_HORIZONTAL_LEN, max_horizontal_len + 1))
        col = width - horizontal_len
        if not (1 <= col <= width - 2):
            return None

        color = int(color)
        grid[row, col] = color
        expected[row, col:width] = color
        expected[row:next_row, width - 1] = color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 99fa7670 input")
