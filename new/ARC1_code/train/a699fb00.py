import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the one-cell black gaps inside dashed horizontal blue bars with red.

    Input structure
    - The grid is a black canvas containing only blue foreground pixels.
    - Every blue pixel belongs to a horizontal dashed bar whose blue pixels are separated by one black gap.

    Rule
    - Preserve all input pixels.
    - Recolor each black pixel to red exactly when its immediate left and right neighbors are blue.

    Key insight
    - The output completes each dashed blue row segment locally; the only changed cells are horizontal 1-0-1 gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width >= 3, "expected room for horizontal gaps"
    assert get_bg(g) == Color.BLACK, "expected black background"
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, Color.BLUE}), "expected only black and blue in the input"

    blue_coords = np.argwhere(g == Color.BLUE)
    assert blue_coords.size > 0, "expected blue foreground pixels"
    for r, c in blue_coords:
        r = int(r)
        c = int(c)
        has_left_mate = c >= 2 and g[r, c - 2] == Color.BLUE
        has_right_mate = c + 2 < width and g[r, c + 2] == Color.BLUE
        assert has_left_mate or has_right_mate, "every blue pixel must be part of a dashed horizontal bar"

    fill_mask = np.zeros((height, width), dtype=bool)
    fill_mask[:, 1:-1] = (
        (g[:, 1:-1] == Color.BLACK)
        & (g[:, :-2] == Color.BLUE)
        & (g[:, 2:] == Color.BLUE)
    )
    assert np.any(fill_mask), "expected at least one horizontal gap to fill"

    out = g.copy()
    out[fill_mask] = Color.RED
    return out

# --------------- generator (o5x) ---------------
# Canvas size and retry limit.
GRID_SIZE_MIN, GRID_SIZE_MAX, RETRY_LIMIT = 5, 10, 300
# Bar count and row layout.
BAR_COUNT_MIN, BAR_COUNT_MAX = 2, 5
ROW_OCCUPANCY_MIN, ROW_OCCUPANCY_MAX = 0.4, 0.5
ROW_STEP_MIN, ROW_STEP_MAX = 2, 4
# Dashed-bar geometry.
BLUE_COUNT_MIN, BLUE_COUNT_MAX, DASH_STEP = 2, 4, 2

def _gen_bar_count_for_height(height: int) -> int:
    counts = [
        n for n in range(BAR_COUNT_MIN, BAR_COUNT_MAX + 1)
        if ROW_OCCUPANCY_MIN <= n / height <= ROW_OCCUPANCY_MAX and height >= ROW_STEP_MIN * n - 1
    ]
    if not counts:
        counts = [
            n for n in range(BAR_COUNT_MIN, BAR_COUNT_MAX + 1)
            if height >= ROW_STEP_MIN * n - 1
        ]
    return random.choice(counts)

def _gen_bar_rows(height: int, count: int) -> list[int] | None:
    gaps = []
    total = 0
    for i in range(count - 1):
        remaining = count - 2 - i
        choices = [
            gap
            for gap in range(ROW_STEP_MIN, ROW_STEP_MAX + 1)
            if total + gap + ROW_STEP_MIN * remaining <= height - 1
        ]
        if not choices:
            return None
        gap = random.choice(choices)
        gaps.append(gap)
        total += gap

    start = random.randint(0, height - 1 - total)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return rows

def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bar_count = _gen_bar_count_for_height(height)
    rows = _gen_bar_rows(height, bar_count)
    if rows is None:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    for row in rows:
        max_blue_count = min(BLUE_COUNT_MAX, (width + 1) // DASH_STEP)
        blue_count = random.randint(BLUE_COUNT_MIN, max_blue_count)
        span = DASH_STEP * blue_count - 1
        start_col = random.randint(0, width - span)
        grid[row, start_col : start_col + span : DASH_STEP] = Color.BLUE
    fill_mask = np.zeros((height, width), dtype=bool)
    fill_mask[:, 1:-1] = (
        (grid[:, 1:-1] == Color.BLACK)
        & (grid[:, :-2] == Color.BLUE)
        & (grid[:, 2:] == Color.BLUE)
    )
    expected = grid.copy()
    expected[fill_mask] = Color.RED
    return grid if np.array_equal(main(grid), expected) else None

def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid a699fb00 input")
