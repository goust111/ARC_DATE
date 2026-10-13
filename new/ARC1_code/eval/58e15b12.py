from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _vertical_bar_pair(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == int(color))
    assert coords.size > 0, "expected a visible bar color"

    cols = sorted(set(map(int, coords[:, 1])))
    assert len(cols) == 2, "each color must form exactly two vertical bars"

    bars: list[tuple[int, int, int]] = []
    for col in cols:
        rows = sorted(map(int, coords[coords[:, 1] == col, 0]))
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be contiguous"
        bars.append((rows[0], rows[-1], col))

    left, right = bars
    assert left[0] == right[0] and left[1] == right[1], "paired bars must have the same height"
    return left[0], left[1], left[2], right[2]


def _outward_offset(row: int, r0: int, r1: int, bar_height: int) -> int:
    if row < r0:
        return ((r0 - 1 - row) // bar_height) + 1
    if row > r1:
        return ((row - r1 - 1) // bar_height) + 1
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each pair of same-color vertical bars into outward-stepping bands and mark cross-color overlaps pink.

    Input structure:
    - The background is black.
    - Each foreground color appears as exactly two same-height vertical bars.
    - The two bars of a color share one row interval and have distinct columns.

    Rule:
    - For each row of the grid, keep a bar in its original column while the row
      is inside the original bar interval.
    - Above or below that interval, shift the left bar one column left and the
      right bar one column right after every full bar-height block of rows.
    - Paint all in-bounds emitted cells; cells emitted by two or more different
      colors become pink.

    Key insight:
    - The short parallel bars encode the period of a pair of rays: bar height
      gives the vertical step length, and each step moves away from the pair's
      center by one column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    source_colors = colors_present(g, exclude={bg})
    assert source_colors, "expected foreground bars"
    assert Color.PINK not in source_colors, "pink is reserved for overlap cells"

    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color in source_colors:
        r0, r1, left_col, right_col = _vertical_bar_pair(g, color)
        assert left_col < right_col, "bar columns must be ordered left to right"
        bar_height = r1 - r0 + 1
        assert bar_height >= 1, "bar height must be positive"

        for row in range(height):
            offset = _outward_offset(row, r0, r1, bar_height)
            for col, direction in ((left_col, -1), (right_col, 1)):
                stepped_col = col + direction * offset
                if 0 <= stepped_col < width:
                    claims[(row, stepped_col)].add(int(color))

    out = np.full_like(g, int(bg))
    for (row, col), colors in claims.items():
        out[row, col] = next(iter(colors)) if len(colors) == 1 else Color.PINK
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
GRID_HEIGHT_RANGE = (22, 24)
GRID_WIDTH_RANGE = (17, 20)

# Bar geometry.
GEN_COLORS = (Color.GREEN, Color.TEAL)
BAR_HEIGHT_RANGE = (1, 5)
COLUMN_GAP_RANGE = (2, 5)

# Placement and ray constraints.
EDGE_COLUMN_MARGIN = 1
MIN_INTER_PAIR_COLUMN_SEPARATION = 2
MIN_SPACIOUS_SIDE_BLOCKS = 3
TOTAL_STEPS_RANGE = (4, 15)

# Oracle and retry limits.
MIN_OUTPUT_CELLS = 20
MAX_PINK_CELLS = 7
GENERATION_ATTEMPTS = 300


def _gen_column_options(width: int, gap: int) -> list[tuple[int, int]]:
    return [(left, left + gap) for left in range(EDGE_COLUMN_MARGIN, width - gap - EDGE_COLUMN_MARGIN)]


def _gen_row_starts(
    height: int, width: int, bar_height: int, left_col: int, right_col: int
) -> list[int]:
    starts = []
    min_steps, max_steps = TOTAL_STEPS_RANGE
    max_offset = max(left_col, width - 1 - right_col)
    for r0 in range(height - bar_height + 1):
        above = r0
        below = height - r0 - bar_height
        if above < MIN_SPACIOUS_SIDE_BLOCKS * bar_height and below < MIN_SPACIOUS_SIDE_BLOCKS * bar_height:
            continue
        above_steps = min((above + bar_height - 1) // bar_height, max_offset)
        below_steps = min((below + bar_height - 1) // bar_height, max_offset)
        if min_steps <= above_steps + below_steps <= max_steps:
            starts.append(r0)
    return starts


def _gen_layout(height: int, width: int, specs: list[tuple[int, int, int]]):
    """Choose compatible columns and row starts for the two bar pairs."""
    col_options = [_gen_column_options(width, gap) for _, _, gap in specs]
    col_pairs = [
        (green_cols, teal_cols)
        for green_cols in col_options[0]
        for teal_cols in col_options[1]
        if min(abs(a - b) for a in green_cols for b in teal_cols) >= MIN_INTER_PAIR_COLUMN_SEPARATION
    ]
    if not col_pairs:
        return None

    chosen_cols = random.choice(col_pairs)
    row_options = [
        _gen_row_starts(height, width, specs[i][1], chosen_cols[i][0], chosen_cols[i][1])
        for i in range(2)
    ]

    row_pairs = []
    for green_r0 in row_options[0]:
        green_r1 = green_r0 + specs[0][1] - 1
        for teal_r0 in row_options[1]:
            teal_r1 = teal_r0 + specs[1][1] - 1
            if green_r1 < teal_r0 or teal_r1 < green_r0:
                row_pairs.append((green_r0, teal_r0))
    if not row_pairs:
        return None

    return chosen_cols, random.choice(row_pairs)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    specs = [
        (int(color), random.randint(*BAR_HEIGHT_RANGE), random.randint(*COLUMN_GAP_RANGE))
        for color in GEN_COLORS
    ]

    layout = _gen_layout(height, width, specs)
    if layout is None:
        return None
    chosen_cols, chosen_rows = layout

    grid = np.zeros((height, width), dtype=int)
    for i, (color, bar_height, _) in enumerate(specs):
        r0 = chosen_rows[i]
        left_col, right_col = chosen_cols[i]
        grid[r0 : r0 + bar_height, left_col] = color
        grid[r0 : r0 + bar_height, right_col] = color

    out = main(grid)
    if int(np.count_nonzero(out)) < MIN_OUTPUT_CELLS:
        return None
    if int(np.count_nonzero(out == Color.PINK)) > MAX_PINK_CELLS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58e15b12 input")
