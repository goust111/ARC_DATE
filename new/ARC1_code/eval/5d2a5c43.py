from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


def _separator_column(g: np.ndarray) -> tuple[int, int]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int]] = []
    for color in np.unique(g):
        coords = np.argwhere(g == int(color))
        cols = np.unique(coords[:, 1])
        rows = np.unique(coords[:, 0])
        if coords.shape[0] == height and rows.size == height and cols.size == 1:
            candidates.append((int(cols[0]), int(color)))

    assert len(candidates) == 1, "expected exactly one full-height separator-color mask"
    sep_col, sep_color = candidates[0]
    assert 0 < sep_col < width - 1, "separator must split two nonempty panels"
    assert sep_col * 2 + 1 == width, "separator must split equal-width panels"
    return sep_col, sep_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two yellow panels split by a blue divider into one teal mask.

    Input structure:
    - A single divider color appears only as one full-height vertical column.
    - The divider separates two equal-width black/yellow panels.
    - Black is empty space and yellow cells are the mask pixels to combine.

    Rule:
    - Remove the divider and align the left and right panels by row and column.
    - A compact output cell is teal if either aligned input panel has a yellow cell.
    - Otherwise the compact output cell remains black.

    Key insight:
    - The input is two same-sized binary masks shown side by side; the output is
      their union, recolored from yellow to teal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col, sep_color = _separator_column(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "panels must have matching shape"

    panel_colors = set(map(int, np.unique(np.concatenate([left, right]))))
    assert sep_color not in panel_colors, "separator color must not appear inside the panels"
    assert Color.BLACK in panel_colors, "panels must include black empty space"
    foreground_colors = panel_colors - {Color.BLACK}
    assert len(foreground_colors) == 1, "expected one panel foreground color"
    foreground = foreground_colors.pop()

    out = np.full(left.shape, Color.BLACK, dtype=int)
    out[(left == foreground) | (right == foreground)] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
HEIGHT_RANGE, PANEL_WIDTH_RANGE = (4, 8), (3, 6)

# Output and panel foreground density constraints.
UNION_PERCENT_RANGE = (70, 85)
MIN_PANEL_FILL_DIVISOR, MAX_PANEL_FILL_PERCENT = 4, 67
PERCENT_DENOMINATOR = 100

# Union-cell assignment and retry limits.
MIN_OVERLAP_COUNT, PREFERRED_OVERLAP_RANGE = 1, (2, 9)
MAX_GENERATION_ATTEMPTS = 300

# Render colors.
EMPTY_COLOR, OUTPUT_COLOR = Color.BLACK, Color.TEAL


def _gen_output_mask(height: int, width: int, teal_count: int) -> np.ndarray | None:
    mask = np.ones((height, width), dtype=bool)
    row_counts = np.full(height, width, dtype=int)
    col_counts = np.full(width, height, dtype=int)
    black_count = height * width - teal_count

    for _ in range(black_count):
        choices = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if mask[r, c] and row_counts[r] > 1 and col_counts[c] > 1
        ]
        if not choices:
            return None
        r, c = random.choice(choices)
        mask[r, c] = False
        row_counts[r] -= 1
        col_counts[c] -= 1

    return mask


def _gen_assignment_counts(area: int, union_count: int) -> tuple[int, int] | None:
    min_panel = (area + MIN_PANEL_FILL_DIVISOR - 1) // MIN_PANEL_FILL_DIVISOR
    max_panel = (MAX_PANEL_FILL_PERCENT * area) // PERCENT_DENOMINATOR
    feasible: list[tuple[int, int, int]] = []

    for both_count in range(MIN_OVERLAP_COUNT, union_count + 1):
        left_only_min = max(0, min_panel - both_count, union_count - max_panel)
        left_only_max = min(
            union_count - both_count,
            max_panel - both_count,
            union_count - min_panel,
        )
        if left_only_min <= left_only_max:
            feasible.append((both_count, left_only_min, left_only_max))

    preferred_min, preferred_max = PREFERRED_OVERLAP_RANGE
    preferred = [item for item in feasible if preferred_min <= item[0] <= preferred_max]
    pool = preferred or feasible
    if not pool:
        return None

    both_count, left_only_min, left_only_max = random.choice(pool)
    if left_only_max - left_only_min >= 2:
        left_only_min = max(left_only_min, 1)
        left_only_max = min(left_only_max, union_count - both_count - 1)
    if left_only_min > left_only_max:
        return None
    return both_count, random.randint(left_only_min, left_only_max)


def _build_grid(height, panel_width, separator_color, panel_fg_color, both, left_only, right_only):
    left = np.full((height, panel_width), EMPTY_COLOR, dtype=int)
    right = np.full((height, panel_width), EMPTY_COLOR, dtype=int)

    paint_plan = ((both, (left, right)), (left_only, (left,)), (right_only, (right,)))
    for cells, panels in paint_plan:
        for r, c in cells:
            for panel in panels:
                panel[r, c] = panel_fg_color

    grid = np.full((height, panel_width * 2 + 1), EMPTY_COLOR, dtype=int)
    grid[:, :panel_width] = left
    grid[:, panel_width] = separator_color
    grid[:, panel_width + 1 :] = right
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    area = height * panel_width

    min_union_percent, max_union_percent = UNION_PERCENT_RANGE
    min_teal = (min_union_percent * area + PERCENT_DENOMINATOR - 1) // PERCENT_DENOMINATOR
    max_teal = min(area - 1, (max_union_percent * area) // PERCENT_DENOMINATOR)
    if min_teal > max_teal:
        return None
    teal_count = random.randint(min_teal, max_teal)

    out_mask = _gen_output_mask(height, panel_width, teal_count)
    if out_mask is None:
        return None

    counts = _gen_assignment_counts(area, teal_count)
    if counts is None:
        return None
    both_count, left_only_count = counts

    coords = [tuple(map(int, rc)) for rc in np.argwhere(out_mask)]
    random.shuffle(coords)
    both = coords[:both_count]
    left_only = coords[both_count : both_count + left_only_count]
    right_only = coords[both_count + left_only_count :]

    colors = list(Color.NOT_BLACK)
    separator_color = random.choice(colors)
    panel_fg_color = random.choice([color for color in colors if color != separator_color])
    grid = _build_grid(height, panel_width, separator_color, panel_fg_color, both, left_only, right_only)

    expected = np.full((height, panel_width), EMPTY_COLOR, dtype=int)
    expected[out_mask] = OUTPUT_COLOR
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5d2a5c43 input")
