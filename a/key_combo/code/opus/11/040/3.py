from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_HEIGHT_RANGE = (13, 19)
GRID_WIDTH_RANGE = (12, 19)
RECT_HEIGHT_RANGE = (3, 6)
RECT_WIDTH_RANGE = (6, 10)
MAX_GENERATION_ATTEMPTS = 600


def _guide_row(grid: np.ndarray) -> int:
    red = np.argwhere(grid == RED)
    assert red.size > 0, "expected a red guide bar"
    rows = np.unique(red[:, 0])
    assert rows.size == 1, "guide bar must occupy exactly one row"
    return int(rows[0])


def _rectangle_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    green = np.argwhere(grid == GREEN)
    assert green.size > 0, "expected a green incomplete rectangle"
    r0, c0 = green.min(axis=0)
    r1, c1 = green.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _signal_columns(grid: np.ndarray, guide_row: int) -> set[int]:
    signal: set[int] = set()
    for r_raw, c_raw in np.argwhere(grid == BLUE):
        r = int(r_raw)
        c = int(c_raw)
        if r < guide_row and grid[guide_row, c] == RED:
            signal.add(c)
    return signal


def _project_mask_transform(grid: np.ndarray, *, ignore_gaps: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    guide_row = _guide_row(g)
    r0, c0, r1, c1 = _rectangle_bbox(g)

    if ignore_gaps:
        signal = {int(c) for _r, c in np.argwhere(g == BLUE) if int(_r) < guide_row}
    else:
        signal = _signal_columns(g, guide_row)

    out = g.copy()
    for c in range(c0, c1 + 1):
        segment = out[r0 : r1 + 1, c].copy()
        if c in signal:
            segment[segment == BLACK] = GREEN
        else:
            segment[segment == GREEN] = BLACK
        out[r0 : r1 + 1, c] = segment
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the blue pixels onto the red guide bar and use the resulting column mask to complete or erase the green rectangle.

    Input structure:
    - The grid has black background, one horizontal red guide bar with a few one-cell gaps, blue single pixels above it, and one incomplete green rectangle below it.
    - Each green rectangle column inside its bounding box is either fully filled green across the rectangle rows or fully missing black.
    - The guide bar spans the whole width except for its black gap cells, and each blue source column contains at most one blue pixel.

    Rule:
    - For every blue pixel, project vertically onto the guide row in the same column.
    - If that guide cell is red, mark the column as having signal; if that guide cell is a black gap, the projection is discarded.
    - Find the green rectangle's bounding box from the existing green pixels.
    - Inside that box, fill every black cell of each signal column with green, and erase every green cell of each non-signal column to black.
    - Keep the red guide bar and the blue pixels unchanged.

    Key insight:
    - The red guide bar is not the final target: its gaps filter the blue projections into a binary column mask, and that mask alone decides which green rectangle columns survive or get reconstructed.
    """
    return _project_mask_transform(np.asarray(grid, dtype=int), ignore_gaps=False)


def _full_green_columns(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> set[int]:
    cols: set[int] = set()
    for c in range(c0, c1 + 1):
        segment = grid[r0 : r1 + 1, c]
        if np.all(segment == GREEN):
            cols.add(c)
        else:
            assert np.all(segment == BLACK), "rectangle columns must be fully present or fully absent"
    return cols


def _choose_column_roles(rect_cols: list[int], gap_cols_rect: set[int]) -> tuple[set[int], set[int], set[int]]:
    left = rect_cols[0]
    right = rect_cols[-1]
    non_gap_rect = [c for c in rect_cols if c not in gap_cols_rect]
    assert len(non_gap_rect) >= 4

    for _ in range(300):
        signal_size = random.randint(2, len(non_gap_rect) - 1)
        signal_cols = set(random.sample(non_gap_rect, signal_size))

        interior = rect_cols[1:-1]
        extra_existing_count = random.randint(1, max(1, len(interior) - 1))
        existing_cols = {left, right}
        existing_cols.update(random.sample(interior, extra_existing_count))

        shared = existing_cols & signal_cols
        fill = signal_cols - existing_cols
        gap_erase = (existing_cols - signal_cols) & gap_cols_rect
        if not shared or not fill or not gap_erase:
            continue

        gap_blue_size = random.randint(1, len(gap_cols_rect))
        gap_blue_cols = set(random.sample(sorted(gap_cols_rect), gap_blue_size))
        if not (gap_blue_cols & gap_erase):
            gap_blue_cols.add(random.choice(sorted(gap_erase)))

        no_blue_erase = existing_cols - signal_cols - gap_blue_cols
        if not no_blue_erase:
            continue

        return existing_cols, signal_cols, gap_blue_cols

    raise RuntimeError("failed to choose non-degenerate rectangle column roles")


def _random_blue_rows(guide_row: int, columns: list[int]) -> dict[int, int]:
    assert guide_row >= 3
    rows = {}
    for c in columns:
        rows[c] = random.randint(0, guide_row - 2)
    return rows


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    rect_h_max = min(RECT_HEIGHT_RANGE[1], height - 8)
    if rect_h_max < RECT_HEIGHT_RANGE[0]:
        return None
    rect_h = random.randint(RECT_HEIGHT_RANGE[0], rect_h_max)

    guide_row_max = height - rect_h - 5
    if guide_row_max < 3:
        return None
    guide_row = random.randint(3, guide_row_max)
    rect_top = random.randint(guide_row + 3, height - rect_h - 1)
    rect_bottom = rect_top + rect_h - 1

    rect_w_max = min(RECT_WIDTH_RANGE[1], width - 2)
    if rect_w_max < RECT_WIDTH_RANGE[0]:
        return None
    rect_w = random.randint(RECT_WIDTH_RANGE[0], rect_w_max)
    rect_left = random.randint(1, width - rect_w - 1)
    rect_right = rect_left + rect_w - 1
    rect_cols = list(range(rect_left, rect_right + 1))

    overlap_gap_max = min(2, rect_w - 4)
    if overlap_gap_max < 1:
        return None
    overlap_gap_count = random.randint(1, overlap_gap_max)
    gap_cols_rect = set(random.sample(rect_cols, overlap_gap_count))

    outside_cols = [c for c in range(width) if c not in rect_cols]
    extra_gap_max = min(2, len(outside_cols))
    extra_gap_count = random.randint(0, extra_gap_max)
    gap_cols_outside = set(random.sample(outside_cols, extra_gap_count))
    gap_cols = gap_cols_rect | gap_cols_outside

    try:
        existing_cols, signal_cols, gap_blue_cols = _choose_column_roles(rect_cols, gap_cols_rect)
    except RuntimeError:
        return None

    valid_outside_candidates = [c for c in outside_cols if c not in gap_cols]
    gap_outside_candidates = [c for c in outside_cols if c in gap_cols]
    extra_valid = set(random.sample(valid_outside_candidates, random.randint(1, min(2, len(valid_outside_candidates))))) if valid_outside_candidates else set()
    extra_gap_blue = set(random.sample(gap_outside_candidates, random.randint(0, min(1, len(gap_outside_candidates))))) if gap_outside_candidates else set()

    blue_cols = sorted(signal_cols | gap_blue_cols | extra_valid | extra_gap_blue)
    blue_rows = _random_blue_rows(guide_row, blue_cols)

    grid = np.zeros((height, width), dtype=int)
    grid[guide_row, :] = RED
    for c in gap_cols:
        grid[guide_row, c] = BLACK

    for c in existing_cols:
        grid[rect_top : rect_bottom + 1, c] = GREEN

    for c, r in blue_rows.items():
        grid[r, c] = BLUE

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    guide_row = _guide_row(grid)
    r0, c0, r1, c1 = _rectangle_bbox(grid)
    rect_cols = set(range(c0, c1 + 1))
    gap_cols = {int(c) for c in np.where(grid[guide_row] == BLACK)[0]}
    blue_cols = {int(c) for r, c in np.argwhere(grid == BLUE) if int(r) < guide_row}
    signal_cols = _signal_columns(grid, guide_row)
    existing_cols = _full_green_columns(grid, r0, r1, c0, c1)

    assert guide_row > 1
    assert r0 > guide_row + 1
    assert rect_cols & gap_cols
    assert len(blue_cols) == len({int(c) for _r, c in np.argwhere(grid == BLUE)})
    assert all(int(r) < guide_row for r, _c in np.argwhere(grid == BLUE))
    assert existing_cols == {c for c in rect_cols if np.all(grid[r0 : r1 + 1, c] == GREEN)}

    fill_cols = signal_cols & rect_cols - existing_cols
    shared_cols = signal_cols & existing_cols
    erased_cols = existing_cols - signal_cols
    gap_blue_cols = (blue_cols & gap_cols) & rect_cols
    gap_erased_cols = erased_cols & gap_blue_cols
    no_blue_erased_cols = erased_cols - blue_cols

    assert fill_cols
    assert shared_cols
    assert erased_cols
    assert gap_erased_cols
    assert no_blue_erased_cols

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    assert np.array_equal(out[:r0], grid[:r0])
    assert np.array_equal(out[r1 + 1 :], grid[r1 + 1 :])

    out_cols = _full_green_columns(out, r0, r1, c0, c1)
    assert out_cols == (signal_cols & rect_cols)

    no_gap_output = _project_mask_transform(grid, ignore_gaps=True)
    assert not np.array_equal(no_gap_output, out), "guide gaps must matter"

    completed_all = grid.copy()
    completed_all[r0 : r1 + 1, c0 : c1 + 1] = GREEN
    assert not np.array_equal(completed_all, out), "cannot collapse to plain rectangle completion"

    erased_all = grid.copy()
    erased_all[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    assert not np.array_equal(erased_all, out), "cannot collapse to plain rectangle erasure"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle after many attempts")
