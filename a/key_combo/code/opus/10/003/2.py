from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ACTIVE_COLORS = (1, 2, 3, 4)
COUNTER_COLOR = 5

GRID_MIN_SIDE = 10
GRID_MAX_SIDE = 16
OFFSET_MIN = 1
OFFSET_MAX = 4
MARKED_ROWS_MIN = 3
MARKED_ROWS_MAX = 6
ACTIVE_ROWS_MIN = 1
ACTIVE_ROWS_MAX = 3
GENERATION_ATTEMPTS = 400


def _counter_offset(grid: np.ndarray) -> int:
    """Count the topmost consecutive grey pixels in the rightmost column."""
    height = int(grid.shape[0])
    width = int(grid.shape[1])
    offset = 0
    while offset < height and int(grid[offset, width - 1]) == COUNTER_COLOR:
        offset += 1
    return offset


def _row_markers(grid: np.ndarray, r: int) -> tuple[int, int]:
    return int(grid[r, 0]), int(grid[r, grid.shape[1] - 1])


def _is_marker_color(value: int) -> bool:
    return int(value) in ACTIVE_COLORS


def _analyze_rows(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[int]]:
    """Return offset, active rows as (row, color), and mismatch rows."""
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    offset = _counter_offset(g)
    assert np.all(g[:, 1 : width - 1] == BACKGROUND), "only boundary columns may contain markers"
    assert not np.any(g[offset:, width - 1] == COUNTER_COLOR), "counter must be a top-contiguous block"

    active_rows: list[tuple[int, int]] = []
    mismatch_rows: list[int] = []
    for r in range(height):
        left, right = _row_markers(g, r)
        if r < offset:
            assert left == BACKGROUND, "counter rows must not also contain left markers"
            assert right == COUNTER_COLOR
            continue
        left_is_marker = _is_marker_color(left)
        right_is_marker = _is_marker_color(right)
        assert left_is_marker == right_is_marker, "rows must have either two markers or none"
        if not left_is_marker:
            assert left == BACKGROUND and right == BACKGROUND
            continue
        if left == right:
            active_rows.append((r, left))
        else:
            mismatch_rows.append(r)

    assert 1 <= offset <= OFFSET_MAX
    assert active_rows
    assert mismatch_rows
    assert not np.any(g[:, :-1] == COUNTER_COLOR), "grey may only appear in the counter"
    assert np.all((g == BACKGROUND) | np.isin(g, ACTIVE_COLORS + (COUNTER_COLOR,)))
    return offset, active_rows, mismatch_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rows whose boundary markers match, then move those full-width bars downward by the grey counter offset.

    Input structure:
    - The grid is mostly black background.
    - The top of the rightmost column contains a consecutive grey counter of length 1-4.
    - Some lower rows have paired colored markers at column 0 and column W-1, using non-black, non-grey colors.

    Rule:
    - Count the consecutive grey pixels from the top of the rightmost column to get offset N.
    - A row is active iff both boundary markers are present and have the same color; active rows become full-width bars of that color.
    - Move every active full-width bar down by N rows, and clear all original markers and grey counter pixels to black.

    Key insight:
    - The same-color endpoint test decides which rows become bars, while the grey counter does not paint anything itself and only supplies the shared vertical shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert width >= 2

    offset = _counter_offset(g)
    out = np.zeros_like(g)
    for r in range(height):
        left, right = _row_markers(g, r)
        if _is_marker_color(left) and left == right:
            target_r = r + offset
            assert target_r < height, "translated active row must stay in bounds"
            out[target_r, :] = left
    return out


def _choose_active_rows(marked_rows: list[int]) -> list[int]:
    max_active = min(ACTIVE_ROWS_MAX, len(marked_rows) - 1)
    n_active = random.randint(ACTIVE_ROWS_MIN, max_active)
    return sorted(random.sample(marked_rows, n_active))


def _choose_active_colors(n_active: int) -> list[int]:
    if n_active == 1:
        return [random.choice(ACTIVE_COLORS)]
    n_distinct = random.randint(2, min(len(ACTIVE_COLORS), n_active))
    base = random.sample(ACTIVE_COLORS, n_distinct)
    while len(base) < n_active:
        base.append(random.choice(base))
    random.shuffle(base)
    return [int(color) for color in base]


def _valid_marked_rows(
    height: int, offset: int, total_marked: int
) -> tuple[list[int], list[int]] | None:
    candidate_rows = list(range(offset, height - offset))
    if len(candidate_rows) < total_marked:
        return None

    for _ in range(100):
        marked_rows = sorted(random.sample(candidate_rows, total_marked))
        active_rows = _choose_active_rows(marked_rows)
        shifted_rows = {r + offset for r in active_rows}

        if shifted_rows & set(marked_rows):
            continue
        if len(shifted_rows) != len(active_rows):
            continue
        return marked_rows, active_rows
    return None


def _build_input(height: int, width: int, offset: int) -> np.ndarray | None:
    total_marked = random.randint(MARKED_ROWS_MIN, MARKED_ROWS_MAX)
    chosen_rows = _valid_marked_rows(height, offset, total_marked)
    if chosen_rows is None:
        return None

    marked_rows, active_rows_list = chosen_rows
    active_rows = set(active_rows_list)
    if len(active_rows) == len(marked_rows):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:offset, width - 1] = COUNTER_COLOR

    active_colors = _choose_active_colors(len(active_rows))
    active_by_row = {row: color for row, color in zip(sorted(active_rows), active_colors)}
    active_palette = list(set(active_colors))

    for r in marked_rows:
        if r in active_by_row:
            color = int(active_by_row[r])
            grid[r, 0] = color
            grid[r, width - 1] = color
            continue

        # Force at least one mismatch row to share a color with an active row,
        # which blocks shortcuts based on looking at only one endpoint.
        if active_palette and random.random() < 0.7:
            shared = random.choice(active_palette)
            other = random.choice([c for c in ACTIVE_COLORS if c != shared])
            if random.random() < 0.5:
                left, right = shared, other
            else:
                left, right = other, shared
        else:
            left, right = random.sample(ACTIVE_COLORS, 2)
        grid[r, 0] = int(left)
        grid[r, width - 1] = int(right)

    if not _semantic_checks(grid):
        return None
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        height, width = map(int, g.shape)
        if not (g.ndim == 2 and max(g.shape) <= 30):
            return False

        offset, active_rows, mismatch_rows = _analyze_rows(g)
        if not (GRID_MIN_SIDE <= height <= GRID_MAX_SIDE):
            return False
        if not (GRID_MIN_SIDE <= width <= GRID_MAX_SIDE):
            return False

        marked_rows = {r for r, _ in active_rows} | set(mismatch_rows)
        shifted_rows = {r + offset for r, _ in active_rows}
        if shifted_rows & marked_rows:
            return False
        if len(active_rows) + len(mismatch_rows) < MARKED_ROWS_MIN:
            return False
        if len(active_rows) == 0 or len(mismatch_rows) == 0:
            return False

        # Avoid easy but wrong correlations such as "shift equals number of active rows".
        if offset == len(active_rows) or offset == len(marked_rows):
            return False

        out = main(g)
        if out.shape != g.shape:
            return False
        if np.any(out[:offset, :] != BACKGROUND):
            return False

        nonzero_rows = np.flatnonzero(np.any(out != BACKGROUND, axis=1)).tolist()
        expected_rows = sorted(shifted_rows)
        if nonzero_rows != expected_rows:
            return False

        expected_colors = {r + offset: color for r, color in active_rows}
        for r in range(height):
            row_vals = set(map(int, out[r]))
            if r in expected_colors:
                if row_vals != {expected_colors[r]}:
                    return False
            elif row_vals != {BACKGROUND}:
                return False

        if np.array_equal(out, g):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        offset = random.randint(OFFSET_MIN, OFFSET_MAX)
        min_height = max(GRID_MIN_SIDE, 2 * offset + MARKED_ROWS_MIN)
        height = random.randint(min_height, GRID_MAX_SIDE)
        width = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)

        grid = _build_input(height, width, offset)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
