from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MARKER_DISTANCE = {RED: 3, GREEN: 4}
PATTERN_CYCLE = (
    ((RED,), (RED,)),
    ((GREEN,), (GREEN,)),
    ((RED, GREEN), (RED,)),
    ((GREEN,), (RED, GREEN)),
    ((RED, GREEN), (RED, GREEN)),
    ((RED, RED), (RED, GREEN)),
)

GRID_ROWS = (13, 18)
GRID_COLS = (13, 18)
FRAME_HEIGHT = (11, 14)
FRAME_WIDTH = (9, 14)
MAX_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _parse_input(grid: np.ndarray) -> tuple[int, int, int, int, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    allowed = {BLACK, BLUE, RED, GREEN, YELLOW}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors in grid"

    blue_pts = np.argwhere(g == BLUE)
    assert len(blue_pts) == 4, "expected exactly four blue control pixels"

    rows = blue_pts[:, 0]
    cols = blue_pts[:, 1]
    top_r = int(rows.min())
    bottom_r = int(rows.max())
    left_c = int(cols.min())
    right_c = int(cols.max())
    assert top_r < bottom_r and left_c < right_c, "control extrema must define a rectangle"
    assert int(np.sum(rows == top_r)) == 1, "top control must be unique"
    assert int(np.sum(rows == bottom_r)) == 1, "bottom control must be unique"
    assert int(np.sum(cols == left_c)) == 1, "left control must be unique"
    assert int(np.sum(cols == right_c)) == 1, "right control must be unique"

    interior_w = right_c - left_c - 1
    assert interior_w >= 2, "frame interior is too narrow"

    pivot_rows: list[int] = []
    pivot_markers: list[list[tuple[int, int]]] = []
    for r in range(top_r + 1, bottom_r):
        segment = g[r, left_c + 1 : right_c]
        if segment.size != interior_w:
            continue
        if not np.all(np.isin(segment, [YELLOW, RED, GREEN])):
            continue
        markers = [(int(c), int(g[r, c])) for c in range(left_c + 1, right_c) if int(g[r, c]) in MARKER_DISTANCE]
        if 2 <= len(markers) <= 4:
            pivot_rows.append(r)
            pivot_markers.append(markers)

    assert len(pivot_rows) == 1, "expected exactly one pivot row"
    pivot_r = int(pivot_rows[0])
    markers = pivot_markers[0]

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BLACK:
                continue
            if color == BLUE:
                assert (r == top_r and left_c < c < right_c) or (r == bottom_r and left_c < c < right_c) or (
                    c == left_c and top_r < r < bottom_r
                ) or (c == right_c and top_r < r < bottom_r), "blue controls must lie on the rectangle sides"
                continue
            assert left_c < c < right_c and r == pivot_r, "non-blue foreground must lie on the pivot row inside the frame"

    return top_r, bottom_r, left_c, right_c, pivot_r, markers


def _target_row(pivot_r: int, col: int, color: int, top_r: int, bottom_r: int, left_c: int, right_c: int) -> int:
    midpoint = (left_c + right_c) / 2.0
    direction = -1 if col < midpoint else 1
    raw_target = pivot_r + direction * MARKER_DISTANCE[color]
    return int(min(max(raw_target, top_r + 1), bottom_r - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the blue frame from four control pixels, then copy each pivot-row marker vertically away from the row by a color-coded distance and fill the stem in yellow.

    Input structure:
    - The background is black.
    - Exactly four blue control pixels occupy the unique top, bottom, left, and right foreground extrema and therefore define a rectangle.
    - Inside that rectangle there is exactly one full interior horizontal yellow row, with 2 to 4 red/green marker pixels replacing some yellow cells.

    Rule:
    - Use the blue control extrema to draw the full blue rectangular frame.
    - Treat the interior yellow row as the pivot row.
    - For each marker on that row, copy red 3 cells away or green 4 cells away in the same column.
    - Markers in the left half of the frame copy upward; markers in the right half copy downward.
    - Fill the intervening cells in that column with yellow, without overwriting the blue frame, and leave all other cells unchanged.

    Key insight:
    - The marker color sets the distance, but the horizontal side of the frame sets the direction: left-half markers go up and right-half markers go down.
    """
    g = np.asarray(grid, dtype=int)
    top_r, bottom_r, left_c, right_c, pivot_r, markers = _parse_input(g)

    out = g.copy()
    out[top_r, left_c : right_c + 1] = BLUE
    out[bottom_r, left_c : right_c + 1] = BLUE
    out[top_r + 1 : bottom_r, left_c] = BLUE
    out[top_r + 1 : bottom_r, right_c] = BLUE

    for col, color in markers:
        target_r = _target_row(pivot_r, col, color, top_r, bottom_r, left_c, right_c)
        step = -1 if target_r < pivot_r else 1
        for r in range(pivot_r + step, target_r, step):
            if out[r, col] != BLUE:
                out[r, col] = YELLOW
        out[target_r, col] = color

    return out


def _choose_spaced_columns(candidates: list[int], count: int) -> list[int] | None:
    if len(candidates) < count:
        return None
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[int] = []
    for col in shuffled:
        if all(abs(col - prev) >= 2 for prev in chosen):
            chosen.append(col)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _solve_with_rule(
    grid: np.ndarray,
    *,
    direction_mode: str,
) -> np.ndarray:
    top_r, bottom_r, left_c, right_c, pivot_r, markers = _parse_input(grid)
    out = np.asarray(grid, dtype=int).copy()
    out[top_r, left_c : right_c + 1] = BLUE
    out[bottom_r, left_c : right_c + 1] = BLUE
    out[top_r + 1 : bottom_r, left_c] = BLUE
    out[top_r + 1 : bottom_r, right_c] = BLUE

    midpoint = (left_c + right_c) / 2.0
    for col, color in markers:
        if direction_mode == "side":
            direction = -1 if col < midpoint else 1
        elif direction_mode == "all_up":
            direction = -1
        elif direction_mode == "all_down":
            direction = 1
        elif direction_mode == "reverse_side":
            direction = 1 if col < midpoint else -1
        elif direction_mode == "red_up_green_down":
            direction = -1 if color == RED else 1
        elif direction_mode == "red_down_green_up":
            direction = 1 if color == RED else -1
        else:
            raise ValueError(f"unknown direction mode: {direction_mode}")

        raw_target = pivot_r + direction * MARKER_DISTANCE[color]
        target_r = int(min(max(raw_target, top_r + 1), bottom_r - 1))
        step = -1 if target_r < pivot_r else 1
        for r in range(pivot_r + step, target_r, step):
            if out[r, col] != BLUE:
                out[r, col] = YELLOW
        out[target_r, col] = color

    return out


def _validate_example(grid: np.ndarray) -> bool:
    top_r, bottom_r, left_c, right_c, pivot_r, markers = _parse_input(grid)
    out = main(grid)

    assert out.shape == grid.shape
    assert np.all(out[top_r, left_c : right_c + 1] == BLUE)
    assert np.all(out[bottom_r, left_c : right_c + 1] == BLUE)
    assert np.all(out[top_r + 1 : bottom_r, left_c] == BLUE)
    assert np.all(out[top_r + 1 : bottom_r, right_c] == BLUE)

    marker_cols = {col for col, _ in markers}
    for c in range(left_c + 1, right_c):
        if c in marker_cols:
            continue
        col_slice = out[top_r + 1 : bottom_r, c].copy()
        col_slice[pivot_r - top_r - 1] = BLACK
        assert np.all(col_slice == BLACK), "non-marker columns must remain empty away from the pivot row"

    for col, color in markers:
        target_r = _target_row(pivot_r, col, color, top_r, bottom_r, left_c, right_c)
        assert int(out[target_r, col]) == color
        lo, hi = sorted((pivot_r, target_r))
        if hi - lo > 1:
            assert np.all(out[lo + 1 : hi, col] == YELLOW), "stem interior must be yellow"

    expected = _solve_with_rule(grid, direction_mode="side")
    assert np.array_equal(out, expected)

    shortcuts = [
        _solve_with_rule(grid, direction_mode="all_up"),
        _solve_with_rule(grid, direction_mode="all_down"),
        _solve_with_rule(grid, direction_mode="reverse_side"),
        _solve_with_rule(grid, direction_mode="red_up_green_down"),
        _solve_with_rule(grid, direction_mode="red_down_green_up"),
    ]
    assert all(not np.array_equal(out, candidate) for candidate in shortcuts), "degenerate shortcut example"
    return True


def _gen_attempt(pattern_idx: int) -> np.ndarray | None:
    left_colors = list(PATTERN_CYCLE[pattern_idx][0])
    right_colors = list(PATTERN_CYCLE[pattern_idx][1])

    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)

    frame_h = random.randint(*FRAME_HEIGHT)
    frame_w = random.randint(*FRAME_WIDTH)
    if frame_h > rows - 2 or frame_w > cols - 2:
        return None

    top_r = random.randint(1, rows - frame_h - 1)
    left_c = random.randint(1, cols - frame_w - 1)
    bottom_r = top_r + frame_h - 1
    right_c = left_c + frame_w - 1

    if bottom_r - top_r < 10:
        return None

    pivot_low = top_r + 5
    pivot_high = bottom_r - 5
    if pivot_low > pivot_high:
        return None
    pivot_r = random.randint(pivot_low, pivot_high)

    midpoint = (left_c + right_c) / 2.0
    left_candidates = [c for c in range(left_c + 1, right_c) if c < midpoint]
    right_candidates = [c for c in range(left_c + 1, right_c) if c >= midpoint]
    left_cols = _choose_spaced_columns(left_candidates, len(left_colors))
    right_cols = _choose_spaced_columns(right_candidates, len(right_colors))
    if left_cols is None or right_cols is None:
        return None

    grid = np.full((rows, cols), BLACK, dtype=int)
    grid[top_r, left_c + 1] = BLUE
    grid[bottom_r, right_c - 1] = BLUE
    grid[top_r + 1, left_c] = BLUE
    grid[bottom_r - 1, right_c] = BLUE

    grid[pivot_r, left_c + 1 : right_c] = YELLOW

    random.shuffle(left_colors)
    random.shuffle(right_colors)
    for col, color in zip(left_cols, left_colors):
        grid[pivot_r, col] = color
    for col, color in zip(right_cols, right_colors):
        grid[pivot_r, col] = color

    try:
        _validate_example(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    pattern_idx = _GEN_CALL_COUNT % len(PATTERN_CYCLE)
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(pattern_idx)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
