from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MARKER_PALETTE = (3, 4, 6, 7)

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
ANCHOR_COUNT_CHOICES = (2, 3)
MARKER_COUNT_BY_ANCHORS = {2: (2,), 3: (2, 3)}
MIN_AXIS_GAP = 2
MAX_ATTEMPTS = 400


def _spaced_sample(start: int, stop: int, count: int, min_gap: int) -> list[int]:
    values = list(range(start, stop + 1))
    choices = [
        combo
        for combo in itertools.combinations(values, count)
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))
    ]
    if not choices:
        raise ValueError("no spaced sample available")
    picked = list(random.choice(choices))
    random.shuffle(picked)
    return picked


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = [int(c) for c in np.flatnonzero(grid[0] != BG)]
    return [(c, int(grid[0, c])) for c in cols]


def _extract_anchors(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == BLUE)
    return [tuple(map(int, rc)) for rc in coords]


def _build_ray_stage(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)

    for col, color in markers:
        out[0, col] = int(color)
    for r, c in anchors:
        out[r, c] = BLUE

    for r, c in anchors:
        row_mask = out[r, :] == BG
        col_mask = out[:, c] == BG
        out[r, row_mask] = BLUE
        out[col_mask, c] = BLUE

    for i, (r, _c) in enumerate(anchors):
        for j, (_r, c) in enumerate(anchors):
            if i != j:
                out[r, c] = RED

    return out


def _fill_intervals(stage: np.ndarray, markers: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    h = stage.shape[0]
    intervals: dict[int, tuple[int, int]] = {}
    for col, _color in markers:
        red_rows = [int(r) for r in np.flatnonzero(stage[1:, col] == RED) + 1]
        if not red_rows:
            intervals[col] = (1, h)
        elif len(red_rows) == 1:
            intervals[col] = (red_rows[0], h)
        else:
            intervals[col] = (red_rows[0], red_rows[1])
    return intervals


def _expected_from_layout(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    stage = _build_ray_stage(shape, anchors, markers)
    out = stage.copy()
    intervals = _fill_intervals(stage, markers)

    for col, _color in markers:
        out[0, col] = BG
    for col, color in markers:
        start, stop = intervals[col]
        out[start:stop, col] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend blue anchor rays into a cross-wall field, then let top markers fall until red crossings and paint downward.

    Input structure:
    - The grid has a black background, 2 or more distinct colored marker pixels on the top row, and 2 or more blue anchor pixels inside the grid.
    - Anchor pixels are isolated single cells with distinct rows and distinct columns.
    - The input contains only the original markers and anchors; the blue ray network and red wall crossings are implicit.

    Rule:
    - For each blue anchor, extend blue rays through its full row and full column across the grid, painting only black cells blue.
    - Wherever a row-ray from one anchor crosses a column-ray from a different anchor, paint that crossing red instead of blue; anchor cells themselves stay blue.
    - Remove each original top-row marker, then in its column find the first red cell below row 0. Recolor from that red cell downward with the marker color until the next red cell or the bottom edge.
    - If a marker column has no red cell at all, recolor that column from row 1 through the bottom edge with the marker color.

    Key insight:
    - Blue rays are guides, but only cross-anchor red intersections act as blockers; the drop result is determined by the full ray network before any marker starts filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    markers = _extract_markers(g)
    anchors = _extract_anchors(g)

    assert 2 <= len(markers) <= 4
    assert len(anchors) >= 2
    assert len({c for c, _color in markers}) == len(markers)
    assert len({color for _c, color in markers}) == len(markers)
    assert len({r for r, _c in anchors}) == len(anchors)
    assert len({c for _r, c in anchors}) == len(anchors)

    stage = _build_ray_stage(g.shape, anchors, markers)
    out = stage.copy()
    intervals = _fill_intervals(stage, markers)

    for col, _color in markers:
        out[0, col] = BG
    for col, color in markers:
        start, stop = intervals[col]
        out[start:stop, col] = int(color)
    return out


def _choose_anchor_layout(h: int, w: int, anchor_count: int) -> list[tuple[int, int]]:
    rows = _spaced_sample(2, h - 2, anchor_count, MIN_AXIS_GAP)
    cols = _spaced_sample(1, w - 2, anchor_count, MIN_AXIS_GAP)
    return list(zip(rows, cols))


def _choose_marker_columns(anchors: list[tuple[int, int]]) -> list[int]:
    by_row = sorted(anchors)
    marker_count = random.choice(MARKER_COUNT_BY_ANCHORS[len(anchors)])
    if len(anchors) == 2 or marker_count == 3:
        cols = [c for _r, c in by_row]
    else:
        top_col = by_row[0][1]
        second_col = random.choice((by_row[1][1], by_row[2][1]))
        cols = [top_col, second_col]
    cols = sorted(cols)
    return cols


def _build_input(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for col, color in markers:
        grid[0, col] = int(color)
    for r, c in anchors:
        grid[r, c] = BLUE
    return grid


def _anchors_are_necessary(inp: np.ndarray, expected: np.ndarray) -> bool:
    for r, c in _extract_anchors(inp):
        altered = inp.copy()
        altered[r, c] = BG
        try:
            alt_out = main(altered)
        except Exception:
            continue
        if np.array_equal(alt_out, expected):
            return False
    return True


def _layout_is_semantic(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> bool:
    h, _w = shape
    marker_cols = [col for col, _color in markers]
    anchor_col_to_row = {c: r for r, c in anchors}

    if any(col not in anchor_col_to_row for col in marker_cols):
        return False

    stage = _build_ray_stage(shape, anchors, markers)
    intervals = _fill_intervals(stage, markers)

    if any(stage[0, col] == RED for col in marker_cols):
        return False
    if any(not np.any(stage[1:, col] == RED) for col in marker_cols):
        return False

    starts_and_stops = {intervals[col] for col in marker_cols}
    if len(markers) > 1 and len(starts_and_stops) == 1:
        return False

    topmost_anchor = min(anchors)[0]
    distinguishes_blue_from_red = False
    for col in marker_cols:
        start, stop = intervals[col]
        own_anchor_row = anchor_col_to_row[col]
        if own_anchor_row == topmost_anchor and own_anchor_row < start:
            distinguishes_blue_from_red = True
        if not (1 <= start < stop <= h):
            return False
    if not distinguishes_blue_from_red:
        return False

    if len(anchors) == 3:
        if not any(stop < h for start, stop in starts_and_stops):
            return False

    return True


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    anchor_count = random.choice(ANCHOR_COUNT_CHOICES)
    anchors = _choose_anchor_layout(h, w, anchor_count)
    marker_cols = _choose_marker_columns(anchors)
    marker_colors = random.sample(MARKER_PALETTE, len(marker_cols))
    markers = list(zip(marker_cols, marker_colors))

    if not _layout_is_semantic((h, w), anchors, markers):
        return None

    inp = _build_input((h, w), anchors, markers)
    expected = _expected_from_layout((h, w), anchors, markers)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if not _anchors_are_necessary(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 028/1")
