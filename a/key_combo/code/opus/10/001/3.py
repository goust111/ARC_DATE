from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8


def _bar_info(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected a grey vertical bar"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, "expected exactly one grey column"
    col = int(cols[0])
    rows = np.sort(coords[:, 0])
    height = int(rows.size)
    assert np.array_equal(rows, np.arange(height)), "grey bar must be top-anchored and contiguous"
    return col, height


def _terrain_starts(grid: np.ndarray, bar_col: int, bar_height: int) -> np.ndarray:
    height, width = grid.shape
    starts = np.empty(width, dtype=int)
    for col in range(width):
        terrain_rows = np.flatnonzero(grid[:, col] == GREEN)
        assert terrain_rows.size > 0, "every column must contain terrain"
        start = int(terrain_rows[0])
        assert np.array_equal(terrain_rows, np.arange(start, height)), "terrain must be a contiguous suffix"
        starts[col] = start

        above = grid[:start, col]
        if col == bar_col:
            assert start > bar_height, "terrain may not overlap the grey bar"
            assert np.all(above[:bar_height] == GREY), "bar column must contain the grey bar above terrain"
            assert np.all(above[bar_height:] == BLACK), "bar column gap above terrain must be black"
        else:
            assert np.all(above == BLACK), "non-bar columns may only contain black above terrain"
    return starts


def _analyze(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREEN, GREY}, "input must only contain black, green, and grey"
    bar_col, bar_height = _bar_info(g)
    assert 0 < bar_col < g.shape[1] - 1, "grey bar must have space on both sides"
    starts = _terrain_starts(g, bar_col, bar_height)
    heights = g.shape[0] - starts
    assert np.count_nonzero(heights == heights.min()) == 1, "lowest terrain column must be unique"
    assert np.count_nonzero(heights == heights.max()) == 1, "highest terrain column must be unique"
    return bar_col, bar_height, starts


def _column_plan(bar_col: int, bar_height: int, starts: np.ndarray) -> list[dict]:
    plans: list[dict] = []
    for col in range(starts.size):
        if col == bar_col:
            continue
        if col < bar_col:
            distance = bar_col - col
            target = bar_height + 2 * distance
            base_color = CYAN
            side = "left"
        else:
            distance = col - bar_col
            target = bar_height - 2 * distance
            base_color = MAGENTA
            side = "right"

        actual = min(target, int(starts[col])) if target > 0 else 0
        plans.append(
            {
                "col": col,
                "side": side,
                "distance": distance,
                "target": target,
                "actual": actual,
                "truncated": target > 0 and actual < target,
                "color": ORANGE if target > 0 and actual < target else base_color,
            }
        )
    return plans


def _render_output(grid: np.ndarray, bar_col: int, bar_height: int, starts: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for plan in _column_plan(bar_col, bar_height, starts):
        if plan["target"] <= 0:
            continue
        out[: plan["actual"], plan["col"]] = int(plan["color"])
    out[grid == GREEN] = GREEN
    out[:bar_height, bar_col] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the space around a top grey bar with distance-based gradients clipped by a green terrain contour.

    Input structure:
    - The grid has a black background, one top-anchored one-cell-wide grey vertical bar, and a green terrain suffix in every column.
    - In each column, green terrain occupies a contiguous run from the bottom upward, so every column has a well-defined terrain surface.
    - The grey bar stands above the terrain with at least one free column on each side.

    Rule:
    - Preserve the grey bar and all green terrain cells.
    - For each column d steps left of the bar, compute target height H + 2*d and fill from the top downward with cyan up to either that height or the terrain surface, whichever comes first.
    - For each column d steps right of the bar, compute target height H - 2*d; if it is positive, fill from the top downward with magenta up to either that height or the terrain surface.
    - Whenever a filled column is clipped by the terrain before reaching its target height, recolor the whole filled segment in that column to orange.

    Key insight:
    - The visible color of each filled column depends on two coupled facts at once: its side-dependent distance formula and whether the terrain surface truncates that theoretical height.
    """
    g = np.asarray(grid, dtype=int)
    bar_col, bar_height, starts = _analyze(g)
    return _render_output(g, bar_col, bar_height, starts)


GRID_HEIGHT_RANGE = (16, 19)
GRID_WIDTH_RANGE = (12, 16)
BAR_HEIGHT_RANGE = (3, 5)
TERRAIN_HEIGHT_MIN = 1
TERRAIN_HEIGHT_MAX = 5
MAX_STEP = 2
MAX_ATTEMPTS = 400
SEGMENT_INTERNAL_VALUES = (2, 3, 4)
STEP_WEIGHTS = {0: 4, 1: 4, 2: 1}


def _weighted_step(prev: int) -> int:
    candidates = []
    weights = []
    for nxt in range(TERRAIN_HEIGHT_MIN, TERRAIN_HEIGHT_MAX + 1):
        delta = abs(nxt - prev)
        if delta <= MAX_STEP:
            candidates.append(nxt)
            weights.append(STEP_WEIGHTS.get(delta, 0))
    return int(random.choices(candidates, weights=weights, k=1)[0])


def _choose_edge_value(anchor_value: int, distance: int) -> int | None:
    candidates = [
        value
        for value in SEGMENT_INTERNAL_VALUES
        if abs(value - anchor_value) <= MAX_STEP * distance
    ]
    if not candidates:
        return None
    return int(random.choice(candidates))


def _segment(start: int, end: int, steps: int) -> list[int] | None:
    if steps == 0:
        return [int(start)] if start == end else None
    values = [int(start)]
    cur = int(start)
    end = int(end)
    for pos in range(1, steps):
        remaining = steps - pos
        candidates = []
        weights = []
        for nxt in SEGMENT_INTERNAL_VALUES:
            if abs(nxt - cur) > MAX_STEP:
                continue
            if abs(end - nxt) > MAX_STEP * remaining:
                continue
            candidates.append(int(nxt))
            weights.append(STEP_WEIGHTS[abs(nxt - cur)])
        if not candidates:
            return None
        cur = int(random.choices(candidates, weights=weights, k=1)[0])
        values.append(cur)
    if abs(end - cur) > MAX_STEP:
        return None
    values.append(end)
    return values


def _terrain_profile(width: int, grid_height: int, bar_col: int, bar_height: int) -> np.ndarray | None:
    truncation_sources = [
        col
        for col in range(bar_col)
        if bar_height + 2 * (bar_col - col) > grid_height - TERRAIN_HEIGHT_MAX
    ]
    if not truncation_sources:
        return None

    peak_col = int(random.choice(truncation_sources))
    valley_choices = [col for col in range(width) if col != peak_col and abs(col - peak_col) >= 2]
    if not valley_choices:
        return None
    valley_col = int(random.choice(valley_choices))

    controls = {peak_col: TERRAIN_HEIGHT_MAX, valley_col: TERRAIN_HEIGHT_MIN}

    if 0 not in controls:
        right = min(col for col in controls if col > 0)
        edge_value = _choose_edge_value(controls[right], right)
        if edge_value is None:
            return None
        controls[0] = edge_value

    last = width - 1
    if last not in controls:
        left = max(col for col in controls if col < last)
        edge_value = _choose_edge_value(controls[left], last - left)
        if edge_value is None:
            return None
        controls[last] = edge_value

    cols = sorted(controls)
    pieces: list[int] = []
    for left, right in zip(cols, cols[1:]):
        seg = _segment(controls[left], controls[right], right - left)
        if seg is None:
            return None
        if pieces:
            pieces.extend(seg[1:])
        else:
            pieces.extend(seg)

    heights = np.array(pieces, dtype=int)
    if heights.size != width:
        return None
    if np.max(np.abs(np.diff(heights))) > MAX_STEP:
        return None
    if np.count_nonzero(heights == TERRAIN_HEIGHT_MIN) != 1:
        return None
    if np.count_nonzero(heights == TERRAIN_HEIGHT_MAX) != 1:
        return None
    return heights


def _build_input(height: int, width: int, bar_col: int, bar_height: int, terrain_heights: np.ndarray) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for col, terrain_height in enumerate(terrain_heights):
        grid[height - int(terrain_height) :, col] = GREEN
    grid[:bar_height, bar_col] = GREY
    return grid


def _semantic_accepts(grid: np.ndarray) -> bool:
    try:
        bar_col, bar_height, starts = _analyze(grid)
    except AssertionError:
        return False

    plans = _column_plan(bar_col, bar_height, starts)
    terrain_heights = grid.shape[0] - starts
    expected = _render_output(grid, bar_col, bar_height, starts)

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    positive = [plan for plan in plans if plan["target"] > 0]
    truncated = [plan for plan in positive if plan["truncated"]]
    left_ok = [plan for plan in positive if plan["side"] == "left" and not plan["truncated"]]
    right_ok = [plan for plan in positive if plan["side"] == "right" and not plan["truncated"]]
    right_zero = [plan for plan in plans if plan["side"] == "right" and plan["target"] <= 0]
    if not truncated or not left_ok or not right_ok or not right_zero:
        return False

    visible_colors = set(map(int, np.unique(actual))) - {BLACK, GREEN, GREY}
    if visible_colors != {CYAN, MAGENTA, ORANGE}:
        return False

    peak_col = int(np.argmax(terrain_heights))
    if peak_col >= bar_col:
        return False

    for plan in positive:
        col = int(plan["col"])
        expected_color = int(plan["color"])
        if not np.all(actual[: plan["actual"], col] == expected_color):
            return False
        if plan["actual"] < starts[col] and actual[plan["actual"], col] != BLACK:
            return False
        if np.any(actual[starts[col] :, col] != GREEN):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bar_height = random.randint(*BAR_HEIGHT_RANGE)

    bar_candidates = [col for col in range(6, width - 3)]
    if not bar_candidates:
        return None
    middle = (width - 1) / 2.0
    weights = [1.0 + 0.6 * abs(col - middle) for col in bar_candidates]
    bar_col = int(random.choices(bar_candidates, weights=weights, k=1)[0])

    terrain_heights = _terrain_profile(width, height, bar_col, bar_height)
    if terrain_heights is None:
        return None

    grid = _build_input(height, width, bar_col, bar_height, terrain_heights)
    return grid if _semantic_accepts(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
