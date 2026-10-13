from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
GREEN = 3
PAIR_COLOR_POOL = [1, 2, 4, 6, 7, 8, 9]

PAIR_COUNT_CHOICES = [2, 3, 4]
PAIR_COUNT_WEIGHTS = [4, 6, 3]
GRID_SIDE_RANGE = (18, 26)
RECT_SIZE_RANGE = (2, 4)
OPEN_DIAGONAL_RANGE = (2, 6)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(cells)
    return components


def _bbox(cells: list[Coord]) -> BBox:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(cells: list[Coord], shape: tuple[int, int]) -> bool:
    last_r = shape[0] - 1
    last_c = shape[1] - 1
    return any(r == 0 or r == last_r or c == 0 or c == last_c for r, c in cells)


def _is_solid_rectangle(cells: list[Coord]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _interior_rectangles(grid: np.ndarray, color: int) -> list[BBox]:
    boxes: list[BBox] = []
    for cells in _find_components(grid == color):
        if _touches_boundary(cells, grid.shape):
            continue
        assert _is_solid_rectangle(cells), "interior components must be solid rectangles"
        boxes.append(_bbox(cells))
    return boxes


def _facing_corners(box_a: BBox, box_b: BBox) -> tuple[Coord, Coord]:
    upper, lower = sorted((box_a, box_b), key=lambda box: (box[0], box[1]))
    candidates = [
        ((upper[2], upper[1]), (lower[0], lower[3])),
        ((upper[2], upper[3]), (lower[0], lower[1])),
    ]
    valid = []
    for start, end in candidates:
        row_delta = end[0] - start[0]
        col_delta = end[1] - start[1]
        if row_delta > 0 and abs(row_delta) == abs(col_delta):
            valid.append((start, end))
    assert len(valid) == 1, "rectangles must have exactly one diagonally aligned facing-corner pair"
    return valid[0]


def _connect_rectangles(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        rectangles = _interior_rectangles(g, color)
        assert len(rectangles) == 2, "each foreground color must have exactly two interior rectangles"
        start, end = _facing_corners(rectangles[0], rectangles[1])
        steps = abs(end[0] - start[0])
        dr = 1 if end[0] > start[0] else -1
        dc = 1 if end[1] > start[1] else -1
        for k in range(1, steps):
            r = start[0] + k * dr
            c = start[1] + k * dc
            assert out[r, c] == bg, "diagonal connector must pass through background"
            out[r, c] = color
    return out


def _corridor_lines(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    rows = [
        r
        for r in range(grid.shape[0])
        if grid[r, 0] != bg and grid[r, -1] != bg and np.all(grid[r, 1:-1] == bg)
    ]
    cols = [
        c
        for c in range(grid.shape[1])
        if grid[0, c] != bg and grid[-1, c] != bg and np.all(grid[1:-1, c] == bg)
    ]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color rectangle pair diagonally, then color the still-open edge-to-edge corridors green.

    Input structure:
    - The grid has a dominant black background with a few foreground pixels on the outer boundary.
    - Each foreground color also has exactly two separated solid interior rectangles.
    - The two rectangles of one color are diagonally offset so one pair of facing corners lies on a 45-degree line.

    Rule:
    - For each color, draw the missing open diagonal segment between the two interior rectangles, leaving the rectangle pixels themselves unchanged.
    - After all such diagonals are added, find every row or column whose two boundary endpoint cells are foreground and whose interior cells are all background.
    - Recolor only those interior corridor cells green, while keeping every original foreground pixel and every newly added diagonal pixel unchanged.

    Key insight:
    - Corridor detection happens after the diagonal connectors are drawn, so some edge-to-edge rows or columns that look open in the input become blocked and must not be colored green.
    """
    connected = _connect_rectangles(np.asarray(grid, dtype=int))
    bg = _get_bg(connected)
    rows, cols = _corridor_lines(connected, bg)

    out = connected.copy()
    for r in rows:
        out[r, 1:-1] = GREEN
    for c in cols:
        out[1:-1, c] = GREEN
    return out


def _box_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _mark_forbidden(forbidden: np.ndarray, cells: list[Coord]) -> None:
    for r, c in cells:
        for rr in range(max(0, r - PLACEMENT_PADDING), min(forbidden.shape[0], r + PLACEMENT_PADDING + 1)):
            for cc in range(max(0, c - PLACEMENT_PADDING), min(forbidden.shape[1], c + PLACEMENT_PADDING + 1)):
                forbidden[rr, cc] = True


def _pair_plan(h: int, w: int, sign: int) -> tuple[BBox, BBox, list[Coord]] | None:
    upper_h = random.randint(*RECT_SIZE_RANGE)
    upper_w = random.randint(*RECT_SIZE_RANGE)
    lower_h = random.randint(*RECT_SIZE_RANGE)
    lower_w = random.randint(*RECT_SIZE_RANGE)
    open_len = random.randint(*OPEN_DIAGONAL_RANGE)
    delta = open_len + 1

    max_r0 = h - upper_h - delta - lower_h
    if max_r0 < 1:
        return None
    upper_r0 = random.randint(1, max_r0)

    if sign == 1:
        max_c0 = w - upper_w - delta - lower_w
        if max_c0 < 1:
            return None
        upper_c0 = random.randint(1, max_c0)
        upper = (upper_r0, upper_c0, upper_r0 + upper_h - 1, upper_c0 + upper_w - 1)
        end_r = upper[2] + delta
        end_c = upper[3] + delta
        lower = (end_r, end_c, end_r + lower_h - 1, end_c + lower_w - 1)
        diag = [(upper[2] + k, upper[3] + k) for k in range(1, delta)]
    else:
        min_c0 = delta + lower_w
        max_c0 = w - upper_w - 1
        if min_c0 > max_c0:
            return None
        upper_c0 = random.randint(min_c0, max_c0)
        upper = (upper_r0, upper_c0, upper_r0 + upper_h - 1, upper_c0 + upper_w - 1)
        end_r = upper[2] + delta
        end_c = upper[1] - delta
        lower = (end_r, end_c - lower_w + 1, end_r + lower_h - 1, end_c)
        diag = [(upper[2] + k, upper[1] - k) for k in range(1, delta)]

    if min(*upper, *lower) < 0:
        return None
    if upper[2] >= h - 1 or upper[3] >= w - 1 or lower[2] >= h - 1 or lower[3] >= w - 1:
        return None
    return upper, lower, diag


def _place_pair(h: int, w: int, forbidden: np.ndarray, color: int, sign: int) -> dict | None:
    for _ in range(200):
        plan = _pair_plan(h, w, sign)
        if plan is None:
            continue
        upper, lower, diag = plan
        cells = _box_cells(upper) + _box_cells(lower) + diag
        if any(forbidden[r, c] for r, c in cells):
            continue
        return {"color": color, "upper": upper, "lower": lower, "diag": diag, "cells": cells}
    return None


def _sample_signs(count: int) -> list[int]:
    signs = [random.choice([-1, 1]) for _ in range(count)]
    if count >= 2 and random.random() < 0.7:
        signs[0] = -1
        signs[1] = 1
        random.shuffle(signs)
    return signs


def _paint_pairs(grid: np.ndarray, pairs: list[dict], include_diagonals: bool) -> None:
    for pair in pairs:
        color = int(pair["color"])
        for box in (pair["upper"], pair["lower"]):
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        if include_diagonals:
            for r, c in pair["diag"]:
                grid[r, c] = color


def _empty_line_candidates(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(1, grid.shape[0] - 1) if np.all(grid[r, 1:-1] == bg)]
    cols = [c for c in range(1, grid.shape[1] - 1) if np.all(grid[1:-1, c] == bg)]
    return rows, cols


def _blocked_line_candidates(input_base: np.ndarray, connected: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    rows = [
        r
        for r in range(1, input_base.shape[0] - 1)
        if np.all(input_base[r, 1:-1] == bg) and np.any(connected[r, 1:-1] != bg)
    ]
    cols = [
        c
        for c in range(1, input_base.shape[1] - 1)
        if np.all(input_base[1:-1, c] == bg) and np.any(connected[1:-1, c] != bg)
    ]
    return rows, cols


def _sample_mode() -> tuple[int, int]:
    return random.choice([(1, 0), (0, 1), (1, 1)])


def _paint_endpoints(grid: np.ndarray, row_lines: list[int], col_lines: list[int], colors: list[int]) -> None:
    for r in row_lines:
        grid[r, 0] = random.choice(colors)
        grid[r, -1] = random.choice(colors)
    for c in col_lines:
        grid[0, c] = random.choice(colors)
        grid[-1, c] = random.choice(colors)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_pairs = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(PAIR_COLOR_POOL, n_pairs)
    signs = _sample_signs(n_pairs)

    forbidden = np.zeros((h, w), dtype=bool)
    pairs: list[dict] = []
    for color, sign in sorted(
        zip(colors, signs),
        key=lambda item: random.random(),
    ):
        pair = _place_pair(h, w, forbidden, color, sign)
        if pair is None:
            return None
        pairs.append(pair)
        _mark_forbidden(forbidden, pair["cells"])

    input_base = np.full((h, w), BACKGROUND, dtype=int)
    connected = np.full((h, w), BACKGROUND, dtype=int)
    _paint_pairs(input_base, pairs, include_diagonals=False)
    _paint_pairs(connected, pairs, include_diagonals=True)

    actual_row_candidates, actual_col_candidates = _empty_line_candidates(connected, BACKGROUND)
    blocked_row_candidates, blocked_col_candidates = _blocked_line_candidates(input_base, connected, BACKGROUND)

    need_actual_rows, need_actual_cols = _sample_mode()
    need_blocked_rows, need_blocked_cols = _sample_mode()

    if len(actual_row_candidates) < need_actual_rows or len(actual_col_candidates) < need_actual_cols:
        return None
    if len(blocked_row_candidates) < need_blocked_rows or len(blocked_col_candidates) < need_blocked_cols:
        return None

    actual_rows = sorted(random.sample(actual_row_candidates, need_actual_rows))
    actual_cols = sorted(random.sample(actual_col_candidates, need_actual_cols))
    blocked_rows = sorted(random.sample(blocked_row_candidates, need_blocked_rows))
    blocked_cols = sorted(random.sample(blocked_col_candidates, need_blocked_cols))

    inp = input_base.copy()
    _paint_endpoints(inp, actual_rows + blocked_rows, actual_cols + blocked_cols, colors)

    pre_rows, pre_cols = _corridor_lines(inp, BACKGROUND)
    post_connected = _connect_rectangles(inp)
    post_rows, post_cols = _corridor_lines(post_connected, BACKGROUND)
    if pre_rows != sorted(actual_rows + blocked_rows):
        return None
    if pre_cols != sorted(actual_cols + blocked_cols):
        return None
    if post_rows != actual_rows or post_cols != actual_cols:
        return None

    out = main(inp)
    if not np.any(out == GREEN):
        return None
    if not ((set(pre_rows) - set(post_rows)) or (set(pre_cols) - set(post_cols))):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
