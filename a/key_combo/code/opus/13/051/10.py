from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

RANK_COLORS = {
    1: BLUE,
    2: RED,
    3: GREEN,
    4: YELLOW,
    5: TEAL,
}
RAY_COLORS = set(RANK_COLORS.values())
PIXEL_COUNT_WEIGHTS = [(3, 2), (4, 3), (5, 4)]
COLUMN_PATTERNS = {
    3: [(2, 1)],
    4: [(2, 1, 1), (3, 1), (2, 2)],
    5: [(2, 1, 1, 1), (3, 1, 1), (2, 2, 1), (3, 2)],
}
MAX_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _axis_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1
    return int(rows[0])


def _rectangle_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components_of_color(grid, ORANGE)
    assert comps
    comp = max(comps, key=lambda pts: int(pts.shape[0]))
    box = _bbox(comp)
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE)
    return box


def _ranked_pixels(grid: np.ndarray, axis_row: int) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == PINK)]
    assert 3 <= len(pts) <= 5
    dists = [abs(r - axis_row) for r, _c in pts]
    assert len(set(dists)) == len(dists)
    return sorted(pts, key=lambda rc: (abs(rc[0] - axis_row), rc[1], rc[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the isolated pixels by distance to the axis, then recolor and extend their rays toward the rectangle.

    Input structure:
    - A black background contains one full-width grey horizontal axis line, one solid orange rectangle, and 3-5 pink isolated pixels.
    - The orange rectangle sits on one side of the axis; all pink pixels sit on the opposite side and are column-aligned with the rectangle span.
    - Each pink pixel has a distinct Manhattan distance to the axis, so the ranking order is unique.

    Rule:
    - Identify the grey axis row, the orange rectangle, and the pink isolated pixels.
    - Rank the pink pixels by increasing distance to the axis; ranks 1..5 map to blue, red, green, yellow, teal.
    - In rank order, recolor the pixel itself and extend a vertical ray toward the rectangle through black cells, leaving the grey axis unchanged.
    - A ray stops before entering the orange rectangle or any previously painted ray pixel, so earlier ranked rays can shorten later ones.

    Key insight:
    - The axis supplies only the distance-ordering reference, while that ranking simultaneously determines ray color, draw order, and where later rays are blocked.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    axis = _axis_row(g)
    rect_r0, _rect_c0, rect_r1, _rect_c1 = _rectangle_bbox(g)
    pixels = _ranked_pixels(g, axis)

    rect_is_above = rect_r1 < axis
    step = -1 if rect_is_above else 1
    rect_color = ORANGE

    for rank, (r, c) in enumerate(pixels, start=1):
        ray_color = RANK_COLORS[rank]
        out[r, c] = ray_color
        rr = r + step
        while 0 <= rr < out.shape[0]:
            cell = int(out[rr, c])
            if cell == rect_color:
                break
            if cell in RAY_COLORS or cell == PINK:
                break
            if cell == GREY:
                rr += step
                continue
            if cell != BLACK:
                break
            out[rr, c] = ray_color
            rr += step

    return out


def _sample_weighted(options: Iterable[tuple[int, int]]) -> int:
    values = []
    weights = []
    for value, weight in options:
        values.append(int(value))
        weights.append(int(weight))
    return random.choices(values, weights=weights, k=1)[0]


def _choose_spaced_rows(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    for _ in range(80):
        rows = sorted(random.sample(candidates, count))
        if all(rows[i + 1] - rows[i] >= min_gap for i in range(len(rows) - 1)):
            return rows
    return None


def _oracle_from_meta(grid: np.ndarray, axis: int, rect_bbox: tuple[int, int, int, int], pixels: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    _r0, _c0, rect_r1, _c1 = rect_bbox
    previous_by_col: dict[int, int] = {}
    ranked = sorted(pixels, key=lambda rc: (rc[0] - axis, rc[1], rc[0]))
    for rank, (r, c) in enumerate(ranked, start=1):
        color = RANK_COLORS[rank]
        out[r, c] = color
        stop_row = previous_by_col.get(c, rect_r1)
        for rr in range(r - 1, stop_row, -1):
            if rr == axis:
                continue
            out[rr, c] = color
        previous_by_col[c] = r
    return out


def _validate_layout(grid: np.ndarray, axis: int, rect_bbox: tuple[int, int, int, int], pixels: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    if not (18 <= h <= 30 and 18 <= w <= 30):
        return False
    if axis <= 0 or axis >= h - 1:
        return False
    rect_r0, rect_c0, rect_r1, rect_c1 = rect_bbox
    if rect_r0 < 1 or rect_c0 < 1 or rect_r1 >= axis - 1 or rect_c1 >= w - 1:
        return False
    if rect_r1 > axis - 2:
        return False
    if rect_r1 - rect_r0 + 1 < 3 or rect_c1 - rect_c0 + 1 < 3:
        return False
    if not np.all(grid[axis] == GREY):
        return False
    if not np.all(grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] == ORANGE):
        return False

    if not (3 <= len(pixels) <= 5):
        return False
    seen_rows = set()
    cols = []
    for r, c in pixels:
        if r <= axis or not (rect_c0 <= c <= rect_c1):
            return False
        if grid[r, c] != PINK:
            return False
        if r in seen_rows:
            return False
        seen_rows.add(r)
        cols.append(c)
    if len(set(cols)) < 2:
        return False

    for i, (r1, c1) in enumerate(pixels):
        for r2, c2 in pixels[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False

    distances = [r - axis for r, _c in pixels]
    if len(set(distances)) != len(distances):
        return False

    column_groups: dict[int, list[int]] = {}
    for r, c in pixels:
        column_groups.setdefault(c, []).append(r)
    dup_groups = [sorted(rows) for rows in column_groups.values() if len(rows) >= 2]
    if not dup_groups:
        return False
    if not any(rows[-1] - rows[0] >= 3 for rows in dup_groups):
        return False

    return True


def _validate_semantics(grid: np.ndarray, out: np.ndarray, axis: int, rect_bbox: tuple[int, int, int, int], pixels: list[tuple[int, int]]) -> bool:
    expected = _oracle_from_meta(grid, axis, rect_bbox, pixels)
    if not np.array_equal(out, expected):
        return False

    rect_r0, rect_c0, rect_r1, rect_c1 = rect_bbox
    if not np.all(out[axis] == GREY):
        return False
    if not np.all(out[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] == ORANGE):
        return False
    if np.any(out == PINK):
        return False

    ranked = sorted(pixels, key=lambda rc: (rc[0] - axis, rc[1], rc[0]))
    by_col: dict[int, list[int]] = {}
    for rank, (r, c) in enumerate(ranked, start=1):
        if int(out[r, c]) != RANK_COLORS[rank]:
            return False
        by_col.setdefault(c, []).append(r)

    collision_visible = False
    for col, rows in by_col.items():
        rows = sorted(rows)
        if len(rows) == 1:
            continue
        for prev, cur in zip(rows, rows[1:]):
            if np.any(out[:axis, col] == out[cur, col]):
                return False
            if cur - prev >= 3:
                collision_visible = True
    if not collision_visible:
        return False

    changed = np.argwhere(out != grid)
    if changed.size == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    axis = random.randint(7, h - 7)
    rect_h = random.randint(3, min(5, axis - 3))
    rect_w = random.randint(5, min(9, w - 2))
    top_max = axis - rect_h - 2
    if top_max < 1:
        return None
    rect_r0 = random.randint(1, top_max)
    rect_c0 = random.randint(1, w - rect_w - 1)
    rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    n_pixels = _sample_weighted(PIXEL_COUNT_WEIGHTS)
    pattern = list(random.choice(COLUMN_PATTERNS[n_pixels]))
    random.shuffle(pattern)

    column_choices = list(range(rect_c0, rect_c0 + rect_w))
    if len(column_choices) < len(pattern):
        return None
    cols = random.sample(column_choices, len(pattern))

    row_candidates = list(range(axis + 2, h - 1))
    if len(row_candidates) < n_pixels:
        return None

    pixels: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    for col, group_size in zip(cols, pattern):
        available = [r for r in row_candidates if r not in used_rows]
        min_gap = 3 if group_size >= 2 else 1
        rows = _choose_spaced_rows(available, group_size, min_gap)
        if rows is None:
            return None
        used_rows.update(rows)
        pixels.extend((r, col) for r in rows)

    pixels.sort()
    grid = np.full((h, w), BLACK, dtype=int)
    grid[axis, :] = GREY
    rr0, cc0, rr1, cc1 = rect_bbox
    grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = ORANGE
    for r, c in pixels:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = PINK

    if not _validate_layout(grid, axis, rect_bbox, pixels):
        return None

    out = main(grid)
    if not _validate_semantics(grid, out, axis, rect_bbox, pixels):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
