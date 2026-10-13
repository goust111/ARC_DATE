from __future__ import annotations

from itertools import combinations
import random

import numpy as np


GREY = 5
BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

RAY_COLORS = (RED, BLUE, YELLOW)
GRID_SIZE_RANGE = (14, 18)
POINT_COUNT_CHOICES = (4, 5, 6)
POINT_COUNT_WEIGHTS = (3, 3, 2)
GENERATE_ATTEMPTS = 400


def _color_from_index(index: int) -> int:
    return RAY_COLORS[index % 3]


def _find_points(grid: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere(grid == GREY)
    return [tuple(map(int, pt)) for pt in pts]


def _ordered_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    if row_span >= col_span:
        return sorted(points)
    return sorted(points, key=lambda pt: (pt[1], pt[0]))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    points = _find_points(g)
    assert 4 <= len(points) <= 6

    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    assert len(set(rows)) == len(points)
    assert len(set(cols)) == len(points)
    assert (max(rows) - min(rows)) != (max(cols) - min(cols))

    ordered = _ordered_points(points)
    out = g.copy()

    for index, (r, c) in enumerate(ordered):
        color = _color_from_index(index)
        row_mask = g[r] == BLACK
        col_mask = g[:, c] == BLACK
        out[r, row_mask] = color
        out[col_mask, c] = color

    for i, (r1, c1) in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            r2, c2 = ordered[j]
            color = _color_from_index(i + j)
            out[r1, c2] = color
            out[r2, c1] = color

    for r, c in points:
        out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend full row and column rays from isolated gray pixels, then recolor ray crossings by index-sum modulo 3.

    Input structure:
    - The grid is mostly black background with 4-6 isolated gray pixels.
    - No two gray pixels share a row or a column.
    - The overall gray-pixel bounding box has unequal row span and column span.

    Rule:
    - Measure the gray-pixel bounding box; if its row span is at least its column span, sort the pixels in row-major order, otherwise sort them in column-major order.
    - Assign indices 0, 1, 2, ... in that order, mapping index mod 3 to ray colors red, blue, yellow.
    - Each indexed pixel paints every background cell in its full row and full column with its ray color.
    - For each pair of distinct indexed pixels, recolor their two row-column crossing cells by (index_a + index_b) mod 3 using the same red/blue/yellow cycle, while original gray pixels remain gray.

    Key insight:
    - The long axis of the gray-pixel arrangement determines the scan order, and that hidden order controls both the ray colors and the nontrivial modulo coloring at ray intersections.
    """
    return _solve(grid)


def _pick_spaced_positions(limit: int, count: int) -> list[int] | None:
    candidates = list(range(1, limit - 1))
    for _ in range(120):
        sample = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _nontrivial_permutation(count: int) -> tuple[int, ...] | None:
    base = tuple(range(count))
    perms = list(range(count))
    for _ in range(100):
        random.shuffle(perms)
        perm = tuple(perms)
        if perm == base:
            continue
        if all(perm[i] < perm[i + 1] for i in range(count - 1)):
            continue
        return perm
    return None


def _orders_differ(points: list[tuple[int, int]]) -> bool:
    return sorted(points) != sorted(points, key=lambda pt: (pt[1], pt[0]))


def _input_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    points = _find_points(grid)
    if not (4 <= len(points) <= 6):
        return False
    if np.count_nonzero(grid != BLACK) != len(points):
        return False
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    if len(set(rows)) != len(points) or len(set(cols)) != len(points):
        return False
    if (max(rows) - min(rows)) == (max(cols) - min(cols)):
        return False
    for (r1, c1), (r2, c2) in combinations(points, 2):
        if abs(r1 - r2) + abs(c1 - c2) == 1:
            return False
    return _orders_differ(points)


def _has_visible_intersection_rule(inp: np.ndarray, out: np.ndarray) -> bool:
    ordered = _ordered_points(_find_points(inp))
    for i, (r1, _c1) in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            _r2, c2 = ordered[j]
            cross_color = int(out[r1, c2])
            a = _color_from_index(i)
            b = _color_from_index(j)
            if cross_color not in (a, b):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    count = random.choices(POINT_COUNT_CHOICES, weights=POINT_COUNT_WEIGHTS, k=1)[0]
    branch = random.choice(("row", "col"))

    rows = _pick_spaced_positions(height, count)
    cols = _pick_spaced_positions(width, count)
    if rows is None or cols is None:
        return None

    row_span = rows[-1] - rows[0]
    col_span = cols[-1] - cols[0]
    if branch == "row":
        if row_span <= col_span:
            return None
    else:
        if col_span <= row_span:
            return None

    perm = _nontrivial_permutation(count)
    if perm is None:
        return None

    points = [(rows[i], cols[perm[i]]) for i in range(count)]
    if not _orders_differ(points):
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, c in points:
        grid[r, c] = GREY

    if not _input_is_valid(grid):
        return None

    out = _solve(grid)
    if len(set(map(int, out.flat))) < 4:
        return None
    if not _has_visible_intersection_rule(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-crossing input")
