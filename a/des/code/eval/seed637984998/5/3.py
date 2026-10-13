from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
AZURE = 8


def _bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _band_rows(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color not in (bg, GREY, MAROON, AZURE) and np.all(grid[r, :] == color):
            rows.append((r, color))
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _bg(g)
    bands = _band_rows(g, bg)
    if not bands:
        return out

    starts = np.argwhere(g == MAROON)
    ends = np.argwhere(g == AZURE)
    if starts.size == 0 or ends.size == 0:
        return out
    sr = int(starts[0, 0])
    er = int(ends[0, 0])
    downward = er >= sr
    ordered = sorted(bands, key=lambda x: x[0], reverse=not downward)

    grey = np.argwhere(g == GREY)
    for r, c in grey:
        rr = int(r)
        color = ordered[0][1]
        for br, bc in ordered:
            if (downward and br <= rr) or ((not downward) and br >= rr):
                color = bc
        out[rr, int(c)] = color
    return out


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted([c0, c1])
        for c in range(lo, hi + 1):
            if grid[r0, c] == BG:
                grid[r0, c] = GREY
    elif c0 == c1:
        lo, hi = sorted([r0, r1])
        for r in range(lo, hi + 1):
            if grid[r, c0] == BG:
                grid[r, c0] = GREY


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(22, 34)
        w = random.randint(18, 30)
        grid = np.zeros((h, w), dtype=int)
        band_count = random.randint(2, 4)
        colors = random.sample([1, 2, 3, 4, 6, 7], band_count)
        rows = sorted(random.sample(range(5, h - 5), band_count))
        for r, color in zip(rows, colors):
            grid[r, :] = color

        col = random.randint(3, w - 4)
        start = (random.randint(1, rows[0] - 2), col)
        end = (random.randint(rows[-1] + 2, h - 2), random.randint(3, w - 4))
        points = [start]
        cur_col = col
        for br in rows:
            nr = br - 1
            if nr > points[-1][0]:
                points.append((nr, cur_col))
            cur_col = random.randint(3, w - 4)
            points.append((br + 1, cur_col))
        points.append((end[0], cur_col))
        points.append(end)

        for a, b in zip(points, points[1:]):
            if a[0] != b[0] and a[1] != b[1]:
                mid = (b[0], a[1])
                _draw_segment(grid, a, mid)
                _draw_segment(grid, mid, b)
            else:
                _draw_segment(grid, a, b)
        grid[start] = MAROON
        grid[end] = AZURE
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate colored path puzzle")


if __name__ == "__main__":
    for _ in range(20):
        x = generate_input()
        y = main(x)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
