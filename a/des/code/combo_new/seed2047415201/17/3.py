from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
MAROON = 9
PAIR_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return [a]
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    sr = 0 if dr == 0 else int(math.copysign(1, dr))
    sc = 0 if dc == 0 else int(math.copysign(1, dc))
    return [(r0 + sr * i, c0 + sc * i) for i in range(steps + 1)]


def _draw_x(out: np.ndarray, original: np.ndarray, center: tuple[int, int], endpoints: set[tuple[int, int]]) -> None:
    h, w = out.shape
    cr, cc = center
    for dr, dc in [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]:
        rr, col = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= col < w:
            if int(original[rr, col]) == BLACK or (rr, col) in endpoints:
                out[rr, col] = MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color pixel pairs when their straight sight line is clear, otherwise replace the pair with X marks.

    Input structure:
    - Black is the background.
    - Every non-background color appears as exactly two singleton pixels.
    - A pair may be horizontally, vertically, or diagonally aligned; other foreground pixels may block that path.

    Rule:
    - For each color pair, inspect the original straight path between the two pixels.
    - If the interior path contains only Black, draw the full connecting line in that color.
    - If another foreground pixel blocks the path, erase the two endpoints and draw clipped 3x3 Maroon X marks centered
      on their original positions while preserving unrelated foreground blockers.

    Key insight: Visibility through the original scene chooses between constructive connection and destructive marking.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BLACK)
    decisions: list[tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]] | None, bool]] = []
    for color in colors:
        pts = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        if len(pts) != 2:
            continue
        cells = _line_cells(pts[0], pts[1])
        clear = cells is not None and all(int(g[r, c]) == BLACK for r, c in cells[1:-1])
        decisions.append((color, pts[0], pts[1], cells, clear))

    for color, a, b, cells, clear in decisions:
        endpoints = {a, b}
        if clear and cells is not None:
            for r, c in cells:
                out[r, c] = int(color)
        else:
            out[a] = BLACK
            out[b] = BLACK
            _draw_x(out, g, a, endpoints)
            _draw_x(out, g, b, endpoints)
    return out


def _place_clear_diagonal(grid: np.ndarray, color: int) -> bool:
    rows, cols = grid.shape
    for _attempt in range(80):
        length = random.randint(3, min(6, rows - 3, cols - 3))
        sr = random.randint(1, rows - length - 1)
        sc = random.randint(1, cols - length - 1)
        if random.choice([True, False]):
            cells = [(sr + i, sc + i) for i in range(length)]
        else:
            cells = [(sr + i, sc + length - 1 - i) for i in range(length)]
        if all(int(grid[r, c]) == BLACK for r, c in cells):
            grid[cells[0]] = int(color)
            grid[cells[-1]] = int(color)
            return True
    return False


def generate_input() -> np.ndarray:
    rows = random.randint(12, 22)
    cols = random.randint(12, 24)
    grid = np.full((rows, cols), BLACK, dtype=int)
    r_block = random.randint(3, rows - 4)
    mid = random.randint(4, cols - 5)
    c_left = random.randint(1, min(3, mid - 2))
    c_right = random.randint(max(mid + 2, cols - 4), cols - 2)
    r_second = random.choice([r for r in range(1, rows - 1) if r != r_block])

    grid[r_block, c_left] = 1
    grid[r_block, c_right] = 1
    grid[r_block, mid] = 2
    grid[r_second, mid] = 2

    _place_clear_diagonal(grid, 3)
    if random.choice([True, False]):
        for _attempt in range(80):
            r = random.randint(1, rows - 2)
            c0 = random.randint(1, cols - 5)
            c1 = c0 + random.randint(3, min(6, cols - c0 - 1))
            if all(int(grid[r, c]) == BLACK for c in range(c0, c1 + 1)):
                grid[r, c0] = 4
                grid[r, c1] = 4
                break
    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
