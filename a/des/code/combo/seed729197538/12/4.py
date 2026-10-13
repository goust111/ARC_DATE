import random
from collections import deque

import numpy as np


BG = 0
TRAIL = 5
DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _components(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _contiguous_line(cells: set[tuple[int, int]], expected: list[tuple[int, int]]) -> bool:
    return cells == set(expected)


def _parse_piece(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    for size in (3, 2):
        for r0 in range(min(rows), max(rows) - size + 2):
            for c0 in range(min(cols), max(cols) - size + 2):
                body = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
                if not body <= cells:
                    continue
                tail = cells - body
                if not 1 <= len(tail) <= 3:
                    continue
                mid_r = r0 + size // 2
                mid_c = c0 + size // 2
                n = len(tail)
                checks = [
                    ([(mid_r, c0 - i) for i in range(1, n + 1)], (0, 1)),
                    ([(mid_r, c0 + size - 1 + i) for i in range(1, n + 1)], (0, -1)),
                    ([(r0 - i, mid_c) for i in range(1, n + 1)], (1, 0)),
                    ([(r0 + size - 1 + i, mid_c) for i in range(1, n + 1)], (-1, 0)),
                ]
                for expected, direction in checks:
                    if _contiguous_line(tail, expected):
                        return body, tail, direction
    return None


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    pieces = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, TRAIL))
    for color in colors:
        for comp in _components(grid, color):
            parsed = _parse_piece(comp)
            if parsed is not None:
                body, tail, direction = parsed
                pieces.append((color, body, tail, direction))

    for _color, body, tail, _direction in pieces:
        for r, c in body | tail:
            out[r, c] = BG

    h, w = grid.shape
    for color, body, _tail, (dr, dc) in pieces:
        shift = 0
        while True:
            nxt = shift + 1
            moved = {(r + dr * nxt, c + dc * nxt) for r, c in body}
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in moved):
                break
            shift = nxt

        center_r = round(sum(r for r, _ in body) / len(body))
        center_c = round(sum(c for _, c in body) / len(body))
        final_body = {(r + dr * shift, c + dc * shift) for r, c in body}
        for step in range(shift):
            rr, cc = center_r + dr * step, center_c + dc * step
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in final_body:
                out[rr, cc] = TRAIL
        for r, c in final_body:
            out[r, c] = color
    return out


def _place_piece(grid: np.ndarray, color: int, body_size: int, tail_len: int, direction: tuple[int, int], r0: int, c0: int) -> None:
    s = body_size
    grid[r0 : r0 + s, c0 : c0 + s] = color
    mid_r = r0 + s // 2
    mid_c = c0 + s // 2
    dr, dc = direction
    if (dr, dc) == (0, 1):
        for i in range(1, tail_len + 1):
            grid[mid_r, c0 - i] = color
    elif (dr, dc) == (0, -1):
        for i in range(1, tail_len + 1):
            grid[mid_r, c0 + s - 1 + i] = color
    elif (dr, dc) == (1, 0):
        for i in range(1, tail_len + 1):
            grid[r0 - i, mid_c] = color
    else:
        for i in range(1, tail_len + 1):
            grid[r0 + s - 1 + i, mid_c] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(200):
        h = rng.randint(12, 20)
        w = rng.randint(12, 20)
        s = rng.choice([2, 3])
        tail = rng.randint(1, 3)
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        color = rng.choice(colors)
        if direction == (0, 1):
            r0 = rng.randint(2, h - s - 3)
            c0 = rng.randint(tail + 1, w - s - 3)
        elif direction == (0, -1):
            r0 = rng.randint(2, h - s - 3)
            c0 = rng.randint(3, w - s - tail - 2)
        elif direction == (1, 0):
            r0 = rng.randint(tail + 1, h - s - 3)
            c0 = rng.randint(2, w - s - 3)
        else:
            r0 = rng.randint(3, h - s - tail - 2)
            c0 = rng.randint(2, w - s - 3)
        grid = np.zeros((h, w), dtype=int)
        _place_piece(grid, color, s, tail, direction, r0, c0)
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("generate_input failed")
