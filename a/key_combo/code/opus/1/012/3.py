import random
from collections import deque

import numpy as np


def _black_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0 and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _containers(grid):
    result = []
    for comp in _black_components(grid):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not (np.all(grid[r0, c0:c1 + 1] == 0) and np.all(grid[r1, c0:c1 + 1] == 0)
                and np.all(grid[r0:r1 + 1, c0] == 0) and np.all(grid[r0:r1 + 1, c1] == 0)):
            continue
        interior = grid[r0 + 1:r1, c0 + 1:c1]
        if interior.size and np.all(np.isin(interior, [5, 7])):
            result.append((int(np.count_nonzero(interior == 5)), r0, r1, c0, c1))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    cons = sorted(_containers(grid), key=lambda x: x[0])
    height = sum(r1 - r0 + 1 for _, r0, r1, _, _ in cons)
    width = max(c1 - c0 + 1 for _, _, _, c0, c1 in cons)
    out = np.full((height, width), 6, dtype=int)
    row = 0
    for _, r0, r1, c0, c1 in cons:
        h, w = r1 - r0 + 1, c1 - c0 + 1
        out[row:row + h, :w] = 0
        out[row + 1:row + h - 1, 1:w - 1] = 7
        row += h
    return out


def _fits(grid, r, c, h, w):
    return r + h <= grid.shape[0] and c + w <= grid.shape[1] and np.all(grid[max(0, r - 1):min(grid.shape[0], r + h + 1), max(0, c - 1):min(grid.shape[1], c + w + 1)] == 6)


def _draw_container(grid, r, c, h, w, gray_count):
    grid[r:r + h, c] = 0
    grid[r:r + h, c + w - 1] = 0
    grid[r, c:c + w] = 0
    grid[r + h - 1, c:c + w] = 0
    cells = [(rr, cc) for rr in range(r + 1, r + h - 1) for cc in range(c + 1, c + w - 1)]
    side = random.choice(["top", "left", "bottom", "right"])
    if side == "top":
        cells = sorted(cells)
    elif side == "bottom":
        cells = sorted(cells, reverse=True)
    elif side == "left":
        cells = sorted(cells, key=lambda p: (p[1], p[0]))
    else:
        cells = sorted(cells, key=lambda p: (-p[1], p[0]))
    orange = len(cells) - gray_count
    for i, p in enumerate(cells):
        grid[p] = 7 if i < orange else 5


def generate_input() -> np.ndarray:
    grid = np.full((18, 18), 6, dtype=int)
    n = random.randint(3, 5)
    used_gray = set()
    for _ in range(n):
        for _attempt in range(200):
            h = random.randint(4, 6)
            w = random.randint(4, 6)
            area = (h - 2) * (w - 2)
            choices = [g for g in range(1, area) if g not in used_gray]
            if not choices:
                continue
            r = random.randint(0, 18 - h)
            c = random.randint(0, 18 - w)
            if _fits(grid, r, c, h, w):
                g = random.choice(choices)
                used_gray.add(g)
                _draw_container(grid, r, c, h, w, g)
                break
    return grid
