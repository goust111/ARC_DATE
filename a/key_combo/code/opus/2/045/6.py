import random
from collections import deque

import numpy as np

BG = 0
GRAY = 5
MISSING = 9
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dr, dc in DIRS4:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _find_rings(grid: np.ndarray):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    rings = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if visited[r, c] or grid[r, c] != BG:
                continue
            nbrs = [
                (r - 1, c - 1),
                (r - 1, c),
                (r - 1, c + 1),
                (r, c + 1),
                (r + 1, c + 1),
                (r + 1, c),
                (r + 1, c - 1),
                (r, c - 1),
            ]
            if all(0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG for nr, nc in nbrs):
                rings.append((r, c))
                visited[r, c] = True
    return rings


def _rotate(values, k: int):
    n = len(values)
    k %= n
    return [values[(i + k) % n] for i in range(n)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    rings = _find_rings(g)
    out = g.copy()

    for r, c in rings:
        corners = [
            (r - 1, c - 1),
            (r - 1, c + 1),
            (r + 1, c + 1),
            (r + 1, c - 1),
        ]
        sides = [
            (r - 1, c),
            (r, c - 1),
            (r + 1, c),
            (r, c + 1),
        ]
        corner_colors = [int(g[rr, cc]) for rr, cc in corners]
        side_colors = [int(g[rr, cc]) for rr, cc in sides]

        miss = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if max(abs(dr), abs(dc)) <= 1:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == MISSING:
                    miss += 1

        new_corners = _rotate(corner_colors, miss)
        new_sides = _rotate(side_colors, miss)

        for (rr, cc), v in zip(corners, new_corners):
            out[rr, cc] = v
        for (rr, cc), v in zip(sides, new_sides):
            out[rr, cc] = v

    out[out == MISSING] = BG
    return out


def _draw_ring(grid: np.ndarray, center: tuple[int, int], colors: list[int]):
    r, c = center
    nbrs = [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c + 1),
        (r + 1, c + 1),
        (r + 1, c),
        (r + 1, c - 1),
        (r, c - 1),
    ]
    for (rr, cc) in nbrs:
        grid[rr, cc] = random.choice(colors)


def generate_input() -> np.ndarray:
    h, w = 15, 15
    g = np.zeros((h, w), dtype=int)

    color_pool = [1, 2, 3, 4, 7]
    n = random.randint(3, 4)
    centers: list[tuple[int, int]] = []
    for _ in range(200):
        if len(centers) >= n:
            break
        cand = (random.randint(2, h - 3), random.randint(2, w - 3))
        if all(max(abs(cand[0] - cr), abs(cand[1] - cc)) >= 6 for cr, cc in centers):
            centers.append(cand)

    for center in centers:
        _draw_ring(g, center, color_pool)

    for r, c in centers:
        candidates = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr = r + dr
                cc = c + dc
                if max(abs(dr), abs(dc)) <= 1:
                    continue
                if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == BG:
                    candidates.append((rr, cc))
        random.shuffle(candidates)
        for rr, cc in candidates[: random.randint(1, 3)]:
            g[rr, cc] = MISSING

    return g
