import random
from collections import deque

import numpy as np


def _same_color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, comp))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Map each component area to a centered Chebyshev ring; keep singleton components and color the center from the nearest one."""
    out = grid.copy()
    h, w = grid.shape
    center = (h // 2, w // 2)
    singletons = []
    for color, comp in _same_color_components(grid):
        if len(comp) == 1:
            singletons.append((color, comp[0]))
            continue
        for r, c in comp:
            out[r, c] = 0
        radius = len(comp)
        for r in range(h):
            for c in range(w):
                if max(abs(r - center[0]), abs(c - center[1])) == radius and out[r, c] == 0:
                    out[r, c] = color
    if singletons:
        best_color = min(singletons, key=lambda item: abs(item[1][0] - center[0]) + abs(item[1][1] - center[1]))[0]
        out[center] = best_color
    return out


def _grow_blob(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    while len(cells) < size:
        r, c = rng.choice(cells)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        for nbr in nbrs:
            if nbr not in cells:
                cells.append(nbr)
                break
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def generate_input() -> np.ndarray:
    rng = random.Random()
    while True:
        n = rng.choice([11, 13])
        center = n // 2
        grid = np.zeros((n, n), dtype=int)
        areas = rng.sample([1, 2, 3, 4, 5], k=rng.randint(3, 5))
        colors = rng.sample([1, 2, 3, 4, 7], k=len(areas))
        placed: list[set[tuple[int, int]]] = []
        singles = []
        ok = True
        for area, color in zip(areas, colors):
            shape = [(0, 0)] if area == 1 else _grow_blob(rng, area)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            spots = [(r, c) for r in range(n - max_r) for c in range(n - max_c)]
            rng.shuffle(spots)
            chosen = None
            for sr, sc in spots:
                cells = {(sr + r, sc + c) for r, c in shape}
                if center in {p[0] for p in cells} and center in {p[1] for p in cells}:
                    pass
                if (center, center) in cells:
                    continue
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in cells for comp in placed for r, c in comp):
                    continue
                chosen = cells
                break
            if chosen is None:
                ok = False
                break
            placed.append(chosen)
            if area == 1:
                singles.append(next(iter(chosen)))
            for r, c in chosen:
                grid[r, c] = color
        if not ok:
            continue
        if len(singles) > 1:
            dists = [abs(r - center) + abs(c - center) for r, c in singles]
            if len(dists) != len(set(dists)):
                continue
        return grid
