import random
from collections import deque

import numpy as np


BG = 0
SEED = 7


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    comps = _components(grid != BG)
    order = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        order.append((min(rows), min(cols), comp))
    out = np.zeros_like(grid)
    for _, _, comp in sorted(order):
        cells = set(comp)
        seeds = [cell for cell in comp if grid[cell] == SEED]
        colors = [int(grid[cell]) for cell in comp if grid[cell] not in (BG, SEED)]
        host = max(set(colors), key=colors.count)
        if seeds:
            expanded = set(cells)
            for r, c in list(cells):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        expanded.add((nr, nc))
            for r, c in expanded:
                if out[r, c] == BG:
                    out[r, c] = SEED if (r, c) not in cells else (SEED if grid[r, c] == SEED else host)
        else:
            for r, c in cells:
                if all((r + dr, c + dc) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    if out[r, c] == BG:
                        out[r, c] = host
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 8, 9], random.randint(3, 5))
    for color in colors:
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        arr = np.full((h, w), color, dtype=int)
        with_seed = random.choice([True, False])
        if with_seed:
            for _ in range(random.randint(1, 2)):
                arr[random.randint(1, h - 2), random.randint(1, w - 2)] = SEED
        for _ in range(300):
            r = random.randint(1, 23 - h)
            c = random.randint(1, 23 - w)
            rr0, rr1 = max(0, r - 2), min(24, r + h + 2)
            cc0, cc1 = max(0, c - 2), min(24, c + w + 2)
            if np.any(used[rr0:rr1, cc0:cc1]):
                continue
            grid[r : r + h, c : c + w] = arr
            used[rr0:rr1, cc0:cc1] = True
            break
    return grid
