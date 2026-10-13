import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
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
            comps.append(comp)
    return comps


def _wall_rows(grid):
    return [r for r in range(grid.shape[0]) if np.count_nonzero(grid[r] == 5) >= grid.shape[1] // 2]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    walls = _wall_rows(grid)
    bounds = [-1] + walls + [grid.shape[0]]
    corridors = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    top0, top1 = corridors[0]
    red_top = grid[top0:top1] == 2
    comps = _components(red_top)
    template_comp = max(comps, key=len)
    rs = [p[0] for p in template_comp]
    cs = [p[1] for p in template_comp]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    template = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in template_comp:
        template[r - r0, c - c0] = 2
    center = (template.shape[0] // 2, template.shape[1] // 2)
    for k, (a, b) in enumerate(corridors[1:], start=1):
        cells = np.argwhere(grid[a:b] == 2)
        marker_cells = []
        for rr, cc in cells:
            gr = a + rr
            if all(grid[nr, nc] != 2 for nr, nc in ((gr - 1, cc), (gr + 1, cc), (gr, cc - 1), (gr, cc + 1))
                   if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                marker_cells.append((gr, cc))
        for mr, mc in marker_cells:
            stamp = np.rot90(template, -k)
            for r in range(stamp.shape[0]):
                for c in range(stamp.shape[1]):
                    if stamp[r, c] == 0:
                        continue
                    tr = mr + r - center[0]
                    tc = mc + c - center[1]
                    if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1] and out[tr, tc] != 5:
                        out[tr, tc] = 2
            out[mr, mc] = int(stamp[center])
    return out


def generate_input() -> np.ndarray:
    corridors = random.randint(4, 5)
    ch = random.choice([3, 4])
    h = corridors * ch + corridors - 1
    w = random.randint(14, 18)
    grid = np.zeros((h, w), dtype=int)
    door_centers = random.sample(range(3, w - 3), corridors - 1)
    for i, dc in enumerate(door_centers, start=1):
        row = i * ch + (i - 1)
        grid[row, :] = 5
        grid[row, dc - 1:dc + 2] = 0
    template = np.array([[2, 2, 0], [0, 2, 0], [0, 2, 2]], dtype=int)
    tr = random.randint(0, ch - 3)
    tc = random.randint(1, w - 5)
    grid[tr:tr + 3, tc:tc + 3] = np.maximum(grid[tr:tr + 3, tc:tc + 3], template)
    for level in range(1, corridors):
        a = level * ch + level
        b = a + ch
        bad = {door_centers[level - 1]}
        mr = random.randint(a, b - 1)
        choices = [c for c in range(2, w - 2) if c not in bad]
        mc = random.choice(choices)
        grid[mr, mc] = 2
    return grid
