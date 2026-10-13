import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    wall = 5
    chambers = []
    for comp in _components(grid != wall):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            chambers.append(comp)
    steps = len(chambers) - 1
    out = grid.copy()
    for comp in chambers:
        cells = set(comp)
        colored = [(r, c) for r, c in comp if grid[r, c] not in (0, wall)]
        center = None
        for r, c in colored:
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if all(p in cells and grid[p] not in (0, wall) for p in neighbors):
                center = (r, c)
                break
        if center is None:
            continue
        cr, cc = center
        center_color = int(grid[cr, cc])
        arms = {
            (-1, 0): int(grid[cr - 1, cc]),
            (1, 0): int(grid[cr + 1, cc]),
            (0, -1): int(grid[cr, cc - 1]),
            (0, 1): int(grid[cr, cc + 1]),
        }
        for (dr, dc), color in arms.items():
            for step in range(1, steps + 2):
                rr, cc2 = cr + dr * step, cc + dc * step
                if (rr, cc2) not in cells:
                    break
                out[rr, cc2] = color
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            for step in range(1, steps + 1):
                rr, cc2 = cr + dr * step, cc + dc * step
                if (rr, cc2) not in cells:
                    break
                out[rr, cc2] = center_color
        for r, c in comp:
            if out[r, c] == 0:
                out[r, c] = center_color
    return out


def _draw_room(grid, top, left, inner, color):
    bottom = top + inner + 1
    right = left + inner + 1
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    grid = np.zeros((29, 29), dtype=int)
    placements = [(1, 1), (1, 10), (1, 19), (15, 1), (15, 10)]
    n = rng.choice([3, 4, 5])
    center_color = 4
    arm_palette = [1, 2, 3, 6, 7, 8, 9]
    for top, left in placements[:n]:
        inner = 7
        _draw_room(grid, top, left, inner, 5)
        cr = top + 1 + inner // 2
        cc = left + 1 + inner // 2
        arms = rng.sample(arm_palette, 4)
        grid[cr, cc] = center_color
        grid[cr - 1, cc] = arms[0]
        grid[cr + 1, cc] = arms[1]
        grid[cr, cc - 1] = arms[2]
        grid[cr, cc + 1] = arms[3]
    return grid
