import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    gray = np.argwhere(grid == 5)
    r0, c0 = gray.min(axis=0)
    r1, c1 = gray.max(axis=0)
    cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
    markers = [(int(r), int(c)) for r, c in zip(*np.where(grid == 9))]
    h_marker = max(markers, key=lambda p: abs(p[1] - cc))
    v_marker = max(markers, key=lambda p: abs(p[0] - cr))
    hbit = 0 if h_marker[1] < cc else 1
    vbit = 0 if v_marker[0] < cr else 1
    tiles = {}
    mask = (grid != 0) & (grid != 5) & (grid != 9)
    for comp in _components(mask):
        pts = np.array(comp)
        tr0, tc0 = pts.min(axis=0)
        tr1, tc1 = pts.max(axis=0)
        trc, tcc = pts.mean(axis=0)
        key = (0 if trc < cr else 1, 0 if tcc < cc else 1)
        tiles[key] = grid[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    tile = tiles[(vbit, hbit)]
    for r in range(int(r0) + 1, int(r1)):
        for c in range(int(c0) + 1, int(c1)):
            out[r, c] = tile[(r - r0 - 1) % tile.shape[0], (c - c0 - 1) % tile.shape[1]]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    r0, r1, c0, c1 = 5, 14, 5, 14
    grid[r0, c0 : c1 + 1] = grid[r1, c0 : c1 + 1] = 5
    grid[r0 : r1 + 1, c0] = grid[r0 : r1 + 1, c1] = 5
    positions = {(0, 0): (1, 1), (0, 1): (1, 14), (1, 0): (14, 1), (1, 1): (14, 14)}
    palettes = [[1, 2], [3, 4], [6, 7], [8, 9]]
    random.shuffle(palettes)
    for key, (tr, tc), colors in zip(positions.keys(), positions.values(), palettes):
        th, tw = random.choice([(2, 2), (2, 3), (3, 2)])
        for r in range(th):
            for c in range(tw):
                grid[tr + r, tc + c] = colors[(r + c) % 2]
    choice = random.choice(list(positions))
    vbit, hbit = choice
    grid[random.choice([3, 15]), 3 if hbit == 0 else 15] = 9
    grid[3 if vbit == 0 else 15, random.choice([7, 10])] = 9
    return grid
