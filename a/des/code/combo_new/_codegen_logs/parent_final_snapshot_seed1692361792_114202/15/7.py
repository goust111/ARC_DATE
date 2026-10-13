import random
from collections import deque

import numpy as np


BLACK = 0


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
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _target_offsets(anchor, shape):
    h, w = shape
    cr = (h - 1) / 2
    cc = (w - 1) / 2
    r, c = anchor
    if r < cr and c < cc:
        return [(0, 0), (0, 1), (1, 0)]
    if r < cr and c > cc:
        return [(0, 0), (0, -1), (1, 0)]
    if r > cr and c < cc:
        return [(0, 0), (0, 1), (-1, 0)]
    return [(0, 0), (0, -1), (-1, 0)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    h, w = grid.shape
    colors = sorted(set(grid.reshape(-1).tolist()) - {BLACK})
    for color in colors:
        comps = _components(grid == color)
        anchor = next(comp[0] for comp in comps if len(comp) == 1)
        shape_comp = next(comp for comp in comps if len(comp) == 3)
        out[shape_comp[:, 0], shape_comp[:, 1]] = BLACK
        for dr, dc in _target_offsets(tuple(anchor), (h, w)):
            out[anchor[0] + dr, anchor[1] + dc] = color
    return out


def _l_offsets(kind):
    if kind == 0:
        return [(0, 0), (1, 0), (0, 1)]
    if kind == 1:
        return [(0, 0), (-1, 0), (0, 1)]
    if kind == 2:
        return [(0, 0), (1, 0), (0, -1)]
    return [(0, 0), (-1, 0), (0, -1)]


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    while True:
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        random.shuffle(colors)
        count = random.randint(3, 4)
        ok = True
        for color in colors[:count]:
            placed = False
            for _ in range(200):
                ar = random.randint(1, h - 2)
                ac = random.randint(1, w - 2)
                target = [(ar + dr, ac + dc) for dr, dc in _target_offsets((ar, ac), (h, w))]
                if any(not (0 <= r < h and 0 <= c < w) for r, c in target):
                    continue
                if any(occupied[r, c] for r, c in target):
                    continue
                if occupied[ar, ac]:
                    continue
                for _ in range(200):
                    kind = random.randint(0, 3)
                    sr = random.randint(1, h - 2)
                    sc = random.randint(1, w - 2)
                    cells = [(sr + dr, sc + dc) for dr, dc in _l_offsets(kind)]
                    if len({tuple(x) for x in cells}) != 3:
                        continue
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                        continue
                    if any(occupied[r, c] for r, c in cells):
                        continue
                    if min(abs(sr - ar), abs(sc - ac)) <= 1:
                        continue
                    grid[ar, ac] = color
                    occupied[ar, ac] = True
                    for r, c in cells:
                        grid[r, c] = color
                        occupied[r, c] = True
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if ok:
            return grid
