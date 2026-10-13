import random
from collections import Counter, deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each country's border pixels with the unique capital color embedded inside that country.

    Input structure:
    - The grid is tiled by large monochrome country regions.
    - Each country contains exactly one singleton capital pixel in a different color.
    - Capital colors are unique and never serve as country colors.

    Rule:
    - For each capital, infer its host country from the surrounding majority color.
    - Find the host country component and recolor its boundary pixels to the capital color.
    - Keep the capital itself and all interior non-boundary country pixels unchanged.

    Key insight:
    - The singleton off-color pixel defines a country's new border color, but only after identifying which surrounding region owns that capital.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for color in [v for v in np.unique(g) if v != BG]:
        pts = list(zip(*np.where(g == int(color)), strict=False))
        if len(pts) != 1:
            continue
        r, c = pts[0]
        neigh = [int(g[r + 1, c]), int(g[r - 1, c]), int(g[r, c + 1]), int(g[r, c - 1])]
        host = Counter(v for v in neigh if v != BG and v != int(color)).most_common(1)[0][0]
        for cells in _components(g == host):
            if sum((abs(x - r) + abs(y - c) == 1) for x, y in cells) == 0:
                continue
            cellset = set(cells)
            for x, y in cells:
                if any((x + dx, y + dy) not in cellset for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= x + dx < g.shape[0] and 0 <= y + dy < g.shape[1]):
                    out[x, y] = color
            break
    return out


def generate_input() -> np.ndarray:
    rows = 2
    cols = 2
    hs = [random.randint(4, 7) for _ in range(rows)]
    ws = [random.randint(4, 7) for _ in range(cols)]
    h, w = sum(hs), sum(ws)
    if max(h, w) > 30:
        return generate_input()
    g = np.zeros((h, w), dtype=int)
    base_colors = random.sample([1, 2, 3, 4], rows * cols)
    cap_colors = random.sample([6, 7, 8, 9], rows * cols)
    rr = 0
    idx = 0
    for rh in hs:
        cc = 0
        for cw in ws:
            g[rr : rr + rh, cc : cc + cw] = base_colors[idx]
            cr = rr + random.randint(1, rh - 2)
            cc2 = cc + random.randint(1, cw - 2)
            g[cr, cc2] = cap_colors[idx]
            cc += cw
            idx += 1
        rr += rh
    return g
