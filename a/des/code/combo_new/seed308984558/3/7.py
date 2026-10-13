import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PRIORITY = {1: 3, 2: 2, 3: 1}


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
    Cast edge rays toward the interior and recolor each grey obstacle by the highest-priority ray that hits it.

    Input structure:
    - A black background contains grey obstacle objects and colored edge markers.
    - A three-pixel corner legend lists red, blue, green in priority order.
    - Markers launch rays inward from their own edge.

    Rule:
    - Erase the corner legend.
    - Extend each ray until the next step would enter a grey object or leave the grid.
    - Recolor each grey object to the highest-priority color among the rays that stop adjacent to it; untouched objects stay grey.

    Key insight:
    - You must combine path tracing from the edge with the separate priority rule encoded by the legend.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    out[0:3, 0] = BG
    hits = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color not in PRIORITY:
                continue
            if (r, c) in [(0, 0), (1, 0), (2, 0)]:
                continue
            if r == 0:
                dr, dc = 1, 0
            elif r == g.shape[0] - 1:
                dr, dc = -1, 0
            elif c == 0:
                dr, dc = 0, 1
            elif c == g.shape[1] - 1:
                dr, dc = 0, -1
            else:
                continue
            x, y = r + dr, c + dc
            while 0 <= x < g.shape[0] and 0 <= y < g.shape[1] and g[x, y] == BG:
                out[x, y] = color
                x += dr
                y += dc
            if 0 <= x < g.shape[0] and 0 <= y < g.shape[1] and g[x, y] == GREY:
                hits.setdefault((x, y), []).append(color)
    for cells in _components(g == GREY):
        hit_colors = []
        for r, c in cells:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                hit_colors += hits.get((r + dx, c + dy), [])
        if hit_colors:
            best = max(set(hit_colors), key=lambda c: PRIORITY[c])
            for r, c in cells:
                out[r, c] = best
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    g[0, 0], g[1, 0], g[2, 0] = 1, 2, 3
    for _ in range(random.randint(2, 4)):
        rh, rw = random.randint(2, 4), random.randint(2, 4)
        r0 = random.randint(4, h - rh - 2)
        c0 = random.randint(2, w - rw - 2)
        if np.any(g[r0 : r0 + rh, c0 : c0 + rw]):
            continue
        g[r0 : r0 + rh, c0 : c0 + rw] = GREY
    for color in random.choices([1, 2, 3], k=random.randint(3, 6)):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            r = 0 if side == "top" else h - 1
            c = random.randint(1, w - 2)
        else:
            c = 0 if side == "left" else w - 1
            r = random.randint(3, h - 2)
        g[r, c] = color
    return g
