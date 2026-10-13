import random
from collections import deque

import numpy as np


BG = 0
MIX = 8


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


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the two original overlapping rectangles from their pure-color parts and mixed overlap, then draw only outlines.

    Input structure:
    - A black background contains two solid rectangles in different colors with a third mix color on their overlap.
    - No triple overlap exists.
    - Each pure-color region still exposes part of its original rectangle.

    Rule:
    - Find the mixed-overlap component and the two pure colors touching it.
    - Reconstruct each rectangle as the bbox of its pure region union the mixed region.
    - Clear the grid and draw only the rectangle outlines, larger first and smaller last.

    Key insight:
    - The overlap color is an evidence region shared by both rectangles, so each full rectangle is recovered by combining that overlap with its visible pure remainder.
    """
    g = np.array(grid, dtype=int)
    out = np.zeros_like(g)
    mixed = max(_components(g == MIX), key=len)
    neigh = set()
    for r, c in mixed:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] not in (BG, MIX):
                neigh.add(int(g[rr, cc]))
    rects = []
    for color in sorted(neigh):
        pure = list(zip(*np.where(g == color), strict=False))
        rect_cells = pure + mixed
        r0, c0, r1, c1 = _bbox(rect_cells)
        rects.append(((r1 - r0 + 1) * (c1 - c0 + 1), color, r0, c0, r1, c1))
    rects.sort(reverse=True)
    for _, color, r0, c0, r1, c1 in rects:
        out[r0 : r1 + 1, c0] = out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = out[r1, c0 : c1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    a, b = random.sample([1, 2, 3, 4, 6, 7], 2)
    r0, c0 = random.randint(1, 8), random.randint(1, 8)
    r1, c1 = r0 + random.randint(5, 8), c0 + random.randint(5, 8)
    s0, t0 = r0 + random.randint(1, 3), c0 + random.randint(1, 3)
    s1, t1 = s0 + random.randint(5, 8), t0 + random.randint(5, 8)
    if s1 >= h or t1 >= w:
        return generate_input()
    g[r0:r1, c0:c1] = a
    g[s0:s1, t0:t1] = np.where(g[s0:s1, t0:t1] == a, MIX, b)
    g[(g == a) & (np.indices(g.shape)[0] >= s0) & (np.indices(g.shape)[0] < s1) & (np.indices(g.shape)[1] >= t0) & (np.indices(g.shape)[1] < t1)] = MIX
    return g
