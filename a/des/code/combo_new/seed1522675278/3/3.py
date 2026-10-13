import random

import numpy as np


BLACK = 0
GREEN = 3
AZURE = 8
RED = 2


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-wall windows inside each green frame and draw an outward red branch from the corresponding wall midpoint.

    Input structure:
    - Green rectangular borders enclose black interiors.
    - Azure window pixels sit one cell inside exactly one wall per rectangle.
    - Window count may vary by rectangle.

    Rule:
    - The wall touched by the windows determines branch direction.
    - The number of windows determines branch length.
    - Draw a red line outward from the midpoint of that wall, clipped by the grid edge.

    Key insight:
    - Window orientation provides the vector, and window count provides the magnitude.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in _components(g == GREEN):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        wins = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] == AZURE]
        if not wins:
            continue
        if all(r == r0 + 1 for r, _ in wins):
            start = (r0 - 1, (c0 + c1) // 2)
            vec = (-1, 0)
        elif all(r == r1 - 1 for r, _ in wins):
            start = (r1 + 1, (c0 + c1) // 2)
            vec = (1, 0)
        elif all(c == c0 + 1 for _, c in wins):
            start = ((r0 + r1) // 2, c0 - 1)
            vec = (0, -1)
        else:
            start = ((r0 + r1) // 2, c1 + 1)
            vec = (0, 1)
        rr, cc = start
        for _ in range(len(wins)):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = RED
            rr += vec[0]
            cc += vec[1]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    for box in [(3, 3, 9, 9), (8, 11, 15, 16)]:
        r0, c0, r1, c1 = box
        g[r0, c0 : c1 + 1] = GREEN
        g[r1, c0 : c1 + 1] = GREEN
        g[r0 : r1 + 1, c0] = GREEN
        g[r0 : r1 + 1, c1] = GREEN
        side = random.choice(["top", "bottom", "left", "right"])
        n = random.randint(1, 3)
        if side in ("top", "bottom"):
            row = r0 + 1 if side == "top" else r1 - 1
            for c in random.sample(range(c0 + 1, c1), n):
                g[row, c] = AZURE
        else:
            col = c0 + 1 if side == "left" else c1 - 1
            for r in random.sample(range(r0 + 1, r1), n):
                g[r, col] = AZURE
    return g
