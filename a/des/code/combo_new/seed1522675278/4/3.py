import random

import numpy as np


BLUE = 1
GREY = 5
COLORS = [2, 3, 4, 6, 7, 8, 9]


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


def _norm(pts):
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((r - r0, c - c0) for r, c in pts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each object's enclosed blue hole, match it to the equal-shaped edge notch, and translate the whole object onto that notch.

    Input structure:
    - Blue background with a grey border around the grid edges.
    - Each colored object contains one fully enclosed blue hole.
    - Each border notch is a blue edge-connected component matching one hole shape.

    Rule:
    - Extract the normalized hole shape from every object.
    - Match it to the unique equal-shaped blue notch component on the border.
    - Translate the object so its hole covers that notch exactly, erasing the original object.

    Key insight:
    - The object's identity comes from its negative-space hole, not its outer silhouette.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    notches = [c for c in _components(g == BLUE) if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in c)]
    notch_map = {_norm(c): c for c in notches}
    for color in [c for c in np.unique(g) if c not in (BLUE, GREY)]:
        for comp in _components(g == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            holes = [h for h in _components(g[r0 : r1 + 1, c0 : c1 + 1] == BLUE) if all(0 < r < r1 - r0 and 0 < c < c1 - c0 for r, c in h)]
            if not holes:
                continue
            hole = [(r + r0, c + c0) for r, c in holes[0]]
            target = notch_map[_norm(hole)]
            dr = min(r for r, _ in target) - min(r for r, _ in hole)
            dc = min(c for _, c in target) - min(c for _, c in hole)
            for r, c in comp:
                out[r, c] = BLUE
            for r, c in comp:
                out[r + dr, c + dc] = color
    return out


def generate_input() -> np.ndarray:
    g = np.full((20, 20), BLUE, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = GREY
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        notch = [(0, 6), (1, 6)]
    elif side == "bottom":
        notch = [(19, 12), (18, 12)]
    elif side == "left":
        notch = [(10, 0), (10, 1)]
    else:
        notch = [(8, 19), (8, 18)]
    for r, c in notch:
        g[r, c] = BLUE
    color = random.choice(COLORS)
    obj = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
    base = (random.randint(8, 13), random.randint(8, 13))
    for r, c in obj:
        g[base[0] + r, base[1] + c] = color
    g[base[0], base[1]] = BLUE
    g[base[0] + 1, base[1]] = BLUE
    return g
