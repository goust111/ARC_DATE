import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


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
    Match external colored fragments to equal-shaped black holes inside the grey template and fill the holes.

    Input structure:
    - One large grey template contains several black holes.
    - Separate colored fragments outside the template are single connected components.
    - Each fragment shape matches exactly one hole by translation only.

    Rule:
    - Find every black hole inside the template.
    - Match each outside fragment to the unique hole with the same normalized shape.
    - Fill the hole with the fragment color and erase the outside fragment.

    Key insight:
    - Color does not help with matching; only exact shape equality does.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    grey = np.argwhere(g == GREY)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    holes = {}
    for comp in _components((g[r0 : r1 + 1, c0 : c1 + 1] == BLACK)):
        global_pts = [(r + r0, c + c0) for r, c in comp]
        if any(r in (r0, r1) or c in (c0, c1) for r, c in global_pts):
            continue
        holes[_norm(global_pts)] = global_pts
    for comp in _components((g != BLACK) & (g != GREY)):
        color = int(g[comp[0]])
        shape = _norm(comp)
        for r, c in comp:
            out[r, c] = BLACK
        for r, c in holes[shape]:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22, 22), dtype=int)
    r0, c0, r1, c1 = 4, 4, 16, 16
    g[r0 : r1 + 1, c0 : c1 + 1] = GREY
    shapes = [[(0, 0), (0, 1)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0)]]
    colors = random.sample(COLORS, len(shapes))
    spots = [(6, 6), (10, 10), (13, 7)]
    for shape, (rr, cc) in zip(shapes, spots):
        for dr, dc in shape:
            g[rr + dr, cc + dc] = BLACK
    outs = [(2, 2), (18, 2), (2, 18)]
    for shape, color, (rr, cc) in zip(shapes, colors, outs):
        for dr, dc in shape:
            g[rr + dr, cc + dc] = color
    return g
