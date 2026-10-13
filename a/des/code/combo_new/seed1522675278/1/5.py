import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SHAPES = {
    "ru": {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    "rd": {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
    "lu": {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)},
    "ld": {(0, 2), (0, 1), (0, 0), (1, 2), (2, 2)},
}
DIRS = {"ru": [(-1, 0), (0, 1)], "rd": [(1, 0), (0, 1)], "lu": [(-1, 0), (0, -1)], "ld": [(1, 0), (0, -1)]}
START = {"ru": (1, 1), "rd": (1, 1), "lu": (1, 1), "ld": (1, 1)}


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


def _kind(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, c0 = min(rs), min(cs)
    rel = {(r - r0, c - c0) for r, c in comp}
    for k, s in SHAPES.items():
        if rel == s:
            return k
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the open-corner direction of each 3x3 L-tower and recolor the first receivers hit by its two rays.

    Input structure:
    - Black background with several monochrome 5-cell L shapes in 3x3 boxes.
    - Grey single-pixel receivers are scattered away from the towers.
    - The L opening points toward one of the four diagonal corners.

    Rule:
    - An opening toward a corner emits one horizontal and one vertical ray in the two cardinal directions of that corner.
    - Rays travel through black cells and stop at the first non-black cell they encounter.
    - A grey receiver on that first-hit cell is recolored; ties are broken by shorter ray length.

    Key insight:
    - The invisible rays are determined entirely by the L opening orientation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    hits = {}
    for comp in _components((g != BLACK) & (g != GREY)):
        color = int(g[comp[0]])
        kind = _kind(comp)
        if kind is None:
            continue
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        sr, sc = START[kind]
        sr += r0
        sc += c0
        for dr, dc in DIRS[kind]:
            rr, cc = sr + dr, sc + dc
            dist = 1
            while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                if g[rr, cc] != BLACK:
                    if g[rr, cc] == GREY and ((rr, cc) not in hits or dist < hits[(rr, cc)][0]):
                        hits[(rr, cc)] = (dist, color)
                    break
                rr += dr
                cc += dc
                dist += 1
    for (r, c), (_, color) in hits.items():
        out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    g = np.zeros((h, w), dtype=int)
    towers = []
    for color in random.sample(COLORS, random.randint(2, 4)):
        for _ in range(100):
            r0 = random.randint(2, h - 5)
            c0 = random.randint(2, w - 5)
            kind = random.choice(list(SHAPES))
            pts = {(r0 + r, c0 + c) for r, c in SHAPES[kind]}
            if any(g[r, c] != BLACK for r, c in pts):
                continue
            for r, c in pts:
                g[r, c] = color
            towers.append((color, kind, r0, c0))
            break
    for color, kind, r0, c0 in towers:
        sr, sc = START[kind]
        sr += r0
        sc += c0
        for dr, dc in DIRS[kind]:
            dist = random.randint(2, 5)
            rr, cc = sr + dr * dist, sc + dc * dist
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == BLACK:
                g[rr, cc] = GREY
    return g
