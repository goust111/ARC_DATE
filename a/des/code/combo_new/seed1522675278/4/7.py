import random

import numpy as np


BLACK = 0
GREY = 5
WALL = 9
ARROWS = [1, 2, 3, 4, 6, 7, 8]
SHAPES = {"up": {(1, 0), (1, 1), (1, 2), (2, 1)}, "down": {(0, 1), (1, 0), (1, 1), (1, 2)}, "left": {(0, 1), (1, 0), (1, 1), (2, 1)}, "right": {(0, 1), (1, 1), (1, 2), (2, 1)}}


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
    for k, v in SHAPES.items():
        if rel == v:
            return k
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each colored arrow along its row or column until a maroon wall, and recolor every grey object it reaches using nearest-arrow priority.

    Input structure:
    - Grey solid rectangles are target objects on black background.
    - Maroon one-cell wall segments block line-of-sight.
    - Colored arrow components encode one of four scan directions.

    Rule:
    - For each arrow, scan forward on its row or column until a maroon wall.
    - Any grey object intersecting that scan line becomes a candidate for recoloring by the arrow color.
    - If multiple arrows reach the same object, use the one with the shortest forward distance.

    Key insight:
    - The arrow direction comes from the missing arm of the T-shape, and walls cut off propagation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objs = []
    for comp in _components(g == GREY):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        objs.append((comp, (min(rs), min(cs), max(rs), max(cs))))
    best = {}
    for comp in _components((g != BLACK) & (g != GREY) & (g != WALL)):
        color = int(g[comp[0]])
        kind = _kind(comp)
        if kind is None:
            continue
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        cr, cc = r0 + 1, c0 + 1
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[kind]
        rr, cc2, dist = cr + dr, cc + dc, 1
        while 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1] and g[rr, cc2] != WALL:
            for idx, (_, (a, b, c, d)) in enumerate(objs):
                if (dr and b <= cc2 <= d and a <= rr <= c) or (dc and a <= rr <= c and b <= cc2 <= d):
                    if idx not in best or dist < best[idx][0]:
                        best[idx] = (dist, color)
            rr += dr
            cc2 += dc
            dist += 1
    for idx, (_, color) in best.items():
        for r, c in objs[idx][0]:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    g[5:8, 10] = WALL
    g[12, 3:8] = WALL
    g[4:7, 14:17] = GREY
    g[10:13, 5:8] = GREY
    for color, kind, pos in zip(random.sample(ARROWS, 2), ["right", "up"], [(4, 2), (16, 6)]):
        r0, c0 = pos
        for dr, dc in SHAPES[kind]:
            g[r0 + dr, c0 + dc] = color
    return g
