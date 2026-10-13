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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the number of marker pixels inside each open container to select the external block, place it inside, and extend a same-colored tail outward.

    Input structure:
    - Grey hollow containers each have one border opening.
    - Inside each container are 1-3 colored singleton markers.
    - Outside the containers are solid monochrome rectangles matching those marker colors.

    Rule:
    - The marker count N selects the unique same-color external block whose area is N.
    - That block is copied into the container interior aligned to the interior top-left.
    - A tail of length N is extended through the container opening; the original markers and external blocks are erased.

    Key insight:
    - The marker count controls both which block to choose and how long the outward tail becomes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    blocks = {}
    for comp in _components((g != BLACK) & (g != GREY)):
        color = int(g[comp[0]])
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        if len(comp) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) and len(comp) > 1:
            blocks[color] = comp
    for comp in _components(g == GREY):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        markers = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] not in (BLACK, GREY)]
        if not markers:
            continue
        n = len(markers)
        colors = {int(g[r, c]) for r, c in markers}
        color = next(c for c in colors if len(blocks.get(c, [])) == n)
        block = blocks[color]
        br0 = min(r for r, _ in block)
        bc0 = min(c for _, c in block)
        norm = [(r - br0, c - bc0) for r, c in block]
        for r, c in block + markers:
            out[r, c] = BLACK
        ir0, ic0 = r0 + 1, c0 + 1
        for dr, dc in norm:
            out[ir0 + dr, ic0 + dc] = color
        side = None
        if any(g[r0, c] == BLACK for c in range(c0 + 1, c1)):
            side = "top"
        elif any(g[r1, c] == BLACK for c in range(c0 + 1, c1)):
            side = "bottom"
        elif any(g[r, c0] == BLACK for r in range(r0 + 1, r1)):
            side = "left"
        else:
            side = "right"
        bh = max(r for r, _ in norm) + 1
        bw = max(c for _, c in norm) + 1
        if side == "top":
            out[max(0, r0 - n) : r0, ic0 : ic0 + bw] = color
        elif side == "bottom":
            out[r1 + 1 : min(out.shape[0], r1 + 1 + n), ic0 : ic0 + bw] = color
        elif side == "left":
            out[ir0 : ir0 + bh, max(0, c0 - n) : c0] = color
        else:
            out[ir0 : ir0 + bh, c1 + 1 : min(out.shape[1], c1 + 1 + n)] = color
    out[g == GREY] = GREY
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22, 22), dtype=int)
    r0, c0, r1, c1 = 8, 8, 14, 15
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        g[r0, 10:12] = BLACK
    elif side == "bottom":
        g[r1, 10:12] = BLACK
    elif side == "left":
        g[10:12, c0] = BLACK
    else:
        g[10:12, c1] = BLACK
    cols = random.sample(COLORS, 3)
    for c, p in zip(cols, [(9, 9), (10, 11), (12, 13)]):
        g[p] = c
    specs = {cols[0]: (1, 3), cols[1]: (1, 2), cols[2]: (2, 2)}
    for c, (hh, ww), pos in zip(cols, [specs[cols[0]], specs[cols[1]], specs[cols[2]]], [(1, 1), (1, 5), (4, 1)]):
        rr, cc = pos
        g[rr : rr + hh, cc : cc + ww] = c
    return g
