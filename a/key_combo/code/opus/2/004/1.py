import random
from collections import deque
import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            col = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == col:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append({"color": col, "cells": cells})
    return comps


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells), (r0, c0), (max(r for r, _ in cells) - r0 + 1, max(c for _, c in cells) - c0 + 1)


def _holes(grid, comp):
    cells = set(comp["cells"])
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    q = deque()
    seen = set()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if (r, c) not in cells:
                q.append((r, c)); seen.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if (r, c) not in cells and (r, c) not in seen:
                q.append((r, c)); seen.add((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in cells and (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    inside = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in cells and (r, c) not in seen]
    if not inside:
        return []
    # Generated structures contain one interior cavity; this also handles multiple
    # disconnected cavities if hand-authored inputs include them.
    temp = np.zeros(grid.shape, dtype=int)
    for p in inside:
        temp[p] = 9
    return [{"owner": comp, "cells": x["cells"]} for x in _components(temp)]


def _complement(a, b, da, db):
    if da != db:
        return False
    full = {(r, c) for r in range(da[0]) for c in range(da[1])}
    return set(a).isdisjoint(b) and set(a) | set(b) == full


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    comps = _components(g)
    structures = [c for c in comps if len(c["cells"]) > 10]
    movers = [c for c in comps if len(c["cells"]) <= 8]
    holes = []
    for s in structures:
        holes.extend(_holes(g, s))
    out = g.copy()
    used = set()
    for hole in holes:
        hshape, htop, hdims = _norm(hole["cells"])
        match = None
        for mov in movers:
            if id(mov) in used:
                continue
            mshape, _, mdims = _norm(mov["cells"])
            if _complement(mshape, hshape, mdims, hdims):
                match = mov
                break
        if match is None:
            continue
        used.add(id(match))
        mshape, _, _ = _norm(match["cells"])
        for r, c in match["cells"]:
            out[r, c] = 0
        for dr, dc in mshape:
            out[htop[0] + dr, htop[1] + dc] = match["color"]
    return out


def _solid_with_hole(g, top, left, hole_shape, hole_dims):
    g[top:top + 5, left:left + 5] = 1
    hr, hc = top + 2, left + 1
    for r, c in hole_shape:
        g[hr + r, hc + c] = 0
    return hr, hc


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    pairs = [
        ([(0, 0), (0, 1), (1, 0)], (2, 3)),
        ([(0, 1), (1, 0), (1, 1)], (2, 3)),
        ([(0, 0), (1, 0), (1, 1)], (2, 2)),
        ([(0, 1), (1, 0), (2, 1)], (3, 2)),
    ]
    slots = [(1, 2, "right"), (1, 13, "left"), (11, 2, "right"), (11, 13, "left")]
    n = random.randint(2, 4)
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], n)
    random.shuffle(pairs)
    for i in range(n):
        top, left, side = slots[i]
        hshape, dims = pairs[i]
        hr, hc = _solid_with_hole(g, top, left, hshape, dims)
        full = {(r, c) for r in range(dims[0]) for c in range(dims[1])}
        comp = sorted(full - set(hshape))
        mleft = left + 7 if side == "right" else left - 5
        mtop = hr
        for r, c in comp:
            g[mtop + r, mleft + c] = colors[i]
    return g
