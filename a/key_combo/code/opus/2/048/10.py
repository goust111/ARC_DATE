from __future__ import annotations

import numpy as np
from collections import deque
import random

BLACK = 0
RED = 2
BLUE = 1
GREY = 5


def _neighbors4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _neighbors8(r: int, c: int, H: int, W: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc


def _components(mask: np.ndarray, eight=False):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in (_neighbors8(r, c, H, W) if eight else _neighbors4(r, c, H, W)):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.array(grid, copy=True)

    # Fill gray outlines row by row
    gray_mask = out == GREY
    comps = _components(gray_mask, eight=True)
    for comp in comps:
        color = GREY
        rows = sorted(set(int(r) for r in comp[:, 0]))
        for r in rows:
            cols = [int(c) for rr, c in comp if int(rr) == r]
            if len(cols) < 2:
                continue
            c0 = min(cols)
            c1 = max(cols)
            out[r, c0 : c1 + 1] = color

    # connected gray regions after fill
    region_mask = out == GREY
    reg_comps = _components(region_mask, eight=False)
    if not reg_comps:
        return out
    areas = [len(c) for c in reg_comps]
    for comp in reg_comps:
        rs = comp[:, 0]
        cs = comp[:, 1]
        cr = int(np.floor(rs.mean()))
        cc = int(np.floor(cs.mean()))  # not used for decision but keep for completeness
        d_top = cr
        d_bottom = (H - 1) - cr
        if d_top < d_bottom:
            out[rs, cs] = RED
        elif d_bottom < d_top:
            out[rs, cs] = BLUE

    return out


def _draw_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for r in range(r0, r1 + 1):
        grid[r, c0] = GREY
        grid[r, c1] = GREY


def generate_input() -> np.ndarray:
    out = np.zeros((16, 16), dtype=int)
    out[0, :] = RED
    out[15, :] = BLUE
    _draw_outline(out, 2, 2, 10, 12)
    _draw_outline(out, 3, 3, 12, 6)
    _draw_outline(out, 6, 8, 14, 13)
    for _ in range(random.randint(1, 3)):
        r = random.randint(1, 14)
        c = random.randint(0, 15)
        if out[r, c] == BLACK:
            out[r, c] = GREY
    return out
