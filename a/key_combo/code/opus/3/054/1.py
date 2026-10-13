import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append({"color": color, "pts": pts})
    return out


def _round_nearest(x):
    return int(np.floor(float(x) + 0.5))


def _decorate(comp):
    pts = np.array(comp["pts"], dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    local = pts - np.array([r0, c0])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    comp.update(
        {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "mask": mask,
            "symmetric": bool(np.array_equal(mask, np.fliplr(mask))),
            "row_mass": float(local[:, 0].mean()),
            "col_mass": float(local[:, 1].mean()),
        }
    )
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """Cast colored rays from left-right asymmetric objects and recolor symmetric blockers they hit."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    comps = [_decorate(c) for c in _components(g)]
    blockers = [c for c in comps if c["symmetric"]]
    sources = sorted((c for c in comps if not c["symmetric"]), key=lambda c: (c["bbox"][0], c["bbox"][1]))

    blocker_at = {}
    for i, comp in enumerate(blockers):
        for pt in comp["pts"]:
            blocker_at[pt] = i

    for comp in sources:
        r0, c0, r1, c1 = comp["bbox"]
        center_col = (c1 - c0) / 2.0
        if comp["col_mass"] > center_col:
            direction = (0, 1)
            rr = r0 + _round_nearest(comp["row_mass"])
            cc = c1 + 1
        elif comp["col_mass"] < center_col:
            direction = (0, -1)
            rr = r0 + _round_nearest(comp["row_mass"])
            cc = c0 - 1
        else:
            direction = (-1, 0)
            rr = r0 - 1
            cc = c0 + _round_nearest(comp["col_mass"])

        dr, dc = direction
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            bid = blocker_at.get((rr, cc))
            if bid is not None:
                for br, bc in blockers[bid]["pts"]:
                    out[br, bc] = comp["color"]
                break
            if g[rr, cc] == 0:
                out[rr, cc] = comp["color"]
            rr += dr
            cc += dc
    return out


def _paint(grid, top, left, coords, color):
    for r, c in coords:
        grid[top + r, left + c] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6], 5)
    jitter = random.choice([-1, 0, 1])
    sym_cross = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    sym_square = [(0, 0), (0, 1), (1, 0), (1, 1)]
    right_src = [(0, 1), (1, 1), (1, 2), (2, 2)]
    left_src = [(0, 1), (1, 1), (1, 0), (2, 0)]
    up_src = [(0, 0), (0, 2), (1, 1), (2, 1)]

    _paint(grid, 4 + jitter, 2, right_src, colors[0])
    _paint(grid, 4 + jitter, 11, sym_cross, colors[1])
    _paint(grid, 11 - jitter, 13, left_src, colors[2])
    _paint(grid, 11 - jitter, 5, sym_cross, colors[3])
    if random.random() < 0.7:
        _paint(grid, 3, 7, sym_square, colors[4])
    else:
        _paint(grid, 10, 9, up_src, colors[4])
        _paint(grid, 4, 10, sym_square, colors[1])
    return grid
