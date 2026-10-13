import random
from collections import deque

import numpy as np


BG = 0
DIV = 5
QORDER = ["TL", "TR", "BL", "BR"]
DIR_TO_Q = {(-1, -1): "TL", (-1, 1): "TR", (1, -1): "BL", (1, 1): "BR"}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _axes(g):
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == DIV)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == DIV)]
    assert len(rows) == 1 and len(cols) == 1
    return rows[0], cols[0]


def _quadrants(g):
    ar, ac = _axes(g)
    return {
        "TL": (slice(0, ar), slice(0, ac), 0, 0),
        "TR": (slice(0, ar), slice(ac + 1, g.shape[1]), 0, ac + 1),
        "BL": (slice(ar + 1, g.shape[0]), slice(0, ac), ar + 1, 0),
        "BR": (slice(ar + 1, g.shape[0]), slice(ac + 1, g.shape[1]), ar + 1, ac + 1),
    }


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _sign(x):
    return 1 if x > 0 else -1


def _sprite_info(sub):
    vals = set(map(int, sub.ravel())) - {BG, DIV}
    assert len(vals) == 1
    color = vals.pop()
    comp = [(int(r), int(c)) for r, c in zip(*np.where(sub == color))]
    r0, c0, r1, c1 = _bbox(comp)
    cr = sum(r for r, _ in comp) / len(comp)
    cc = sum(c for _, c in comp) / len(comp)
    br = (r0 + r1) / 2
    bc = (c0 + c1) / 2
    dest = DIR_TO_Q[(_sign(cr - br), _sign(cc - bc))]
    rel = [(r, c) for r, c in comp]
    return {"color": color, "dest": dest, "rel": rel}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each quadrant sprite encodes its destination quadrant by centroid bias. Move
    the shape to that quadrant at the same quadrant-local offset, but recolor it
    with the destination quadrant's original resident color.
    """
    g = np.asarray(grid, dtype=int)
    qs = _quadrants(g)
    infos = {}
    for name, (rs, cs, roff, coff) in qs.items():
        infos[name] = _sprite_info(g[rs, cs])
    dests = [info["dest"] for info in infos.values()]
    assert set(dests) == set(QORDER) and len(dests) == len(set(dests))

    out = np.zeros_like(g)
    ar, ac = _axes(g)
    out[ar, :] = DIV
    out[:, ac] = DIV
    for src_name, info in infos.items():
        dest = info["dest"]
        _, _, droff, dcoff = qs[dest]
        dest_color = infos[dest]["color"]
        for rr, cc in info["rel"]:
            out[droff + rr, dcoff + cc] = dest_color
    return out


SHAPES = {
    (-1, -1): [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    (-1, 1): [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    (1, -1): [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)],
    (1, 1): [(1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
}


def _place(sub, shape, color):
    sh = sw = 4
    for _ in range(100):
        r0 = random.randint(1, sub.shape[0] - sh - 1)
        c0 = random.randint(1, sub.shape[1] - sw - 1)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if all(sub[r, c] == BG for r, c in cells):
            for r, c in cells:
                sub[r, c] = color
            return True
    return False


def _gen_attempt():
    q = random.randint(7, 10)
    n = q * 2 + 1
    grid = np.zeros((n, n), dtype=int)
    grid[q, :] = DIV
    grid[:, q] = DIV
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    dest_perm = random.sample(QORDER, 4)
    qs = _quadrants(grid)
    for src, dest, color in zip(QORDER, dest_perm, colors):
        direction = next(k for k, v in DIR_TO_Q.items() if v == dest)
        rs, cs, _, _ = qs[src]
        if not _place(grid[rs, cs], SHAPES[direction], color):
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
