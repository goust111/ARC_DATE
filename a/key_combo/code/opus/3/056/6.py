import random
import numpy as np


def _components(mask, eight=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if eight:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = [(r, c)]
        seen[r, c] = True
        comp = []
        while q:
            cr, cc = q.pop()
            comp.append((cr, cc))
            for dr, dc in dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _corner_of(comp, shape):
    h, w = shape
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    top, bottom = min(rs) == 0, max(rs) == h - 1
    left, right = min(cs) == 0, max(cs) == w - 1
    if top and left:
        return "tl"
    if top and right:
        return "tr"
    if bottom and right:
        return "br"
    if bottom and left:
        return "bl"
    return None


def _target_origin(corner, bbox, shape):
    h, w = shape
    r0, c0, r1, c1 = bbox
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    return {
        "tl": (0, 0),
        "tr": (0, w - bw),
        "br": (h - bh, w - bw),
        "bl": (h - bh, 0),
    }[corner]


def main(grid: np.ndarray) -> np.ndarray:
    """Copy green corner objects clockwise, then cast red diagonal rays outward until any obstacle."""
    out = grid.copy()
    h, w = grid.shape
    nxt = {"tl": "tr", "tr": "br", "br": "bl", "bl": "tl"}
    for comp in _components(grid == 3):
        corner = _corner_of(comp, grid.shape)
        if corner is None:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        tr0, tc0 = _target_origin(nxt[corner], bbox, grid.shape)
        for r, c in comp:
            out[tr0 + r - bbox[0], tc0 + c - bbox[1]] = 3

    blue = np.argwhere(out == 1)
    if blue.size == 0:
        return out
    center = blue.mean(axis=0)
    for comp in _components(out == 2, eight=True):
        cr = sum(r for r, _ in comp) / len(comp)
        cc = sum(c for _, c in comp) / len(comp)
        dr = 1 if cr > center[0] else -1
        dc = 1 if cc > center[1] else -1
        endpoints = []
        s = set(comp)
        for r, c in comp:
            deg = sum((r + ar, c + ac) in s for ar, ac in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            if deg <= 1:
                endpoints.append((r, c))
        for r, c in endpoints:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                out[nr, nc] = 2
                nr += dr
                nc += dc
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    for r, c in [(5, 7), (6, 6), (6, 8), (7, 5), (7, 9), (8, 6), (8, 8), (9, 7)]:
        g[r, c] = 1
    for cells in [[(4, 5), (3, 4)], [(4, 9), (3, 10)], [(10, 5), (11, 4)], [(10, 9), (11, 10)]]:
        for r, c in cells:
            g[r, c] = 2
    shape = random.choice([
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    ])
    corners = [("tl", (0, 0)), ("br", (12, 13))] if random.random() < 0.5 else [("tr", (0, 13)), ("bl", (12, 0))]
    for _, (r0, c0) in corners:
        for dr, dc in shape:
            cc = c0 + (dc if c0 == 0 else -dc)
            g[r0 + dr, cc] = 3
    return g
