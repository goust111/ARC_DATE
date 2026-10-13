import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mask_and_bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r1, c1, r2, c2 = min(rs), min(cs), max(rs), max(cs)
    m = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
    for r, c in cells:
        m[r - r1, c - c1] = True
    return m, (r1, c1, r2, c2)


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    red = _components(grid == 2)
    blue = _components(grid == 1)
    if not red or not blue:
        return out
    marker, mb = _mask_and_bbox(red[0])
    missing = None
    for r in range(2):
        for c in range(2):
            if r >= marker.shape[0] or c >= marker.shape[1] or not marker[r, c]:
                missing = (r, c)
    corner = {(1, 1): "tl", (1, 0): "tr", (0, 1): "bl", (0, 0): "br"}[missing]
    sprite, _ = _mask_and_bbox(max(blue, key=len))
    sh, sw = sprite.shape
    out[out == 1] = 0
    out[out == 2] = 0
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - sw
    elif corner == "bl":
        r0, c0 = h - sh, 0
    else:
        r0, c0 = h - sh, w - sw
    placed = []
    for rr in range(sh):
        for cc in range(sw):
            if sprite[rr, cc]:
                out[r0 + rr, c0 + cc] = 1
                placed.append((r0 + rr, c0 + cc))
    for r, c in sorted(placed):
        offsets = []
        if r == 0:
            offsets.append((sh, 0))
        if r == h - 1:
            offsets.append((-sh, 0))
        if c == 0:
            offsets.append((0, sw))
        if c == w - 1:
            offsets.append((0, -sw))
        for dr, dc in offsets:
            sr, sc = r0 + dr, c0 + dc
            for rr in range(sh):
                for cc in range(sw):
                    nr, nc = sr + rr, sc + cc
                    if sprite[rr, cc] and 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = 3
    return out


def generate_input() -> np.ndarray:
    n = random.choice([15, 17, 19, 21])
    grid = np.zeros((n, n), dtype=int)
    sprites = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    ]
    sprite = random.choice(sprites)
    sh, sw = sprite.shape
    r0 = random.randint(5, n - sh - 5)
    c0 = random.randint(5, n - sw - 5)
    grid[r0:r0 + sh, c0:c0 + sw][sprite] = 1
    missing_by_corner = {"tl": (1, 1), "tr": (1, 0), "bl": (0, 1), "br": (0, 0)}
    corner = random.choice(list(missing_by_corner))
    mr, mc = random.choice([(1, 1), (1, n - 3), (n - 3, 1), (n - 3, n - 3)])
    while abs(mr - r0) <= sh + 1 and abs(mc - c0) <= sw + 1:
        mr, mc = random.choice([(1, 1), (1, n - 3), (n - 3, 1), (n - 3, n - 3)])
    miss = missing_by_corner[corner]
    for rr in range(2):
        for cc in range(2):
            if (rr, cc) != miss:
                grid[mr + rr, mc + cc] = 2
    return grid
