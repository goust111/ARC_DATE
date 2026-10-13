import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _periodic(shape, bbox, mask):
    r0, r1, c0, c1 = bbox
    th, tw = mask.shape
    canvas = np.zeros(shape, dtype=bool)
    for r in range(r0, shape[0], th + 1):
        for c in range(c0, shape[1], tw + 1):
            rh = min(th, shape[0] - r)
            cw = min(tw, shape[1] - c)
            canvas[r : r + rh, c : c + cw] |= mask[:rh, :cw]
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    blue = max(_components(grid == 1), key=len)
    red = max(_components(grid == 2), key=len)
    bb = _bbox(blue)
    rb = _bbox(red)
    bm = grid[bb[0] : bb[1], bb[2] : bb[3]] == 1
    rm = grid[rb[0] : rb[1], rb[2] : rb[3]] == 2
    a = _periodic(grid.shape, bb, bm)
    b = _periodic(grid.shape, rb, rm)
    out = np.zeros_like(grid)
    out[a & ~b] = 1
    out[b & ~a] = 2
    out[a & b] = 3
    return out


def _stamp(mask, color):
    return np.where(mask, color, 0)


def _can_put(grid, arr, r, c):
    h, w = arr.shape
    if r + h > grid.shape[0] or c + w > grid.shape[1]:
        return False
    return not np.any(grid[max(0, r - 1) : min(grid.shape[0], r + h + 1), max(0, c - 1) : min(grid.shape[1], c + w + 1)] != 0)


def generate_input() -> np.ndarray:
    amasks = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    ]
    bmasks = [
        np.array([[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    ]
    for _ in range(1000):
        h = w = random.randint(15, 18)
        grid = np.zeros((h, w), dtype=int)
        am = random.choice(amasks)
        bm = random.choice(bmasks)
        ar, ac = random.randint(0, 4), random.randint(0, 4)
        br, bc = random.randint(0, 6), random.randint(0, 6)
        a_arr = _stamp(am, 1)
        b_arr = _stamp(bm, 2)
        if not _can_put(grid, a_arr, ar, ac):
            continue
        grid[ar : ar + am.shape[0], ac : ac + am.shape[1]] = a_arr
        if not _can_put(grid, b_arr, br, bc):
            continue
        grid[br : br + bm.shape[0], bc : bc + bm.shape[1]] = b_arr
        hint = (h - 1, w - 1)
        if grid[hint] != 0:
            continue
        grid[hint] = 3
        out = main(grid)
        if np.count_nonzero(out == 3) >= 2 and np.count_nonzero(out == 1) > np.count_nonzero(am) and np.count_nonzero(out == 2) > np.count_nonzero(bm):
            return grid
    return grid
