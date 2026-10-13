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


def _mask(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r1, c1 = min(rs), min(cs)
    arr = np.zeros((max(rs) - r1 + 1, max(cs) - c1 + 1), dtype=bool)
    for r, c in cells:
        arr[r - r1, c - c1] = True
    return arr


def _key(arr):
    return (arr.shape, tuple(arr.astype(int).ravel()))


def _cw(arr, k):
    return np.rot90(arr, -k)


def _angle(mask, rots):
    for k, r in enumerate(rots):
        if mask.shape == r.shape and np.array_equal(mask, r):
            return k
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    greens = np.argwhere(out == 3)
    if len(greens) == 0:
        return out
    s = int(greens[0, 0] + greens[0, 1])
    c0, c1 = int(greens[:, 1].min()), int(greens[:, 1].max())
    L = c1 - c0 + 1
    out[out == 3] = 0
    new_cells = []
    for c in range(c1 + 1, c1 + L + 2):
        r = s - c
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = 3
            new_cells.append((r, c))
    if not new_cells:
        return out
    p_top = min(new_cells)
    p_bot = max(new_cells)

    obstacle_comps = _components(out == 5)
    masks = [_mask(comp) for comp in obstacle_comps]
    t0 = min(masks, key=_key) if masks else np.zeros((1, 1), dtype=bool)
    rots = [_cw(t0, k) for k in range(4)]
    cell_to_comp = {}
    for i, comp in enumerate(obstacle_comps):
        for cell in comp:
            cell_to_comp[cell] = i

    for start, direction in ((p_top, (-1, 1)), (p_bot, (1, -1))):
        r, c = start
        last = start
        hit = None
        while True:
            nr, nc = r + direction[0], c + direction[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if out[nr, nc] == 5:
                hit = (nr, nc)
                break
            r, c = nr, nc
            out[r, c] = 3
            last = (r, c)
        if hit is not None:
            comp_i = cell_to_comp[hit]
            ang = _angle(masks[comp_i], rots)
            out[last] = [2, 1, 4, 7][ang]
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(18, 22)
    grid = np.zeros((h, w), dtype=int)
    L = random.randint(3, 5)
    c0 = random.randint(3, 6)
    s = random.randint(10, h - 6)
    c1 = c0 + L - 1
    for c in range(c0, c1 + 1):
        grid[s - c, c] = 3
    new_top = (s - (c1 + L + 1), c1 + L + 1)
    new_bot = (s - (c1 + 1), c1 + 1)
    base = np.array([[1, 1], [0, 1]], dtype=bool)
    angles = random.sample(range(4), 2)
    for (start, dr, dc), ang in zip(((new_top, -1, 1), (new_bot, 1, -1)), angles):
        dist = random.randint(2, 4)
        hit = (start[0] + dr * dist, start[1] + dc * dist)
        m = _cw(base, ang)
        mh, mw = m.shape
        offsets = [(rr, cc) for rr in range(mh) for cc in range(mw) if m[rr, cc]]
        rr, cc = random.choice(offsets)
        r0, c0p = hit[0] - rr, hit[1] - cc
        if 0 <= r0 < h - mh and 0 <= c0p < w - mw:
            grid[r0:r0 + mh, c0p:c0p + mw][m] = 5
    return grid
