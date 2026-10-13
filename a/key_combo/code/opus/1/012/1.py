import random
from collections import Counter, deque

import numpy as np


def _component(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return []
    start = tuple(pts[0])
    q = deque([start])
    seen = {start}
    comp = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and mask[p] and p not in seen:
                seen.add(p)
                q.append(p)
    return comp


def _rot180_symmetric(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    pts = {(r - r0, c - c0) for r, c in comp}
    h, w = r1 - r0 + 1, c1 - c0 + 1
    return {(h - 1 - r, w - 1 - c) for r, c in pts} == pts


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    gr = next(r for r in range(grid.shape[0]) if np.all(grid[r] == 5))
    gc = next(c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5))
    slices = [
        (slice(0, gr), slice(0, gc)),
        (slice(0, gr), slice(gc + 1, grid.shape[1])),
        (slice(gr + 1, grid.shape[0]), slice(0, gc)),
        (slice(gr + 1, grid.shape[0]), slice(gc + 1, grid.shape[1])),
    ]
    colors = []
    for s in slices:
        q = grid[s]
        comp = _component((q != 0) & (q != 5))
        vals = [int(q[p]) for p in comp]
        counts = Counter(vals)
        major, minor = counts.most_common()[0][0], counts.most_common()[-1][0]
        colors.append(minor if _rot180_symmetric(comp) else major)
    out = np.zeros((3, 3), dtype=int)
    out[1, :] = 5
    out[:, 1] = 5
    out[0, 0], out[0, 2], out[2, 0], out[2, 2] = colors
    return out


def _put_shape(q, top, left, cells, a, b):
    majority = random.choice([a, b])
    minority = b if majority == a else a
    split = len(cells) - max(1, len(cells) // 3)
    for i, (r, c) in enumerate(cells):
        q[top + r, left + c] = majority if i < split else minority


def generate_input() -> np.ndarray:
    grid = np.zeros((21, 21), dtype=int)
    grid[10, :] = 5
    grid[:, 10] = 5
    sym_shape = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    asym_shape = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3)]
    pairs = [(1, 2), (3, 4), (6, 7), (8, 9)]
    flags = [True, False] + [random.choice([True, False]) for _ in range(2)]
    random.shuffle(flags)
    origins = [(1, 1), (1, 12), (12, 1), (12, 12)]
    for (r0, c0), pair, sym in zip(origins, pairs, flags):
        q = grid[r0:r0 + 8, c0:c0 + 8]
        _put_shape(q, random.randint(1, 3), random.randint(1, 3), sym_shape if sym else asym_shape, *pair)
    return grid
