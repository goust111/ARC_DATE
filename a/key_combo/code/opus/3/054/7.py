import random

import numpy as np


def _groups(grid):
    cols = np.where(np.any(grid != 0, axis=0))[0]
    runs = []
    if cols.size == 0:
        return runs
    start = prev = int(cols[0])
    for c in map(int, cols[1:]):
        if c == prev + 1:
            prev = c
        else:
            runs.append((start, prev))
            start = prev = c
    runs.append((start, prev))
    return [grid[:, a : b + 1].copy() for a, b in runs]


def _edge_gray(crop, left=True):
    col = 0 if left else crop.shape[1] - 1
    pts = np.argwhere(crop[:, col] == 5)
    if pts.size:
        return int(pts[0, 0])
    pts = np.argwhere(crop[:, col] == 3)
    if pts.size:
        return int(pts[0, 0])
    pts = np.argwhere(crop[:, col] != 0)
    return int(pts[0, 0])


def main(grid: np.ndarray) -> np.ndarray:
    """Concatenate endpoint-aligned path segments, remember gray seam points, and color the path alternating at those seams."""
    g = np.array(grid, dtype=int)
    groups = _groups(g)
    if not groups:
        return g.copy()
    width = sum(x.shape[1] for x in groups)
    out = np.zeros((g.shape[0], width), dtype=int)
    seams = set()
    x = 0
    prev_right = None
    seed = None
    for i, crop in enumerate(groups):
        shift = 0 if i == 0 else prev_right - _edge_gray(crop, left=True)
        for r in range(crop.shape[0]):
            for c in range(crop.shape[1]):
                v = int(crop[r, c])
                if v == 0:
                    continue
                rr, cc = r + shift, x + c
                if 0 <= rr < out.shape[0]:
                    out[rr, cc] = v
                    if v == 5:
                        seams.add((rr, cc))
                    elif v == 3:
                        seed = (rr, cc)
        prev_right = _edge_gray(crop, left=False) + shift
        x += crop.shape[1]

    path = out != 0
    out[out == 5] = 8
    if seed is None:
        pts = np.argwhere(out == 3)
        if pts.size == 0:
            return out
        seed = tuple(map(int, pts[0]))

    color = 2
    prev = None
    cur = seed
    visited = {seed}
    while True:
        nxts = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if (
                0 <= nr < out.shape[0]
                and 0 <= nc < out.shape[1]
                and path[nr, nc]
                and (nr, nc) != prev
                and (nr, nc) not in visited
            ):
                nxts.append((nr, nc))
        if not nxts:
            break
        nxt = nxts[0]
        visited.add(nxt)
        if nxt in seams:
            color = 1 if color == 2 else 2
        if out[nxt] != 3:
            out[nxt] = color
        prev, cur = cur, nxt
    out[seed] = 3
    return out


def _segment(width, left_row, right_row):
    pts = [(left_row, 0)]
    row = left_row
    for c in range(1, width):
        if c == width - 1:
            while row != right_row:
                row += 1 if right_row > row else -1
                pts.append((row, c - 1))
        elif random.random() < 0.4:
            step = random.choice([-1, 1])
            if 0 <= row + step < 4:
                row += step
                pts.append((row, c - 1))
        pts.append((row, c))
    crop = np.zeros((4, width), dtype=int)
    for r, c in pts:
        crop[r, c] = 8
    crop[left_row, 0] = 5
    crop[right_row, width - 1] = 5
    return crop


def generate_input() -> np.ndarray:
    n = random.choice([4, 5])
    widths = [random.choice([3, 4, 5]) for _ in range(n)]
    left_rows = [random.choice([1, 2]) for _ in range(n)]
    right_rows = [random.choice([1, 2]) for _ in range(n)]
    groups = [_segment(widths[i], left_rows[i], right_rows[i]) for i in range(n)]
    groups[0][left_rows[0], 0] = 3
    total_w = sum(widths) + n - 1
    grid = np.zeros((4, total_w), dtype=int)
    x = 0
    for i, crop in enumerate(groups):
        grid[:, x : x + crop.shape[1]] = crop
        x += crop.shape[1]
        if i != n - 1:
            x += 1
    return grid
