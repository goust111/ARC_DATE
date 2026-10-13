import random
from collections import deque

import numpy as np


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(grid, allowed):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] not in allowed:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] in allowed:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _runs(arr, axis):
    lines = [arr[i, :] if axis == 0 else arr[:, i] for i in range(arr.shape[axis])]
    runs = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i - start, lines[start]))
            start = i
    return runs


def _stretch_axis(arr, target, axis):
    runs = _runs(arr, axis)
    if len(runs) <= 2:
        idx = len(runs) // 2
    else:
        scores = [len(set(int(x) for x in run[2])) for run in runs[1:-1]]
        idx = 1 + max(range(len(scores)), key=lambda i: scores[i])
    fixed = arr.shape[axis] - runs[idx][1]
    new_len = max(1, target - fixed)
    pieces = []
    for i, (_, length, _) in enumerate(runs):
        repeat = new_len if i == idx else length
        sl = arr[runs[i][0] : runs[i][0] + 1, :] if axis == 0 else arr[:, runs[i][0] : runs[i][0] + 1]
        pieces.append(np.repeat(sl, repeat, axis=axis))
    return np.concatenate(pieces, axis=axis)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    markers = np.argwhere(grid == 1)
    rows = sorted(set(int(r) for r, _ in markers))
    cols = sorted(set(int(c) for _, c in markers))
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    out[r0, c0 : c1 + 1] = 1
    out[r1, c0 : c1 + 1] = 1
    out[r0 : r1 + 1, c0] = 1
    out[r0 : r1 + 1, c1] = 1

    comps = _components(grid, set(range(2, 10)))
    exemplar = max(comps, key=lambda cells: (_bbox(cells)[2] - _bbox(cells)[0] + 1) * (_bbox(cells)[3] - _bbox(cells)[1] + 1))
    er0, ec0, er1, ec1 = _bbox(exemplar)
    template = grid[er0 : er1 + 1, ec0 : ec1 + 1]
    stretched = _stretch_axis(template, r1 - r0 - 1, 0)
    stretched = _stretch_axis(stretched, c1 - c0 - 1, 1)
    out[r0 + 1 : r1, c0 + 1 : c1] = stretched[: r1 - r0 - 1, : c1 - c0 - 1]
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([15, 16, 17])
    grid = np.zeros((h, w), dtype=int)
    r0 = random.randint(4, 6)
    c0 = random.randint(5, 7)
    r1 = random.randint(r0 + 5, h - 2)
    c1 = random.randint(c0 + 5, w - 2)
    for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        grid[p] = 1
    patterns = [
        np.array([[2, 2, 2], [2, 3, 2], [2, 2, 2]], dtype=int),
        np.array([[4, 4, 4, 4], [4, 6, 6, 4], [4, 7, 6, 4], [4, 4, 4, 4]], dtype=int),
        np.array([[8, 8, 8], [8, 9, 8], [8, 9, 8], [8, 8, 8]], dtype=int),
    ]
    t = random.choice(patterns)
    grid[0 : t.shape[0], 0 : t.shape[1]] = t
    return grid
