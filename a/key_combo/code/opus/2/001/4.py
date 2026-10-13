import random
from collections import deque

import numpy as np


def _components(grid, colors):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] not in colors:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    brackets = []
    for _, cells in _components(grid, {2}):
        r0, c0, r1, c1 = _bbox(cells)
        if r0 <= 1 and c1 - c0 >= 3:
            brackets.append((c0 + 1, c1 - 1))
    brackets.sort()

    sprites = []
    for color, cells in _components(grid, set(range(1, 10)) - {2}):
        r0, c0, r1, c1 = _bbox(cells)
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        center = (c0 + c1) / 2
        channel = next(i for i, (a, b) in enumerate(brackets) if a <= center <= b)
        sprites.append((channel, len(cells), r0, color, mask, (r0, c0, r1, c1)))
        for r, c in cells:
            out[r, c] = 0

    for idx, (left, right) in enumerate(brackets):
        stack = sorted([s for s in sprites if s[0] == idx], key=lambda s: (s[1], s[2]))
        bottom = grid.shape[0] - 1
        mid = (left + right) // 2
        for _, _, _, color, mask, _ in stack:
            mh, mw = mask.shape
            top = bottom - mh + 1
            start = mid - mw // 2
            for r in range(mh):
                for c in range(mw):
                    if mask[r, c]:
                        out[top + r, start + c] = color
            bottom = top - 1
    return out


def _shape(n):
    shapes = [
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    ]
    return shapes[n % len(shapes)]


def generate_input() -> np.ndarray:
    n_ch = random.choice([2, 3])
    h, w = (16, random.choice([23, 24, 25])) if n_ch == 3 else (16, random.choice([19, 20, 21]))
    grid = np.zeros((h, w), dtype=int)
    starts = [1, 9, 17] if n_ch == 3 else [3, 13]
    widths = [random.choice([3, 4, 5]) for _ in range(n_ch)]
    channels = []
    for s, iw in zip(starts, widths):
        l, r = s, min(w - 2, s + iw + 1)
        grid[0, l:r + 1] = 2
        grid[0:3, l] = 2
        grid[0:3, r] = 2
        channels.append((l + 1, r - 1))

    counts = [2 for _ in range(n_ch)] if n_ch == 3 else [random.choice([2, 3]) for _ in range(n_ch)]
    total = sum(counts)
    colors = random.sample([1, 3, 4, 6, 7, 8, 9], total)
    ci = 0
    occupied = np.zeros_like(grid, dtype=bool)
    for ch, (left, right) in enumerate(channels):
        count = counts[ch]
        rows = [4, 8, 12][:count]
        for j, row in enumerate(rows):
            sh = _shape(ci + j)
            col = random.randint(left, max(left, right - sh.shape[1] + 1))
            grid[row : row + sh.shape[0], col : col + sh.shape[1]][sh == 1] = colors[ci]
            occupied[row : row + sh.shape[0], col : col + sh.shape[1]] |= sh.astype(bool)
            ci += 1
    return grid
