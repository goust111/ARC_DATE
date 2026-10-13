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
    instr = grid[:6, :6]
    mask = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            block = instr[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2]
            mask[r, c] = int(np.any(block == 2))

    bar_rows = np.where(np.sum(grid == 5, axis=1) >= 5)[0]
    bar_top, bar_bottom = int(bar_rows.min()), int(bar_rows.max())
    grey_cols = np.where(np.any(grid[bar_top : bar_bottom + 1] == 5, axis=0))[0]
    notch_cols = []
    c = int(grey_cols.min())
    while c <= int(grey_cols.max()):
        if grid[bar_top, c] == 0 and grid[bar_bottom, c] == 5:
            start = c
            while c <= int(grey_cols.max()) and grid[bar_top, c] == 0 and grid[bar_bottom, c] == 5:
                c += 1
            notch_cols.append((start, c - 1))
        c += 1

    sprites = []
    for color, cells in _components(grid, set(range(1, 10)) - {2, 4, 5}):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 < bar_top:
            sprites.append((color, grid[r0 : r1 + 1, c0 : c1 + 1] == color, cells))
            for r, c in cells:
                out[r, c] = 0
    out[:6, :6] = 0

    sprites.sort(key=lambda item: item[1].shape[1])
    notch_cols.sort(key=lambda p: p[0])
    for col, (left, right) in enumerate(notch_cols[:3]):
        width = right - left + 1
        candidates = [s for s in sprites if s[1].shape[1] == width]
        color, shape, _ = candidates[0] if candidates else sprites[col]
        top = bar_top - shape.shape[0]
        base_left = left
        step = 0
        while True:
            if step > 0 and (step > 2 or mask[step, col] == 0):
                break
            t = top - step * shape.shape[0]
            if t < 0:
                break
            for r in range(shape.shape[0]):
                for c in range(shape.shape[1]):
                    if shape[r, c]:
                        out[t + r, base_left + c] = color
            step += 1
    return out


def _shape(width):
    if width == 2:
        return np.array([[1, 1], [1, 0]], dtype=int)
    if width == 3:
        return np.array([[1, 1, 1], [0, 1, 0]], dtype=int)
    return np.array([[1, 1, 1, 1], [1, 0, 0, 1]], dtype=int)


def generate_input() -> np.ndarray:
    h, w = 20, 20
    grid = np.zeros((h, w), dtype=int)
    mask = np.zeros((3, 3), dtype=int)
    while mask[0].sum() == 0 or len({tuple(mask[:, c]) for c in range(3)}) < 2:
        mask = np.random.randint(0, 2, (3, 3))
    for r in range(3):
        for c in range(3):
            if mask[r, c]:
                grid[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2] = 2
    for i in range(6):
        if grid[i, :6].sum() == 0:
            grid[i, i % 6] = 4

    bar_top, bar_bottom = 12, 14
    grid[bar_top : bar_bottom + 1, 2:18] = 5
    widths = random.sample([2, 3, 4], 3)
    starts = [4, 9, 14]
    for s, width in zip(starts, widths):
        grid[bar_top, s : s + width] = 0
    colors = random.sample([1, 3, 6, 7, 8, 9], 3)
    for idx, (s, width, color) in enumerate(zip(starts, widths, colors)):
        shape = _shape(width)
        top = 7 + (idx % 2)
        left = s
        grid[top : top + shape.shape[0], left : left + shape.shape[1]][shape == 1] = color
    return grid
