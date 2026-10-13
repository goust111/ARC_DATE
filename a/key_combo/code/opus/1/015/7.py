import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
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


def _dilate(points, shape):
    h, w = shape
    result = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result.add((nr, nc))
    return result


def _slot_colors(grid):
    grey = np.argwhere(grid == 5)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    line_cols = [c0 + i for i in range(sub.shape[1]) if np.all(sub[:, i] == 5)]
    colors = []
    for a, b in zip(line_cols, line_cols[1:]):
        vals = [int(v) for v in grid[r0 + 1:r1, a + 1:b].ravel() if v not in (0, 5)]
        if vals:
            colors.append(max(set(vals), key=vals.count))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    comps = sorted(_components(grid == 7), key=len, reverse=True)[:2]
    if len(comps) < 2:
        return out
    channel = (_dilate(comps[0], grid.shape) & _dilate(comps[1], grid.shape)) - set(comps[0]) - set(comps[1])
    colors = _slot_colors(grid)
    by_col = {}
    for r, c in channel:
        by_col.setdefault(c, []).append(r)
    for c, rows in by_col.items():
        width = len(rows)
        if 1 <= width <= len(colors):
            for r in rows:
                if out[r, c] == 0:
                    out[r, c] = colors[width - 1]
    return out


def _draw_bottom_grid(grid, n, colors):
    top, left = grid.shape[0] - 3, 1
    grid[top, left:left + 2 * n + 1] = 5
    grid[top + 2, left:left + 2 * n + 1] = 5
    for i in range(n + 1):
        grid[top:top + 3, left + 2 * i] = 5
    for i, color in enumerate(colors):
        grid[top + 1, left + 2 * i + 1] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 18), dtype=int)
    l1, l2 = random.sample([1, 2, 3], 2)
    base = 3
    connector = base + max(l1, l2) + 3
    grid[0, 3:11] = 7
    grid[0:base + l1, 3] = 7
    grid[0:base + l2, 10] = 7
    grid[base:connector + 1, 5] = 7
    grid[base:connector + 1, 12] = 7
    grid[connector, 5:13] = 7
    colors = random.sample([1, 2, 3, 4, 6, 8, 9], 6)
    _draw_bottom_grid(grid, 6, colors)
    return grid
