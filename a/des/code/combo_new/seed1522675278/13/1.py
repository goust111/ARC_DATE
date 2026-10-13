import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _count_holes(shape):
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    box = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in shape:
        box[r - r0, c - c0] = True
    h, w = box.shape
    outside = np.zeros_like(box, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not box[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not box[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not box[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    holes = (~box) & (~outside)
    return len(_components(holes))


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    legend_rows = []
    legend_width = 0
    for r in range(h):
        left = grid[r, :6]
        if np.any(left != BG):
            colors = [v for v in left if v not in (BG, GREY)]
            if len(colors) == 1:
                legend_rows.append((r, colors[0], int(np.count_nonzero(left == GREY))))
                legend_width = max(legend_width, int(np.max(np.where(left != BG)[0])) + 1)
    legend = {count: (row, color) for row, color, count in legend_rows}
    mask = np.zeros_like(grid, dtype=bool)
    mask[:, legend_width + 1 :] = grid[:, legend_width + 1 :] != BG
    comps = _components(mask)
    items = []
    for comp in comps:
        hole_count = _count_holes(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        obj = np.where(patch != BG, legend[hole_count][1], BG)
        items.append((hole_count, c0, obj))
    out = np.zeros_like(grid)
    out[:, :legend_width] = grid[:, :legend_width]
    next_col = {k: legend_width + 1 for k in legend}
    for hole_count, _, obj in sorted(items, key=lambda x: x[1]):
        row = legend[hole_count][0]
        oh, ow = obj.shape
        col = next_col[hole_count]
        oh = min(oh, h - row)
        ow = min(ow, w - col)
        if oh > 0 and ow > 0:
            view = out[row : row + oh, col : col + ow]
            out[row : row + oh, col : col + ow] = np.where(obj[:oh, :ow] != BG, obj[:oh, :ow], view)
        next_col[hole_count] += obj.shape[1] + 1
    return out


def _shape_with_holes(hole_count, color):
    if hole_count == 0:
        hh = random.randint(2, 4)
        ww = random.randint(2, 5)
        return np.full((hh, ww), color, dtype=int)
    hh = random.randint(5, 7)
    ww = random.randint(2 * hole_count + 3, 2 * hole_count + 5)
    arr = np.zeros((hh, ww), dtype=int)
    arr[[0, -1], :] = color
    arr[:, [0, -1]] = color
    for i in range(1, hole_count):
        cc = 1 + 2 * i
        arr[:, cc] = color
    return arr


def generate_input() -> np.ndarray:
    legend_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    object_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(5, 8))
    h, w = 24, 28
    grid = np.zeros((h, w), dtype=int)
    rows = [1, 7, 13, 19]
    for hole_count, row in enumerate(rows):
        grid[row, 0] = legend_colors[hole_count]
        for k in range(hole_count):
            grid[row, 2 + k] = GREY
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:, :6] = True
    color_idx = 0
    for hole_count in range(4):
        for _ in range(random.randint(1, 2)):
            shape = _shape_with_holes(hole_count, object_colors[color_idx % len(object_colors)])
            color_idx += 1
            sh, sw = shape.shape
            for _ in range(200):
                r = random.randint(0, h - sh)
                c = random.randint(8, w - sw)
                rr0, rr1 = max(0, r - 1), min(h, r + sh + 1)
                cc0, cc1 = max(0, c - 1), min(w, c + sw + 1)
                if np.any(occupied[rr0:rr1, cc0:cc1]):
                    continue
                occupied[rr0:rr1, cc0:cc1] = True
                grid[r : r + sh, c : c + sw] = np.where(shape != BG, shape, grid[r : r + sh, c : c + sw])
                break
    return grid
