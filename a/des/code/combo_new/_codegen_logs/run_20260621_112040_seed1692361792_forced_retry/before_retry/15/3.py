import random
from collections import deque

import numpy as np


BLACK = 0


def _is_ring(mask: np.ndarray):
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0] = True
    border[-1] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    rings = []
    for color in sorted(set(grid.reshape(-1).tolist()) - {BLACK}):
        cells = np.argwhere(grid == color)
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        border = np.zeros_like(mask, dtype=bool)
        border[0] = True
        border[-1] = True
        border[:, 0] = True
        border[:, -1] = True
        if mask.shape[0] >= 3 and mask.shape[1] >= 3 and np.all(mask[border]):
            rings.append(((r0, r1, c0, c1), color))

    rings.sort(key=lambda x: (x[0][1] - x[0][0] + 1) * (x[0][3] - x[0][2] + 1), reverse=True)
    inner_bbox = rings[-1][0]
    ir0, ir1, ic0, ic1 = inner_bbox
    inner_pixels = []
    for r in range(ir0 + 1, ir1):
        for c in range(ic0 + 1, ic1):
            if grid[r, c] != BLACK:
                inner_pixels.append((c, int(grid[r, c])))
    marker_colors = [color for _, color in sorted(inner_pixels)]
    heights = [r1 - r0 + 1 for (r0, r1, c0, c1), _ in rings]
    widths = [c1 - c0 + 1 for (r0, r1, c0, c1), _ in rings]
    out_h = heights[0]
    out_w = sum(widths) + len(widths) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    offset = 0
    for i, ((r0, r1, c0, c1), _) in enumerate(rings):
        rh = r1 - r0 + 1
        rw = c1 - c0 + 1
        top = (out_h - rh) // 2
        color = marker_colors[i]
        out[top, offset : offset + rw] = color
        out[top + rh - 1, offset : offset + rw] = color
        out[top : top + rh, offset] = color
        out[top : top + rh, offset + rw - 1] = color
        offset += rw + 1
    return out


def generate_input() -> np.ndarray:
    colors_all = [1, 2, 3, 4, 6, 7, 8, 9]
    h = 13
    w = 13
    grid = np.zeros((h, w), dtype=int)
    random.shuffle(colors_all)
    ring_colors = colors_all[:3]

    sizes = [(h, w), (h - 4, w - 4), (h - 8, w - 8)]
    for color, (rh, rw) in zip(ring_colors, sizes):
        top = (h - rh) // 2
        left = (w - rw) // 2
        grid[top, left : left + rw] = color
        grid[top + rh - 1, left : left + rw] = color
        grid[top : top + rh, left] = color
        grid[top : top + rh, left + rw - 1] = color

    inner_h, inner_w = sizes[-1][0] - 2, sizes[-1][1] - 2
    inner_top = (h - sizes[-1][0]) // 2 + 1
    inner_left = (w - sizes[-1][1]) // 2 + 1
    cols = [inner_left + i for i in range(3)]
    rows = random.sample(range(inner_top, inner_top + inner_h), 3)
    for c, r, color in zip(cols, rows, random.sample(ring_colors, 3)):
        grid[r, c] = color
    return grid
