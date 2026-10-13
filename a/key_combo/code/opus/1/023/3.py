import random
from collections import deque

import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r, c in zip(*np.where(grid == color)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def _frame_info(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), max(rs), min(cs), max(cs)


def _contains(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return ar0 < br0 and br1 < ar1 and ac0 < bc0 and bc1 < ac1


def _complete_border(cells, box):
    r0, r1, c0, c1 = box
    s = set(cells)
    for c in range(c0, c1 + 1):
        if (r0, c) not in s or (r1, c) not in s:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in s or (r, c1) not in s:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    frames = _components(grid, 5)
    boxes = [_frame_info(comp) for comp in frames]
    colors = {0: 1, 1: 2, 2: 3}
    frame_colors = []
    for comp, box in zip(frames, boxes):
        depth = sum(1 for other in boxes if _contains(other, box))
        color = colors.get(depth, 3)
        frame_colors.append(color)
        for r, c in comp:
            out[r, c] = color

    h, w = grid.shape
    for comp, box, color in zip(frames, boxes, frame_colors):
        r0, r1, c0, c1 = box
        if r1 - r0 != c1 - c0 or not _complete_border(comp, box):
            continue
        marks = [
            (r0 - 1, c0), (r0, c0 - 1),
            (r0 - 1, c1), (r0, c1 + 1),
            (r1 + 1, c0), (r1, c0 - 1),
            (r1 + 1, c1), (r1, c1 + 1),
        ]
        for r, c in marks:
            if 0 <= r < h and 0 <= c < w and out[r, c] == 0:
                out[r, c] = color
    return out


def _draw_frame(grid, r0, c0, height, width):
    grid[r0, c0:c0 + width] = 5
    grid[r0 + height - 1, c0:c0 + width] = 5
    grid[r0:r0 + height, c0] = 5
    grid[r0:r0 + height, c0 + width - 1] = 5


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    outer_size = random.choice([15, 16, 17])
    _draw_frame(grid, 1, 1, outer_size, outer_size)
    _draw_frame(grid, 3, 3, random.choice([5, 6]), random.choice([7, 8]))
    sq_size = random.choice([7, 8])
    sq_r = outer_size - sq_size
    sq_c = outer_size - sq_size
    _draw_frame(grid, sq_r, sq_c, sq_size, sq_size)
    if sq_size >= 7:
        _draw_frame(grid, sq_r + 2, sq_c + 2, 3, 3)
    if outer_size <= 15:
        if random.random() < 0.5:
            _draw_frame(grid, 2, outer_size + 2, 5, 3)
        else:
            _draw_frame(grid, outer_size + 2, 2, 3, 5)
    return grid
