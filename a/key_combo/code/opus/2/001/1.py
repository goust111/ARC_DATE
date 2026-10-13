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
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _hole_count(grid, cells):
    r0, c0, r1, c1 = _bbox(cells)
    color = grid[cells[0]]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or sub[r, c] != 0:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches = r in (0, h - 1) or c in (0, w - 1)
            enclosed = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches = True
                    if sub[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif sub[nr, nc] not in (0, color):
                        enclosed = False
            if enclosed and not touches:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    grey = np.argwhere(grid == 5)
    gr0, gc0 = grey.min(axis=0)
    gr1, gc1 = grey.max(axis=0)
    center_r = (gr0 + gr1) // 2
    center_c = (gc0 + gc1) // 2

    sprites = []
    for color, cells in _components(grid, set(range(1, 10)) - {5}):
        r0, c0, r1, c1 = _bbox(cells)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        holes = _hole_count(grid, cells)
        sprites.append(
            {
                "color": color,
                "mask": mask,
                "bbox": (r0, c0, r1, c1),
                "holes": holes,
                "anchor": ((r1 - r0) // 2, (c1 - c0) // 2),
            }
        )

    for sprite in sorted(sprites, key=lambda s: (-s["holes"], s["bbox"][0], s["bbox"][1])):
        top = center_r - sprite["anchor"][0]
        left = center_c - sprite["anchor"][1]
        mask = sprite["mask"]
        for rr in range(mask.shape[0]):
            for cc in range(mask.shape[1]):
                if mask[rr, cc]:
                    out[top + rr, left + cc] = sprite["color"]
    return out


def _shape_for_holes(n):
    if n == 0:
        return np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int)
    if n == 1:
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    if n == 2:
        return np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=int)
    return np.array([[1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=int)


def _can_place(grid, shape, top, left):
    h, w = shape.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    r0, c0 = max(0, top - 1), max(0, left - 1)
    r1, c1 = min(grid.shape[0], top + h + 1), min(grid.shape[1], left + w + 1)
    return not np.any(grid[r0:r1, c0:c1] != 0)


def generate_input() -> np.ndarray:
    h = w = random.choice([17, 18, 19])
    grid = np.zeros((h, w), dtype=int)
    fr0, fc0 = h // 2 - 4, w // 2 - 4
    fr1, fc1 = fr0 + 8, fc0 + 8
    grid[fr0, fc0 : fc1 + 1] = 5
    grid[fr1, fc0 : fc1 + 1] = 5
    grid[fr0 : fr1 + 1, fc0] = 5
    grid[fr0 : fr1 + 1, fc1] = 5

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    holes = random.sample([0, 1, 2, 3], random.choice([3, 4]))
    spots = [(1, 1), (1, w - 9), (h - 5, 1), (h - 5, w - 9), (1, w // 2 - 3), (h - 5, w // 2 - 3)]
    random.shuffle(spots)
    for color, n, (top, left) in zip(colors, holes, spots):
        shape = _shape_for_holes(n)
        if _can_place(grid, shape, top, left):
            grid[top : top + shape.shape[0], left : left + shape.shape[1]][shape == 1] = color
    return grid
