import random
from collections import deque

import numpy as np


BG = 0


def _components(grid, eight=False):
    mask = grid != BG
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if eight:
        neigh = neigh + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            mono = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        if int(grid[nr, nc]) != color:
                            mono = False
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells, mono))
    return comps


def _arrow_info(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rel = {(r - r0, c - c0) for r, c in cells}
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    if len({c for _, c in rel}) == 3:
        if {(0, 2), (2, 2)}.issubset(rel):
            return (0, 1), center
        return (0, -1), center
    if {(2, 0), (2, 2)}.issubset(rel):
        return (1, 0), center
    return (-1, 0), center


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    arrows = []
    objects = []
    for color, cells, mono in _components(grid, eight=True):
        if mono and len(cells) == 5:
            arrows.append((color, cells))
        else:
            objects.append((color, cells))
    object_map = np.zeros_like(grid, dtype=int)
    for idx, (_, cells) in enumerate(objects, start=1):
        for r, c in cells:
            object_map[r, c] = idx
    for color, cells in arrows:
        direction, center = _arrow_info(cells)
        for r, c in cells:
            out[r, c] = BG
        cr, cc = center
        target = None
        rr, cc2 = cr + direction[0], cc + direction[1]
        while 0 <= rr < grid.shape[0] and 0 <= cc2 < grid.shape[1]:
            idx = object_map[rr, cc2]
            if idx:
                target = objects[idx - 1][1]
                break
            rr += direction[0]
            cc2 += direction[1]
        if target is None:
            continue
        for r, c in target:
            nr = 2 * cr - r
            nc = 2 * center[1] - c
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                out[nr, nc] = color
    return out


def _random_object(color):
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    arr = np.zeros((h, w), dtype=int)
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < random.randint(2, min(5, h * w)):
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    for r, c in cells:
        arr[r, c] = color
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    return arr[rows][:, cols]


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    obj_colors = random.sample([1, 2, 3, 4, 5, 6], random.randint(2, 3))
    objects = []
    for color in obj_colors:
        obj = _random_object(color)
        oh, ow = obj.shape
        for _ in range(200):
            top = random.randint(3, h - oh - 4)
            left = random.randint(3, w - ow - 4)
            if occupied[top - 1 : top + oh + 1, left - 1 : left + ow + 1].any():
                continue
            grid[top : top + oh, left : left + ow][obj != 0] = color
            occupied[top : top + oh, left : left + ow] |= obj != 0
            objects.append(((top, left), obj))
            break

    arrow_colors = random.sample([7, 8, 9], len(objects))
    for arrow_color, ((top, left), obj) in zip(arrow_colors, objects):
        oh, ow = obj.shape
        if left > w - (left + ow):
            continue
        center_r = top + oh // 2
        center_c = max(3, left - 4)
        coords = [(center_r, center_c - 1), (center_r, center_c), (center_r, center_c + 1), (center_r - 1, center_c + 1), (center_r + 1, center_c + 1)]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
            continue
        if any(occupied[r, c] for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = arrow_color
            occupied[r, c] = True
    return grid
