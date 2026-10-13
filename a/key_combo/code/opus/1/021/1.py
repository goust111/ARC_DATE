import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    out = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            x, y = q.popleft()
            cells.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        out.append(cells)
    return out


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _hole_count(crop):
    fg = crop != 0
    bg = ~fg
    seen = np.zeros(bg.shape, bool)
    holes = 0
    for r, c in zip(*np.where(bg)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        touches_edge = False
        while q:
            x, y = q.popleft()
            if x in (0, bg.shape[0] - 1) or y in (0, bg.shape[1] - 1):
                touches_edge = True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < bg.shape[0] and 0 <= ny < bg.shape[1] and bg[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        if not touches_edge:
            holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    grey = np.argwhere(grid == 5)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    inside = (r0 + 1, r1 - 1, c0 + 1, c1 - 1)
    anchors = {}
    for color in (1, 2, 3):
        pts = np.argwhere(grid[inside[0] : inside[1] + 1, inside[2] : inside[3] + 1] == color)
        if len(pts):
            rr, cc = pts[0]
            anchors[color] = (inside[0] + rr, inside[2] + cc)

    sprites = []
    for color in (1, 2, 3):
        mask = (grid == color)
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        for comp in _components(mask):
            br0, br1, bc0, bc1 = _bbox(comp)
            crop = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=grid.dtype)
            for rr, cc in comp:
                crop[rr - br0, cc - bc0] = color
            sprites.append((br0, color, crop))

    for _, color, crop in sorted(sprites):
        h = _hole_count(crop)
        rotated = np.rot90(crop, -h)
        ar, ac = anchors[color]
        top = ar - rotated.shape[0] // 2
        left = ac - rotated.shape[1] // 2
        for rr in range(rotated.shape[0]):
            for cc in range(rotated.shape[1]):
                if rotated[rr, cc] != 0:
                    out[top + rr, left + cc] = rotated[rr, cc]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    r0, r1, c0, c1 = 3, 18, 9, 20
    grid[r0 : r1 + 1, c0] = 5
    grid[r0 : r1 + 1, c1] = 5
    grid[r0, c0 : c1 + 1] = 5
    grid[r1, c0 : c1 + 1] = 5

    shapes = {
        0: np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),
        1: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        2: np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]]),
    }
    colors = [1, 2, 3]
    random.shuffle(colors)
    rows = [1, 8, 15]
    random.shuffle(rows)
    for holes, color, row in zip((0, 1, 2), colors, rows):
        shape = shapes[holes] * color
        grid[row : row + shape.shape[0], 1 : 1 + shape.shape[1]] = shape

    anchors = [(7, 13), (7, 17), (14, 15)]
    random.shuffle(anchors)
    for color, pos in zip(colors, anchors):
        grid[pos] = color
    return grid
