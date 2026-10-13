import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(cells: np.ndarray):
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    return r0, r1, c0, c1


def _arrow_direction(cells: np.ndarray):
    r0, r1, c0, c1 = _bbox(cells)
    if (c1 - c0) >= (r1 - r0):
        counts = {c: int(np.sum(cells[:, 1] == c)) for c in range(c0, c1 + 1)}
        if counts[c1] == 1:
            return "right"
        return "left"
    counts = {r: int(np.sum(cells[:, 0] == r)) for r in range(r0, r1 + 1)}
    if counts[r1] == 1:
        return "down"
    return "up"


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    arrow_cells = _components(grid == GREY)[0]
    direction = _arrow_direction(arrow_cells)

    blocks = []
    for comp in _components((grid != BLACK) & (grid != GREY)):
        r0, r1, c0, c1 = _bbox(comp)
        color = int(grid[comp[0, 0], comp[0, 1]])
        side = r1 - r0 + 1
        if direction == "right":
            key = c0
        elif direction == "left":
            key = -c1
        elif direction == "down":
            key = r0
        else:
            key = -r1
        blocks.append((key, color, side))
    blocks.sort(key=lambda x: x[0])

    total = sum(side for _, _, side in blocks)
    if direction in ("left", "right"):
        out = np.zeros((1, total), dtype=int)
        pos = 0
        for _, color, side in blocks:
            out[0, pos : pos + side] = color
            pos += side
    else:
        out = np.zeros((total, 1), dtype=int)
        pos = 0
        for _, color, side in blocks:
            out[pos : pos + side, 0] = color
            pos += side
    return out


def _arrow_pixels(direction: str):
    if direction == "right":
        return {(1, 0), (1, 1), (1, 2), (1, 3), (0, 3), (2, 3), (1, 4)}
    if direction == "left":
        return {(1, 4), (1, 3), (1, 2), (1, 1), (0, 1), (2, 1), (1, 0)}
    if direction == "down":
        return {(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (3, 2), (4, 1)}
    return {(4, 1), (3, 1), (2, 1), (1, 1), (1, 0), (1, 2), (0, 1)}


def generate_input() -> np.ndarray:
    colors = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON]
    while True:
        h = random.randint(16, 20)
        w = random.randint(16, 20)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        direction = random.choice(["up", "down", "left", "right"])
        arrow = _arrow_pixels(direction)
        ah = max(r for r, _ in arrow) + 1
        aw = max(c for _, c in arrow) + 1
        ar = random.randint(0, h - ah)
        ac = random.randint(0, w - aw)
        for dr, dc in arrow:
            grid[ar + dr, ac + dc] = GREY
            occupied[ar + dr, ac + dc] = True

        random.shuffle(colors)
        count = random.randint(3, 5)
        ok = True
        for color in colors[:count]:
            side = random.randint(2, 4)
            placed = False
            for _ in range(200):
                r = random.randint(0, h - side)
                c = random.randint(0, w - side)
                area = occupied[r : r + side, c : c + side]
                if np.any(area):
                    continue
                grid[r : r + side, c : c + side] = color
                area[:] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
