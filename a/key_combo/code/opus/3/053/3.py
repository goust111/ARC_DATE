import random

import numpy as np


BG = 7
WALL = 6
BLOCK_COLORS = [1, 3, 8, 9]


def _find_blocks(grid):
    h, w = grid.shape
    blocks = []
    occupied = set()
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            color = int(patch[1, 1])
            if color in BLOCK_COLORS and np.all(patch == color):
                cells = {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)}
                if not cells & occupied:
                    blocks.append({"color": color, "center": (r + 1, c + 1), "cells": cells})
                    occupied |= cells
    return blocks


def _trace(grid, block, body_cells):
    h, w = grid.shape
    color = block["color"]
    cr, cc = block["center"]
    reached = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = cr + dr, cc + dc
        while 0 <= r < h and 0 <= c < w and grid[r, c] != WALL:
            reached.add((r, c))
            if (r, c) not in body_cells:
                yield r, c, color
            r += dr
            c += dc
    return reached


def _reachable_from(grid, center, point):
    if center[0] != point[0] and center[1] != point[1]:
        return False
    dr = 0 if center[0] == point[0] else (1 if point[0] > center[0] else -1)
    dc = 0 if center[1] == point[1] else (1 if point[1] > center[1] else -1)
    r, c = center[0] + dr, center[1] + dc
    while (r, c) != point:
        if grid[r, c] == WALL:
            return False
        r += dr
        c += dc
    return grid[point] != WALL


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each 3x3 block into wall-bounded cross rays and erase only reached intersections."""
    out = grid.copy()
    blocks = _find_blocks(grid)
    body_cells = set().union(*(b["cells"] for b in blocks)) if blocks else set()
    for block in blocks:
        for r, c, color in _trace(grid, block, body_cells):
            if grid[r, c] == BG:
                out[r, c] = color
            elif (r, c) not in body_cells and grid[r, c] != WALL:
                out[r, c] = color
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 :]:
            for point in ((a["center"][0], b["center"][1]), (b["center"][0], a["center"][1])):
                ar = _reachable_from(grid, a["center"], point)
                br = _reachable_from(grid, b["center"], point)
                if ar and br and point not in body_cells and grid[point] != WALL:
                    out[point] = BG
    for block in blocks:
        for r, c in block["cells"]:
            out[r, c] = block["color"]
    return out


def generate_input() -> np.ndarray:
    h = w = 15
    grid = np.full((h, w), BG, dtype=int)
    centers = random.sample([(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)], random.choice([2, 3]))
    rows = set()
    cols = set()
    chosen = []
    for r, c in centers:
        if r not in rows and c not in cols:
            chosen.append((r, c))
            rows.add(r)
            cols.add(c)
    while len(chosen) < 2:
        r, c = random.choice([(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)])
        if r not in rows and c not in cols:
            chosen.append((r, c))
            rows.add(r)
            cols.add(c)
    for color, (r, c) in zip(random.sample(BLOCK_COLORS, len(chosen)), chosen):
        grid[r - 1 : r + 2, c - 1 : c + 2] = color
    if len(chosen) >= 2 and random.random() < 0.7:
        (r1, c1), (r2, c2) = chosen[:2]
        rr = (r1 + r2) // 2
        c_start, c_end = sorted((c1, c2))
        grid[rr, c_start + 1 : c_end] = WALL
    for _ in range(random.randint(0, 2)):
        if random.random() < 0.5:
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 5)
            grid[r, c : c + 4] = WALL
        else:
            r = random.randint(1, h - 5)
            c = random.randint(1, w - 2)
            grid[r : r + 4, c] = WALL
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
