import random
from collections import deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _legend(grid):
    h, w = grid.shape
    candidates = []
    for r in [0, h - 1]:
        for c in range(w - 2):
            vals = [int(grid[r, c + i]) for i in range(3)]
            if BG not in vals and len(set(vals)) == 3:
                candidates.append([(r, c + i) for i in range(3)])
    for c in [0, w - 1]:
        for r in range(h - 2):
            vals = [int(grid[r + i, c]) for i in range(3)]
            if BG not in vals and len(set(vals)) == 3:
                candidates.append([(r + i, c) for i in range(3)])
    if not candidates:
        return [], set(), {}
    cells = list(candidates[0])
    chain = [int(grid[r, c]) for r, c in cells]
    beats = {chain[0]: chain[1], chain[1]: chain[2], chain[2]: chain[0]}
    return chain, set(cells), beats


def _components(grid, colors, skip):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if (r, c) in skip or seen[r, c] or int(grid[r, c]) not in colors:
                continue
            col = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in skip and not seen[nr, nc] and int(grid[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [x for x, _ in cells]
            cs = [y for _, y in cells]
            comps.append({"color": col, "bbox": (min(rs), min(cs), max(rs), max(cs)), "cells": set(cells)})
    return comps


def _overlap(a0, a1, b0, b1):
    return max(a0, b0) <= min(a1, b1)


def main(grid: np.ndarray) -> np.ndarray:
    chain, legend_cells, beats = _legend(grid)
    if not chain:
        return grid.copy()
    comps = _components(grid, set(chain), legend_cells)
    out = grid.copy()
    actions = []
    for i, a in enumerate(comps):
        for j, b in enumerate(comps):
            if i == j or beats.get(a["color"]) != b["color"]:
                continue
            ar0, ac0, ar1, ac1 = a["bbox"]
            br0, bc0, br1, bc1 = b["bbox"]
            if _overlap(ar0, ar1, br0, br1) or _overlap(ac0, ac1, bc0, bc1):
                actions.append((min(ar0, br0), min(ac0, bc0), max(ar1, br1), max(ac1, bc1), a["color"]))
    for r0, c0, r1, c1, color in actions:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    for r, c in legend_cells:
        out[r, c] = grid[r, c]
    return out


def _free(grid, r0, c0, r1, c1):
    h, w = grid.shape
    if not (1 <= r0 <= r1 < h - 1 and 1 <= c0 <= c1 < w - 1):
        return False
    return np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG)


def _place_rect(grid, color, r0, c0, h, w):
    r1, c1 = r0 + h - 1, c0 + w - 1
    if not _free(grid, r0, c0, r1, c1):
        return None
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return (r0, c0, r1, c1)


def _random_rect(grid, color):
    h, w = grid.shape
    for _ in range(100):
        rh = random.randint(2, 5)
        rw = random.randint(2, 5)
        r0 = random.randint(2, h - rh - 2)
        c0 = random.randint(2, w - rw - 2)
        box = _place_rect(grid, color, r0, c0, rh, rw)
        if box is not None:
            return box
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(20, 32)
        w = random.randint(20, 32)
        grid = np.zeros((h, w), dtype=int)
        chain = random.sample(COLORS, 3)
        if random.random() < 0.5:
            start = random.randint(2, w - 5)
            for i, color in enumerate(chain):
                grid[0, start + i] = color
        else:
            start = random.randint(2, h - 5)
            for i, color in enumerate(chain):
                grid[start + i, 0] = color
        winner, loser = chain[0], chain[1]
        row = random.randint(5, h - 8)
        ah, aw = random.randint(2, 4), random.randint(2, 4)
        bh, bw = random.randint(2, 4), random.randint(2, 4)
        c0 = random.randint(2, max(2, w // 2 - aw - 2))
        gap = random.randint(2, 5)
        b0 = c0 + aw + gap
        if b0 + bw >= w - 2:
            continue
        if _place_rect(grid, winner, row, c0, ah, aw) is None:
            continue
        if _place_rect(grid, loser, row + random.randint(0, max(0, ah - 1)), b0, bh, bw) is None:
            continue
        for color in chain:
            for _j in range(random.randint(1, 2)):
                _random_rect(grid, color)
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate seed138090197/4/5 input")
