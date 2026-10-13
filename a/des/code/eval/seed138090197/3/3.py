import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
COLORS = [1, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(grid, r, c):
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def _source_blocks(grid):
    h, w = grid.shape
    blocks = []
    seen = set()
    for r in range(h - 1):
        for c in range(w - 1):
            col = int(grid[r, c])
            if col in (BG, RED, GREY):
                continue
            cells = ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1))
            if all(int(grid[rr, cc]) == col for rr, cc in cells):
                key = tuple(sorted(cells))
                if key not in seen:
                    seen.add(key)
                    blocks.append({"color": col, "cells": set(cells), "bbox": (r, c, r + 1, c + 1)})
    return blocks


def _entry_from_source(grid, block):
    cells = block["cells"]
    col = block["color"]
    candidates = []
    for r, c in cells:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells or not _in_bounds(grid, nr, nc):
                continue
            if int(grid[nr, nc]) in (col, RED):
                candidates.append((nr, nc, dr, dc))
    if not candidates:
        return None
    # Prefer cells that are not adjacent to two source cells; they are the wire endpoint.
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0]


def _turn_at_cross(grid, r, c, dr, dc, prev):
    options = []
    if dr == 0:
        dirs = [(-1, 0), (1, 0)]
    else:
        dirs = [(0, -1), (0, 1)]
    for nd in dirs:
        nr, nc = r + nd[0], c + nd[1]
        if (nr, nc) == prev or not _in_bounds(grid, nr, nc):
            continue
        if int(grid[nr, nc]) not in (BG, GREY):
            options.append(nd)
    if options:
        return options[0]
    return dr, dc


def _trace(grid, start, dr, dc):
    r, c = start
    prev = None
    count = 0
    seen = set()
    while _in_bounds(grid, r, c) and int(grid[r, c]) not in (BG, GREY):
        state = (r, c, dr, dc)
        if state in seen:
            break
        seen.add(state)
        count += 1
        if int(grid[r, c]) == RED:
            dr, dc = _turn_at_cross(grid, r, c, dr, dc, prev)
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid, nr, nc) or int(grid[nr, nc]) in (BG, GREY):
            break
        prev = (r, c)
        r, c = nr, nc
    anchor = None
    for ar, ac in [(r + dr, c + dc)] + [(r + x, c + y) for x, y in DIRS]:
        if _in_bounds(grid, ar, ac) and int(grid[ar, ac]) == GREY:
            anchor = (ar, ac)
            break
    return count, anchor


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for block in _source_blocks(grid):
        entry = _entry_from_source(grid, block)
        if entry is None:
            continue
        r, c, dr, dc = entry
        length, anchor = _trace(grid, (r, c), dr, dc)
        if anchor is None or length <= 0:
            continue
        ar, ac = anchor
        for rr in range(ar, min(grid.shape[0], ar + length)):
            out[rr, ac] = block["color"]
    return out


def _can_place(grid, cells):
    h, w = grid.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if int(grid[r, c]) != BG:
            return False
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, GREY):
                return False
    return True


def _draw_wire_case(grid, color):
    h, w = grid.shape
    for _ in range(200):
        orient = random.choice(["h", "v"])
        length = random.randint(3, 8)
        if orient == "h":
            hi_r = h - 4
            if hi_r < 2:
                continue
            r = random.randint(2, hi_r)
            c = random.randint(3, w - length - 5)
            source = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            wire = [(r, c + 2 + i) for i in range(length)]
            anchor = [(r, c + 2 + length)]
        else:
            hi_r = h - length - 4
            if hi_r < 3:
                continue
            r = random.randint(3, hi_r)
            c = random.randint(2, w - 4)
            source = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            wire = [(r + 2 + i, c) for i in range(length)]
            anchor = [(r + 2 + length, c)]
        cells = source + wire + anchor
        if _can_place(grid, cells):
            for rr, cc in source + wire:
                grid[rr, cc] = color
            for rr, cc in anchor:
                grid[rr, cc] = GREY
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        grid = np.zeros((h, w), dtype=int)
        colors = random.sample(COLORS, random.randint(1, 3))
        ok = True
        for color in colors:
            if not _draw_wire_case(grid, color):
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate seed138090197/3/3 input")
