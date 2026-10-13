import random
from collections import deque

import numpy as np


BACKGROUND = 0
START = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0),
}


def _arrow_cells(direction, length):
    if direction == "R":
        cells = {(1, c) for c in range(length)}
        cells.update({(0, length), (1, length), (2, length), (1, length + 1)})
    elif direction == "L":
        cells = {(1, c) for c in range(2, length + 2)}
        cells.update({(1, 0), (0, 1), (1, 1), (2, 1)})
    elif direction == "D":
        cells = {(r, 1) for r in range(length)}
        cells.update({(length, 0), (length, 1), (length, 2), (length + 1, 1)})
    elif direction == "U":
        cells = {(r, 1) for r in range(2, length + 2)}
        cells.update({(0, 1), (1, 0), (1, 1), (1, 2)})
    else:
        raise ValueError(f"unknown direction: {direction}")
    return cells


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _decode_arrow(cells):
    norm = _normalize(cells)
    for direction in DIRECTIONS:
        for length in range(1, 27):
            if norm == _arrow_cells(direction, length):
                return direction, length
    raise ValueError("component is not a supported arrow shape")


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BACKGROUND, START) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRECTIONS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Erase arrow instruction objects and draw the ordered cursor trail they encode.

    Input structure:
    A black grid contains several separated single-color 4-connected arrow objects, each
    in a distinct non-gray color, plus one gray single-cell cursor start marker. Each
    arrow points left, right, up, or down and has a one-pixel-wide tail whose length is
    the distance instruction.

    Rule:
    Decode every arrow's direction and tail length from its shape, compute its area,
    and execute the arrows from largest area to smallest area. Starting at the gray
    marker, draw the current arrow color on the cursor cell and on every cell reached
    while moving the decoded number of steps in the decoded direction. Later trail
    segments overwrite earlier trail colors. The input arrows and gray marker are
    otherwise erased to black.

    Key insight:
    The object area gives instruction order, while the visible arrow orientation and
    tail length give the movement vector and distance.
    """
    grid = np.asarray(grid)
    starts = np.argwhere(grid == START)
    if len(starts) != 1:
        raise ValueError("expected exactly one gray start pixel")
    cursor = tuple(map(int, starts[0]))

    arrows = []
    for color, cells in _components(grid):
        direction, length = _decode_arrow(cells)
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        arrows.append({
            "color": color,
            "cells": cells,
            "area": len(cells),
            "direction": direction,
            "length": length,
            "top_left": (min_r, min_c),
        })

    out = np.zeros_like(grid)
    for arrow in sorted(arrows, key=lambda a: (-a["area"], a["top_left"])):
        dr, dc = DIRECTIONS[arrow["direction"]]
        out[cursor] = arrow["color"]
        for _ in range(arrow["length"]):
            cursor = (cursor[0] + dr, cursor[1] + dc)
            out[cursor] = arrow["color"]
    return out


def _relative_trail(commands):
    pos = (0, 0)
    cells = {pos}
    for command in sorted(commands, key=lambda x: -(x[0] + 4)):
        length, direction, _ = command
        dr, dc = DIRECTIONS[direction]
        cells.add(pos)
        for _ in range(length):
            pos = (pos[0] + dr, pos[1] + dc)
            cells.add(pos)
    return cells


def _with_margin(cells, h, w, margin=1):
    out = set()
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _place_shape(grid, used, cells, color):
    h, w = grid.shape
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    for _ in range(300):
        top = random.randint(1, h - max_r - 2)
        left = random.randint(1, w - max_c - 2)
        placed = {(top + r, left + c) for r, c in cells}
        if _with_margin(placed, h, w, 1) & used:
            continue
        for r, c in placed:
            grid[r, c] = color
        used.update(_with_margin(placed, h, w, 1))
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = random.randint(16, 30)
        w = random.randint(16, 30)
        n = random.randint(4, 5)
        lengths = random.sample(range(1, 6), n)
        dirs = ["R", "L", "U", "D"]
        if n > 4:
            dirs.append(random.choice(["R", "L", "U", "D"]))
        random.shuffle(dirs)
        colors = random.sample(COLORS, n)
        commands = list(zip(lengths, dirs, colors))

        rel = _relative_trail(commands)
        min_r = min(r for r, _ in rel)
        max_r = max(r for r, _ in rel)
        min_c = min(c for _, c in rel)
        max_c = max(c for _, c in rel)
        r0_lo, r0_hi = 2 - min_r, h - 3 - max_r
        c0_lo, c0_hi = 2 - min_c, w - 3 - max_c
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue
        start = (random.randint(r0_lo, r0_hi), random.randint(c0_lo, c0_hi))
        trail = {(start[0] + r, start[1] + c) for r, c in rel}

        grid = np.zeros((h, w), dtype=np.uint8)
        grid[start] = START
        used = _with_margin(trail, h, w, 1)
        ok = True
        for length, direction, color in commands:
            if not _place_shape(grid, used, _arrow_cells(direction, length), color):
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
