import random
from collections import deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


L_MASKS = {
    (0, 0): np.array([[0, 1], [1, 1]], dtype=bool),
    (0, 1): np.array([[1, 0], [1, 1]], dtype=bool),
    (1, 0): np.array([[1, 1], [0, 1]], dtype=bool),
    (1, 1): np.array([[1, 1], [1, 0]], dtype=bool),
}


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _key_column(grid):
    best_col = None
    best = []
    for c in range(grid.shape[1] - 1, -1, -1):
        vals = [(r, int(grid[r, c])) for r in range(grid.shape[0]) if grid[r, c] != BG]
        if len(vals) > len(best):
            best_col = c
            best = vals
    colors = [v for _, v in sorted(best)]
    return best_col, colors


def _components(grid, ignore_col):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if c == ignore_col or color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if nc != ignore_col and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(cells) == 3:
                rows = [x for x, _ in cells]
                cols = [y for _, y in cells]
                if max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1:
                    mask = np.zeros((2, 2), dtype=bool)
                    for rr, cc in cells:
                        mask[rr - min(rows), cc - min(cols)] = True
                    missing = tuple(int(x[0]) for x in np.argwhere(~mask))
                    comps.append({
                        "color": color,
                        "cells": cells,
                        "center": ((min(rows) + max(rows)) / 2, (min(cols) + max(cols)) / 2),
                        "mask": mask,
                        "missing": missing,
                    })
    return comps


def _compose(a, b):
    sq = np.zeros((2, 2), dtype=int)
    for r, c in zip(*np.where(a["mask"])):
        sq[r, c] = a["color"]
    for r, c in zip(*np.where(b["mask"])):
        if sq[r, c] == BG:
            sq[r, c] = b["color"]
    for r in range(2):
        for c in range(2):
            if sq[r, c] == BG:
                sq[r, c] = b["color"]
    return sq


def main(grid: np.ndarray) -> np.ndarray:
    """Pair L fragments into 2x2 squares and sort them by a right-side color key.

    Input structure:
    The black grid contains scattered three-pixel L fragments and a right-side column
    of one-pixel color keys.  Fragment colors all appear in the key column.

    Rule:
    Nearby L fragments of different colors are paired and overlaid into a 2x2 square.
    The resulting squares are sorted left to right by the key-column position of the
    square's upper-left color.  The output has height 2 and width two times the number
    of pairs.

    Key insight:
    Shape complementarity supplies pairing, while the right-side color list supplies
    the ordering key using the color in the square's upper-left corner.
    """
    key_col, key = _key_column(grid)
    order = {color: i for i, color in enumerate(key)}
    comps = _components(grid, key_col)
    unused = set(range(len(comps)))
    squares = []
    while unused:
        i = min(unused)
        unused.remove(i)
        a = comps[i]
        choices = []
        for j in unused:
            b = comps[j]
            if b["color"] == a["color"]:
                continue
            dist = abs(a["center"][0] - b["center"][0]) + abs(a["center"][1] - b["center"][1])
            choices.append((dist, j))
        if not choices:
            continue
        _, j = min(choices)
        unused.remove(j)
        squares.append(_compose(a, comps[j]))
    squares.sort(key=lambda sq: order.get(int(sq[0, 0]), 99))
    out = np.zeros((2, max(1, 2 * len(squares))), dtype=int)
    for i, sq in enumerate(squares):
        out[:, 2 * i:2 * i + 2] = sq
    return out


def _place_l(grid, top, left, missing, color):
    mask = L_MASKS[missing]
    cells = [(top + int(r), left + int(c)) for r, c in zip(*np.where(mask))]
    margin = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if any(r < 1 or r >= grid.shape[0] - 1 or c < 1 or c >= grid.shape[1] - 2 for r, c in cells):
        return False
    if any(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != BG for r, c in margin):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(1000):
        n_pairs = random.randint(2, 4)
        h = random.randint(max(14, 2 * (2 * n_pairs) + 2), 26)
        w = random.randint(14, 28)
        grid = np.zeros((h, w), dtype=int)
        colors = random.sample(COLORS, 2 * n_pairs)
        key_order = colors[:]
        random.shuffle(key_order)
        key_col = w - 1
        for i, color in enumerate(key_order):
            grid[1 + i * 2, key_col] = color
        ok = True
        for i in range(n_pairs):
            base_r = random.randint(2, h - 5)
            base_c = random.randint(2, w - 7)
            if i % 2:
                base_r = min(h - 5, base_r + 3)
            m1, m2 = random.sample(list(L_MASKS), 2)
            if not _place_l(grid, base_r, base_c, m1, colors[2 * i]):
                ok = False
                break
            if not _place_l(grid, base_r + random.choice([-3, 3]), base_c + random.choice([-3, 3]), m2, colors[2 * i + 1]):
                ok = False
                break
        if ok:
            out = main(grid)
            if out.shape[1] == 2 * n_pairs:
                return grid
    raise RuntimeError("failed to generate a valid L-pair sorting puzzle")
