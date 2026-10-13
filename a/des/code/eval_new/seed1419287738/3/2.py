import random

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


ARROWS = {
    (0, 1): lambda r, c: {(r, c), (r - 1, c + 1), (r + 1, c + 1)},
    (0, -1): lambda r, c: {(r, c), (r - 1, c - 1), (r + 1, c - 1)},
    (1, 0): lambda r, c: {(r, c), (r + 1, c - 1), (r + 1, c + 1)},
    (-1, 0): lambda r, c: {(r, c), (r - 1, c - 1), (r - 1, c + 1)},
}


def _rectangles(grid):
    grey = np.argwhere(grid == GREY)
    rows = sorted(set(int(r) for r, _ in grey))
    cols = sorted(set(int(c) for _, c in grey))
    rects = []
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0 + 2:
                continue
            for c0 in cols:
                for c1 in cols:
                    if c1 <= c0 + 2:
                        continue
                    if np.all(grid[r0, c0:c1 + 1] == GREY) and np.all(grid[r1, c0:c1 + 1] == GREY):
                        if np.all(grid[r0:r1 + 1, c0] == GREY) and np.all(grid[r0:r1 + 1, c1] == GREY):
                            if not np.any(grid[r0 + 1:r1, c0 + 1:c1] == GREY):
                                rects.append((r0, r1, c0, c1))
    unique = []
    seen = set()
    for rect in rects:
        if rect not in seen:
            seen.add(rect)
            unique.append(rect)
    return unique


def _detect_arrows(grid):
    h, w = grid.shape
    arrows = []
    seen = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color in (BG, GREY):
                continue
            for direction, shape_fn in ARROWS.items():
                cells = shape_fn(r, c)
                if all(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color for rr, cc in cells):
                    key = frozenset(cells)
                    if key not in seen:
                        seen.add(key)
                        arrows.append({"center": (r, c), "cells": cells, "dir": direction, "color": color})
    return arrows


def _target_rect(arrow, rects):
    r, c = arrow["center"]
    dr, dc = arrow["dir"]
    candidates = []
    for rect in rects:
        r0, r1, c0, c1 = rect
        if dc == 1 and r0 < r < r1 and c < c0:
            candidates.append((c0 - c, rect))
        elif dc == -1 and r0 < r < r1 and c > c1:
            candidates.append((c - c1, rect))
        elif dr == 1 and c0 < c < c1 and r < r0:
            candidates.append((r0 - r, rect))
        elif dr == -1 and c0 < c < c1 and r > r1:
            candidates.append((r - r1, rect))
    return min(candidates)[1] if candidates else None


def _sequence(grid, arrow, rect):
    r, c = arrow["center"]
    dr, dc = arrow["dir"]
    r += dr
    c += dc
    seq = []
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != GREY:
        if (r, c) not in arrow["cells"] and grid[r, c] not in (BG, GREY):
            seq.append(int(grid[r, c]))
        r += dr
        c += dc
    return seq


def _push(out, rect, arrow, seq):
    if not seq:
        return
    r0, r1, c0, c1 = rect
    ar, ac = arrow["center"]
    dr, dc = arrow["dir"]
    if dc == 1:
        row = ar
        old = [int(out[row, c]) for c in range(c0 + 1, c1) if out[row, c] != BG]
        vals = (seq + old)[:c1 - c0 - 1]
        out[row, c0 + 1:c1] = BG
        out[row, c0 + 1:c0 + 1 + len(vals)] = vals
    elif dc == -1:
        row = ar
        old = [int(out[row, c]) for c in range(c1 - 1, c0, -1) if out[row, c] != BG]
        vals = (seq + old)[:c1 - c0 - 1]
        out[row, c0 + 1:c1] = BG
        for i, val in enumerate(vals):
            out[row, c1 - 1 - i] = val
    elif dr == 1:
        col = ac
        old = [int(out[r, col]) for r in range(r0 + 1, r1) if out[r, col] != BG]
        vals = (seq + old)[:r1 - r0 - 1]
        out[r0 + 1:r1, col] = BG
        for i, val in enumerate(vals):
            out[r0 + 1 + i, col] = val
    else:
        col = ac
        old = [int(out[r, col]) for r in range(r1 - 1, r0, -1) if out[r, col] != BG]
        vals = (seq + old)[:r1 - r0 - 1]
        out[r0 + 1:r1, col] = BG
        for i, val in enumerate(vals):
            out[r1 - 1 - i, col] = val


def main(grid: np.ndarray) -> np.ndarray:
    """Push colored sequences from arrows into adjacent grey containers.

    Input structure:
    Grey rectangular containers have black interiors with optional colored contents.
    Outside them, colored sequences end in three-pixel arrow markers pointing toward
    a container row or column.

    Rule:
    Each arrow selects the nearest container in its pointing direction and the colored
    sequence between arrow and border.  Sequences are pushed into the corresponding
    interior row or column from the entry side; existing contents are shifted deeper
    and overflow disappears.  Arrow and exterior sequence pixels are erased.

    Key insight:
    The arrow groups the exterior pixels and also supplies the stack direction; near
    pushes on the same line must happen before farther pushes.
    """
    out = np.array(grid, copy=True)
    rects = _rectangles(grid)
    ops = []
    for arrow in _detect_arrows(grid):
        rect = _target_rect(arrow, rects)
        if rect is None:
            continue
        seq = _sequence(grid, arrow, rect)
        if not seq:
            continue
        r, c = arrow["center"]
        r0, r1, c0, c1 = rect
        dist = min(abs(r - r0), abs(r - r1), abs(c - c0), abs(c - c1))
        ops.append((dist, arrow, rect, seq))
        for rr, cc in arrow["cells"]:
            out[rr, cc] = BG
    for _, arrow, rect, seq in sorted(ops, key=lambda x: x[0]):
        r, c = arrow["center"]
        dr, dc = arrow["dir"]
        rr, cc = r + dr, c + dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] != GREY:
            out[rr, cc] = BG
            rr += dr
            cc += dc
        _push(out, rect, arrow, seq)
    return out


def _draw_rect(grid, r0, c0, ih, iw):
    r1, c1 = r0 + ih + 1, c0 + iw + 1
    if np.any(grid[r0:r1 + 1, c0:c1 + 1] != BG):
        return None
    grid[r0, c0:c1 + 1] = GREY
    grid[r1, c0:c1 + 1] = GREY
    grid[r0:r1 + 1, c0] = GREY
    grid[r0:r1 + 1, c1] = GREY
    for _ in range(random.randint(0, 3)):
        grid[random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)] = random.choice(COLORS)
    return (r0, r1, c0, c1)


def _draw_arrow(grid, center, direction, color):
    cells = ARROWS[direction](*center)
    if any(grid[r, c] != BG for r, c in cells):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(18, 29)
        w = random.randint(18, 29)
        grid = np.zeros((h, w), dtype=int)
        rect = _draw_rect(grid, random.randint(5, h - 10), random.randint(5, w - 10), random.randint(3, 5), random.randint(4, 6))
        if rect is None:
            continue
        r0, r1, c0, c1 = rect
        sides = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(sides)
        ok = True
        for direction in sides[:random.randint(2, 4)]:
            color = random.choice(COLORS)
            if direction == (0, 1):
                row = random.randint(r0 + 1, r1 - 1)
                center = (row, max(2, c0 - random.randint(4, 6)))
                seq_cols = range(center[1] + 2, c0)
                for c in seq_cols:
                    grid[row, c] = color
            elif direction == (0, -1):
                row = random.randint(r0 + 1, r1 - 1)
                center = (row, min(w - 3, c1 + random.randint(4, 6)))
                for c in range(c1 + 1, center[1] - 1):
                    grid[row, c] = color
            elif direction == (1, 0):
                col = random.randint(c0 + 1, c1 - 1)
                center = (max(2, r0 - random.randint(4, 6)), col)
                for r in range(center[0] + 2, r0):
                    grid[r, col] = color
            else:
                col = random.randint(c0 + 1, c1 - 1)
                center = (min(h - 3, r1 + random.randint(4, 6)), col)
                for r in range(r1 + 1, center[0] - 1):
                    grid[r, col] = color
            if not _draw_arrow(grid, center, direction, color):
                ok = False
                break
        if ok and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid push-container puzzle")
