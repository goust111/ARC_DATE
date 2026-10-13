import random

import numpy as np


def _decode(top):
    instr = []
    cols = np.where(np.any(top == 2, axis=0))[0]
    if cols.size == 0:
        return instr
    runs = []
    start = prev = int(cols[0])
    for c in map(int, cols[1:]):
        if c == prev + 1:
            prev = c
        else:
            runs.append((start, prev))
            start = prev = c
    runs.append((start, prev))
    for c0, c1 in runs:
        crop = top[:, c0 : c1 + 1] == 2
        if crop.shape[1] == 1:
            instr.append("F")
            continue
        top_cols = np.where(crop[0])[0]
        bot_cols = np.where(crop[1])[0]
        if len(top_cols) == 0 or len(bot_cols) == 0:
            continue
        entry = int(top_cols[0])
        extra = [int(c) for c in bot_cols if int(c) != entry]
        instr.append("L" if extra and extra[0] < entry else "R")
    return instr


def _chambers(grid):
    lower = grid[3:, :]
    pts = np.argwhere(lower == 8)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    box = lower[r0 : r1 + 1, c0 : c1 + 1]
    srows = [int(i) + r0 + 3 for i in range(box.shape[0]) if np.count_nonzero(box[i] == 8) > box.shape[1] // 2]
    scols = [int(i) + c0 for i in range(box.shape[1]) if np.count_nonzero(box[:, i] == 8) > box.shape[0] // 2]
    row_spans = [(srows[i] + 1, srows[i + 1] - 1) for i in range(len(srows) - 1) if srows[i + 1] - srows[i] > 1]
    col_spans = [(scols[i] + 1, scols[i + 1] - 1) for i in range(len(scols) - 1) if scols[i + 1] - scols[i] > 1]
    return row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """Decode top glyphs as chamber-navigation instructions and fill every lattice chamber visited by the path."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    instr = _decode(g[:2, :])
    row_spans, col_spans = _chambers(g)
    marker = np.argwhere(g == 3)[0]
    cr = next(i for i, (a, b) in enumerate(row_spans) if a <= marker[0] <= b)
    cc = next(i for i, (a, b) in enumerate(col_spans) if a <= marker[1] <= b)
    direction = 1
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = {(cr, cc)}
    for op in instr:
        if op == "L":
            direction = (direction - 1) % 4
        elif op == "R":
            direction = (direction + 1) % 4
        dr, dc = dirs[direction]
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < len(row_spans) and 0 <= nc < len(col_spans):
            cr, cc = nr, nc
            visited.add((cr, cc))
    for rr, cc in visited:
        r0, r1 = row_spans[rr]
        c0, c1 = col_spans[cc]
        out[r0 : r1 + 1, c0 : c1 + 1][g[r0 : r1 + 1, c0 : c1 + 1] == 0] = 2
    out[marker[0], marker[1]] = 3
    return out


def _write_glyph(grid, col, op):
    if op == "F":
        grid[0:2, col] = 2
        return col + 2
    if op == "L":
        grid[0, col + 1] = 2
        grid[1, col : col + 2] = 2
    else:
        grid[0, col] = 2
        grid[1, col : col + 2] = 2
    return col + 3


def generate_input() -> np.ndarray:
    length = random.randint(4, 7)
    prog = [random.choice(["F", "F", "L", "R"]) for _ in range(length)]
    if "F" not in prog:
        prog[random.randrange(length)] = "F"
    rows, cols = 3, 5
    h, w = 3 + rows * 4 + 1, cols * 4 + 1
    grid = np.zeros((h, w), dtype=int)
    col = 0
    for op in prog:
        col = _write_glyph(grid, col, op)
    base = 3
    for r in range(rows + 1):
        grid[base + 4 * r, :w] = 8
    for c in range(cols + 1):
        grid[base : base + rows * 4 + 1, 4 * c] = 8
    grid[base + 2 + random.choice([0, 4]), 2 + random.choice([0, 4])] = 3
    return grid
