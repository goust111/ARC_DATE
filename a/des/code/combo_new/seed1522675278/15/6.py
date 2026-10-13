import random

import numpy as np


BG = 0
GREY = 5
TYPES = ["H", "V", "R", "HV"]


def _sym_type(arr):
    mask = arr != BG
    h = np.array_equal(mask, mask[::-1, :])
    v = np.array_equal(mask, mask[:, ::-1])
    r = np.array_equal(mask, mask[::-1, ::-1])
    if h and v:
        return "HV"
    if h:
        return "H"
    if v:
        return "V"
    return "R"


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    row_lines = [-1] + [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)] + [grid.shape[0]]
    col_lines = [-1] + [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)] + [grid.shape[1]]
    regions = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            r0, r1 = row_lines[i] + 1, row_lines[i + 1]
            c0, c1 = col_lines[j] + 1, col_lines[j + 1]
            if r0 < r1 and c0 < c1:
                regions.append((r0, r1, c0, c1))
    legend_region = max(regions, key=lambda b: np.count_nonzero(grid[b[0] : b[1], b[2] : b[3]] != BG))
    legend = {}
    sub = grid[legend_region[0] : legend_region[1], legend_region[2] : legend_region[3]]
    for start in range(0, sub.shape[0], 3):
        block = sub[start : start + 3]
        if block.shape[0] < 3 or np.count_nonzero(block) == 0:
            continue
        pattern = block[:, :3]
        color = int(next(v for v in block[:, 4] if v != BG))
        legend[_sym_type(pattern)] = color
    out = grid.copy()
    for box in regions:
        if box == legend_region:
            continue
        r0, r1, c0, c1 = box
        region = grid[r0:r1, c0:c1]
        t = _sym_type(np.where(region == GREY, BG, region))
        cr = r0 + (r1 - r0) // 2
        cc = c0 + (c1 - c0) // 2
        out[cr, cc] = legend[t]
    return out


def _pattern(kind, color):
    arr = np.zeros((3, 3), dtype=int)
    if kind == "H":
        arr[0, 0] = arr[2, 0] = arr[1, 2] = color
    elif kind == "V":
        arr[0, 0] = arr[0, 2] = arr[2, 1] = color
    elif kind == "R":
        arr[0, 0] = arr[1, 2] = arr[2, 2] = color
        arr = np.where(arr + arr[::-1, ::-1] > 0, color, BG)
    else:
        arr[0, 1] = arr[1, 0] = arr[1, 2] = arr[2, 1] = color
    return arr


def generate_input() -> np.ndarray:
    grid = np.zeros((19, 19), dtype=int)
    for r in (6, 12):
        grid[r, :] = GREY
    for c in (6, 12):
        grid[:, c] = GREY
    map_colors = dict(zip(TYPES, random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)))
    legend_box = (0, 6, 0, 6)
    for i, kind in enumerate(TYPES):
        pat = _pattern(kind, random.choice([1, 2, 3, 4]))
        grid[i * 3 : i * 3 + 3, 0:3] = pat
        grid[i * 3 + 1, 4] = map_colors[kind]
    boxes = [(0, 6, 7, 12), (0, 6, 13, 19), (7, 12, 0, 6), (7, 12, 7, 12), (7, 12, 13, 19), (13, 19, 0, 6), (13, 19, 7, 12), (13, 19, 13, 19)]
    for box in boxes:
        kind = random.choice(TYPES)
        color = random.choice([v for v in range(1, 10) if v not in (GREY,)])
        pat = _pattern(kind, color)
        r0, r1, c0, c1 = box
        rr = r0 + (r1 - r0 - 3) // 2
        cc = c0 + (c1 - c0 - 3) // 2
        grid[rr : rr + 3, cc : cc + 3] = np.where(pat != BG, pat, grid[rr : rr + 3, cc : cc + 3])
    return grid
