import random

import numpy as np


BG = 0
FRAME = 5
MARK = 2


def _scale(pattern: np.ndarray, shape):
    h, w = shape
    ph, pw = pattern.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        sr = min(ph - 1, int(r * ph / h))
        for c in range(w):
            sc = min(pw - 1, int(c * pw / w))
            out[r, c] = int(pattern[sr, sc])
    return out


def _find_library(grid: np.ndarray):
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for cr, cc in corners:
        if grid[cr, cc] != FRAME:
            continue
        rs = range(h) if cr == 0 else range(h - 1, -1, -1)
        cs = range(w) if cc == 0 else range(w - 1, -1, -1)
        r_end = next(r for r in rs if grid[r, cc] != FRAME) - (1 if cr == 0 else -1)
        c_end = next(c for c in cs if grid[cr, c] != FRAME) - (1 if cc == 0 else -1)
        r0, r1 = sorted((cr, r_end))
        c0, c1 = sorted((cc, c_end))
        if r1 - r0 < 4 or c1 - c0 < 8:
            continue
        return r0, c0, r1, c1
    raise AssertionError("library not found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the red markers inside each framed region, pick that indexed library pattern, and scale it to fill the region interior.

    Input structure:
    - Black background with several grey rectangular frames.
    - One corner contains a larger grey-framed horizontal pattern library split into equal cells by grey dividers.
    - Every non-library frame contains 1 to 4 red marker pixels inside it.

    Rule:
    - Read the corner library as an ordered left-to-right list of patterns.
    - For every other grey frame, count its red interior markers to obtain N.
    - Select the N-th library pattern, nearest-neighbor scale it to the frame's interior size, and replace the entire interior with that scaled pattern.
    - Keep the library unchanged.

    Key insight:
    - The red pixels are not output content; they only index which stored pattern to fetch before resizing it to the destination frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    lr0, lc0, lr1, lc1 = _find_library(g)
    inner = g[lr0 + 1 : lr1, lc0 + 1 : lc1]
    divider_cols = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == FRAME)]
    cuts = [-1] + divider_cols + [inner.shape[1]]
    patterns = []
    for i in range(len(cuts) - 1):
        c0 = cuts[i] + 1
        c1 = cuts[i + 1]
        if c1 - c0 <= 0:
            continue
        patterns.append(inner[:, c0:c1].copy())
    for r0 in range(g.shape[0]):
        for c0 in range(g.shape[1]):
            if g[r0, c0] != FRAME:
                continue
            for r1 in range(r0 + 2, g.shape[0]):
                if g[r1, c0] != FRAME:
                    break
                for c1 in range(c0 + 2, g.shape[1]):
                    if g[r0, c1] != FRAME:
                        break
                    if (r0, c0, r1, c1) == (lr0, lc0, lr1, lc1):
                        continue
                    if np.all(g[r0, c0 : c1 + 1] == FRAME) and np.all(g[r1, c0 : c1 + 1] == FRAME) and np.all(g[r0 : r1 + 1, c0] == FRAME) and np.all(g[r0 : r1 + 1, c1] == FRAME):
                        interior = g[r0 + 1 : r1, c0 + 1 : c1]
                        count = int(np.count_nonzero(interior == MARK))
                        if 1 <= count <= len(patterns):
                            out[r0 + 1 : r1, c0 + 1 : c1] = _scale(patterns[count - 1], interior.shape)
    return out


def _random_pattern():
    arr = np.zeros((3, 3), dtype=int)
    color = random.choice([1, 3, 4, 6, 7, 8, 9])
    kind = random.choice(["diag", "cross", "corner", "stripe"])
    if kind == "diag":
        for i in range(3):
            arr[i, i] = color
    elif kind == "cross":
        arr[1, :] = color
        arr[:, 1] = color
    elif kind == "corner":
        arr[0, :2] = color
        arr[:2, 0] = color
    else:
        arr[:, 1] = color
    return arr


def generate_input() -> np.ndarray:
    for _ in range(500):
        patterns = [_random_pattern() for _ in range(4)]
        cell_h, cell_w = 3, 3
        lib_h = cell_h + 2
        lib_w = 4 * cell_w + 5
        h = random.randint(max(16, lib_h + 8), 28)
        w = random.randint(max(18, lib_w + 2), 28)
        grid = np.zeros((h, w), dtype=int)
        grid[0, 0:lib_w] = FRAME
        grid[lib_h - 1, 0:lib_w] = FRAME
        grid[0:lib_h, 0] = FRAME
        grid[0:lib_h, lib_w - 1] = FRAME
        for k in range(1, 4):
            col = k * (cell_w + 1)
            grid[1 : lib_h - 1, col] = FRAME
        for idx, pat in enumerate(patterns):
            c0 = 1 + idx * (cell_w + 1)
            grid[1 : 1 + cell_h, c0 : c0 + cell_w] = pat

        occupied = {(r, c) for r in range(lib_h + 1) for c in range(lib_w + 1)}
        for _rect in range(random.randint(2, 4)):
            placed = False
            for _attempt in range(120):
                ih = random.randint(3, 7)
                iw = random.randint(3, 7)
                r0 = random.randint(lib_h + 1, h - ih - 2)
                c0 = random.randint(1, w - iw - 2)
                box = {(r, c) for r in range(r0 - 1, r0 + ih + 2) for c in range(c0 - 1, c0 + iw + 2)}
                if box & occupied:
                    continue
                occupied |= box
                grid[r0, c0 : c0 + iw + 2] = FRAME
                grid[r0 + ih + 1, c0 : c0 + iw + 2] = FRAME
                grid[r0 : r0 + ih + 2, c0] = FRAME
                grid[r0 : r0 + ih + 2, c0 + iw + 1] = FRAME
                count = random.randint(1, 4)
                cells = random.sample([(r, c) for r in range(r0 + 1, r0 + ih + 1) for c in range(c0 + 1, c0 + iw + 1)], count)
                for r, c in cells:
                    grid[r, c] = MARK
                placed = True
                break
            if not placed:
                break
        else:
            return grid
    raise RuntimeError("failed to generate valid input")
