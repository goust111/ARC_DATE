import random
import numpy as np


BG = 0


def _rect_candidates(grid: np.ndarray):
    rects = []
    for color in [int(c) for c in np.unique(grid) if c != BG]:
        pts = np.argwhere(grid == color)
        if len(pts) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if np.all(grid[r0, c0 : c1 + 1] == color) and np.all(grid[r1, c0 : c1 + 1] == color) and np.all(grid[r0 : r1 + 1, c0] == color) and np.all(grid[r0 : r1 + 1, c1] == color):
            rects.append((color, r0, r1, c0, c1))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored fence interior by tiling the legend pattern with the same border color.

    Input structure:
    - A legend strip contains small bordered squares with color-keyed interior patterns.
    - The main area contains larger hollow rectangles whose border colors match legend squares.

    Rule:
    - Read each legend square's interior pattern.
    - For every larger rectangle with the same border color, tile that pattern from the
      rectangle interior's top-left corner until the interior is filled.
    - Keep both borders and the legend unchanged.

    Key insight:
    - Border color acts only as a lookup key; the fill is the repeated pattern hidden
      inside the corresponding legend square.
    """
    out = grid.copy()
    rects = _rect_candidates(grid)
    legends = {}
    for color, r0, r1, c0, c1 in rects:
        if r1 < grid.shape[0] // 3:
            legends[color] = grid[r0 + 1 : r1, c0 + 1 : c1]
    for color, r0, r1, c0, c1 in rects:
        if color not in legends or r1 < grid.shape[0] // 3:
            continue
        pat = legends[color]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = pat[(r - (r0 + 1)) % pat.shape[0], (c - (c0 + 1)) % pat.shape[1]]
    return out


PATS = [
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[1, 1], [0, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
]


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    patterns = random.sample(PATS, len(colors))
    x = 1
    for color, pat in zip(colors, patterns):
        h, w = pat.shape[0] + 2, pat.shape[1] + 2
        grid[1, x : x + w] = color
        grid[h, x : x + w] = color
        grid[1 : h + 1, x] = color
        grid[1 : h + 1, x + w - 1] = color
        block = grid[2:h, x + 1 : x + w - 1]
        block[:] = pat
        block[block == 1] = color
        x += w + 2
    for color, pat in zip(colors, patterns):
        scale = random.randint(2, 3)
        ih, iw = pat.shape[0] * scale, pat.shape[1] * scale
        h, w = ih + 2, iw + 2
        for _ in range(200):
            r = random.randint(8, grid.shape[0] - h - 1)
            c = random.randint(1, grid.shape[1] - w - 1)
            if np.all(grid[r : r + h, c : c + w] == 0):
                grid[r, c : c + w] = color
                grid[r + h - 1, c : c + w] = color
                grid[r : r + h, c] = color
                grid[r : r + h, c + w - 1] = color
                break
    return grid
