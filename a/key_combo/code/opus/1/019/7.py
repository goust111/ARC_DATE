import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Split terrain by marker bands and highlight each band's local extreme columns.

    Input structure: Gray terrain rises continuously from the bottom of each
    column, and colored singleton markers above the terrain define horizontal
    bands by row-order midpoints.
    Rule: In each marker band, compute gray terrain counts per column restricted
    to that band. The maximum-count column is colored with the band marker color
    above the terrain, and the minimum positive-count column is colored yellow.
    Key insight: Terrain extremes are local to each horizontal band rather than
    global over the whole landscape.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    markers = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w)
               if grid[r, c] not in (0, 5)]
    markers.sort(key=lambda x: x[0])
    if not markers:
        return out
    mids = [(markers[i][0] + markers[i + 1][0]) // 2 for i in range(len(markers) - 1)]
    for i, (mr, mc, color) in enumerate(markers):
        start = 0 if i == 0 else mids[i - 1]
        end = h if i == len(markers) - 1 else mids[i]
        counts = []
        for c in range(w):
            n = int(np.sum(grid[start:end, c] == 5))
            if n > 0:
                counts.append((c, n))
        if not counts:
            continue
        max_n = max(n for _, n in counts)
        min_n = min(n for _, n in counts)
        shallow_col = max(c for c, n in counts if n == max_n)
        deep_col = min(c for c, n in counts if n == min_n)
        for r in range(start, end):
            if out[r, shallow_col] == 0 or (r, shallow_col) == (mr, mc):
                out[r, shallow_col] = color
            if out[r, deep_col] == 0 or (r, deep_col) == (mr, mc):
                out[r, deep_col] = 4
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 20)
    w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    heights = []
    for c in range(w):
        # Wide spread makes each band see different local extremes.
        height = random.randint(3, h - 2)
        heights.append(height)
        grid[h - height:h, c] = 5
    rows = sorted(random.sample(range(3, h - 4), random.randint(2, 4)))
    while any(b - a < 4 for a, b in zip(rows, rows[1:])):
        rows = sorted(random.sample(range(3, h - 4), random.randint(2, 4)))
    colors = random.sample([1, 2, 3, 7], len(rows))
    for row, color in zip(rows, colors):
        possible = [c for c, height in enumerate(heights) if row < h - height]
        if not possible:
            possible = list(range(w))
        col = random.choice(possible)
        grid[row, col] = color
    return grid

