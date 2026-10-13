import random

import numpy as np


BLACK = 0
DIVIDER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PAYLOAD_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each cross-delimited region's payload pixels into its top-left corner.

    Input structure:
    - The background is black.
    - Several colors each form a full-width row and a full-height column that cross
      at the same color, partitioning the grid into rectangles.
    - Region interiors may contain scattered payload pixels whose colors differ from
      all divider colors and from black.

    Rule:
    - Identify every divider row/column pair produced by a same-colored full row and
      full column.
    - Keep divider rows and columns unchanged.
    - For each rectangular region between dividers, read its payload pixels in row-
      major order and repack them densely from that region's top-left corner.
    - Set all remaining cells inside the region to black.

    Key insight:
    - The colored crosses are not ordinary lines; each full row/full column pair is a
    structural partition, and the transformation happens independently inside every
    region they carve out.
    """
    h, w = grid.shape
    row_color = {}
    col_color = {}
    for r in range(h):
        vals = np.unique(grid[r])
        if len(vals) == 1 and vals[0] != BLACK:
            row_color[r] = int(vals[0])
    for c in range(w):
        vals = np.unique(grid[:, c])
        if len(vals) == 1 and vals[0] != BLACK:
            col_color[c] = int(vals[0])
    divider_rows = sorted(r for r, color in row_color.items() if color in col_color.values())
    divider_cols = sorted(c for c, color in col_color.items() if color in row_color.values())
    divider_colors = {row_color[r] for r in divider_rows} | {col_color[c] for c in divider_cols}

    out = grid.copy()
    row_bounds = [-1] + divider_rows + [h]
    col_bounds = [-1] + divider_cols + [w]
    for i in range(len(row_bounds) - 1):
        r0, r1 = row_bounds[i] + 1, row_bounds[i + 1]
        if r0 >= r1:
            continue
        for j in range(len(col_bounds) - 1):
            c0, c1 = col_bounds[j] + 1, col_bounds[j + 1]
            if c0 >= c1:
                continue
            region = grid[r0:r1, c0:c1]
            seq = [int(v) for v in region.ravel() if v not in divider_colors and v != BLACK]
            packed = np.zeros_like(region)
            for k, color in enumerate(seq):
                rr, cc = divmod(k, region.shape[1])
                if rr < region.shape[0]:
                    packed[rr, cc] = color
            out[r0:r1, c0:c1] = packed
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(14, 24)
        w = random.randint(14, 24)
        n_cross = random.randint(2, 3)
        rows = sorted(random.sample(range(2, h - 2), n_cross))
        cols = sorted(random.sample(range(2, w - 2), n_cross))
        if len(set(rows)) < n_cross or len(set(cols)) < n_cross:
            continue
        colors = random.sample(DIVIDER_COLORS, n_cross)
        payload_colors = [c for c in PAYLOAD_COLORS if c not in colors]
        grid = np.zeros((h, w), dtype=int)
        for r, c, color in zip(rows, cols, colors):
            grid[r, :] = color
            grid[:, c] = color
        row_bounds = [-1] + rows + [h]
        col_bounds = [-1] + cols + [w]
        for i in range(len(row_bounds) - 1):
            r0, r1 = row_bounds[i] + 1, row_bounds[i + 1]
            for j in range(len(col_bounds) - 1):
                c0, c1 = col_bounds[j] + 1, col_bounds[j + 1]
                if r1 - r0 <= 0 or c1 - c0 <= 0:
                    continue
                area = (r1 - r0) * (c1 - c0)
                k = random.randint(0, min(area, max(1, area // 3)))
                cells = random.sample([(r, c) for r in range(r0, r1) for c in range(c0, c1)], k)
                for r, c in cells:
                    grid[r, c] = random.choice(payload_colors)
        return grid
