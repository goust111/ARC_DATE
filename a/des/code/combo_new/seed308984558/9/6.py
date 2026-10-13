import random

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each table cell with row stripes then leftover column-driven colors.

    Input structure:
    - Full-width colored horizontal lines and full-height colored vertical lines form a
      rectangular table on a black background.
    - The enclosed regions start black; each region is bounded by one top line, one
      bottom line, one left line, and one right line, each with its own color.

    Rule:
    - For every region, first fill alternating interior rows using the top/bottom edge
      colors: same colors yield every other row in that color; different colors yield
      alternating top-color and bottom-color rows.
    - Then fill the still-black rows using the left/right edge colors: same colors fill
      whole rows with that color, different colors alternate by column between left and
      right edge colors.
    - Boundary lines remain unchanged.

    Key insight:
    - Each region depends on four edge colors in two stages: horizontal edges define a
      stripe scaffold, and vertical edges only resolve the rows left untouched by that
      first stage.
    """
    out = grid.copy()
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] != BLACK)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] != BLACK)]
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            top = int(grid[r0, c0 + 1])
            bottom = int(grid[r1, c0 + 1])
            left = int(grid[r0 + 1, c0])
            right = int(grid[r0 + 1, c1])
            inner = out[r0 + 1 : r1, c0 + 1 : c1]
            inner[:] = BLACK
            for rr in range(inner.shape[0]):
                if top == bottom:
                    if rr % 2 == 0:
                        inner[rr, :] = top
                else:
                    inner[rr, :] = top if rr % 2 == 0 else bottom
            for rr in range(inner.shape[0]):
                if np.all(inner[rr] != BLACK):
                    continue
                if left == right:
                    inner[rr, :] = left
                else:
                    for cc in range(inner.shape[1]):
                        inner[rr, cc] = left if cc % 2 == 0 else right
    return out


def generate_input() -> np.ndarray:
    row_sizes = [random.randint(3, 5) for _ in range(random.randint(2, 3))]
    col_sizes = [random.randint(3, 5) for _ in range(random.randint(2, 3))]
    h = sum(row_sizes) + len(row_sizes) + 1
    w = sum(col_sizes) + len(col_sizes) + 1
    grid = np.zeros((h, w), dtype=int)
    rows = [0]
    for sz in row_sizes:
        rows.append(rows[-1] + sz + 1)
    cols = [0]
    for sz in col_sizes:
        cols.append(cols[-1] + sz + 1)
    for r in rows:
        grid[r, :] = random.choice(COLORS)
    for c in cols:
        grid[:, c] = random.choice(COLORS)
    return grid
