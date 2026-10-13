import random
import numpy as np


BG = 0
RED = 2
GREY = 5
BLUE = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red markers in each grey-bounded region and draw that many inward blue rings.

    Input structure:
    - Grey full rows and columns partition the black background into rectangular regions.
    - Each region contains zero or more red marker pixels.

    Rule:
    - For every region, count its red markers.
    - If the count is positive, paint that many nested blue rectangular rings from the
      region boundary inward, stopping naturally if the region becomes too small.
    - Remove the red markers and keep all grey partition lines unchanged.

    Key insight:
    - Only the count of markers matters; their exact locations are irrelevant once the
      region they belong to has been identified.
    """
    out = grid.copy()
    grey_rows = [-1] + [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)] + [grid.shape[0]]
    grey_cols = [-1] + [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)] + [grid.shape[1]]
    out[out == RED] = BG
    for r0, r1 in zip(grey_rows, grey_rows[1:]):
        for c0, c1 in zip(grey_cols, grey_cols[1:]):
            rs, re = r0 + 1, r1
            cs, ce = c0 + 1, c1
            if rs >= re or cs >= ce:
                continue
            count = int(np.sum(grid[rs:re, cs:ce] == RED))
            for layer in range(count):
                a, b = rs + layer, re - 1 - layer
                x, y = cs + layer, ce - 1 - layer
                if a > b or x > y:
                    break
                out[a, x : y + 1] = BLUE
                out[b, x : y + 1] = BLUE
                out[a : b + 1, x] = BLUE
                out[a : b + 1, y] = BLUE
    return out


def generate_input() -> np.ndarray:
    row_heights = [random.randint(3, 5) for _ in range(random.randint(2, 4))]
    col_widths = [random.randint(3, 5) for _ in range(random.randint(2, 4))]
    H = sum(row_heights) + len(row_heights) - 1
    W = sum(col_widths) + len(col_widths) - 1
    grid = np.zeros((H, W), dtype=int)
    r = 0
    for h in row_heights[:-1]:
        r += h
        grid[r, :] = GREY
        r += 1
    c = 0
    for w in col_widths[:-1]:
        c += w
        grid[:, c] = GREY
        c += 1
    r0 = 0
    for h in row_heights:
        c0 = 0
        for w in col_widths:
            n = random.randint(0, 4)
            cells = [(rr, cc) for rr in range(r0, r0 + h) for cc in range(c0, c0 + w)]
            for rr, cc in random.sample(cells, min(n, len(cells))):
                grid[rr, cc] = RED
            c0 += w + 1
        r0 += h + 1
    return grid
