import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Turn same-colored seed counts inside square partitions into centered diamond fills.

    Input structure: black grid partitioned by maroon lines into equal odd-sized
    square cells; each cell contains one to three scattered single-pixel seeds
    of one color.
    Rule: for each partition, count its seeds and remember their color, erase
    them, then draw a centered solid Manhattan diamond with radius count minus
    one, never covering the maroon partition lines.
    Key insight: seed positions are distractors; only their color and quantity
    control the centered replacement shape.
    """
    out = grid.copy()
    maroon = 9
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == maroon)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == maroon)]
    for a, b in zip(row_lines, row_lines[1:]):
        for c, d in zip(col_lines, col_lines[1:]):
            cells = [(r, x) for r in range(a + 1, b) for x in range(c + 1, d) if grid[r, x] not in (0, maroon)]
            if not cells:
                continue
            color = int(grid[cells[0]])
            count = len(cells)
            out[a + 1:b, c + 1:d] = 0
            cr, cc = (a + b) // 2, (c + d) // 2
            rad = count - 1
            for r in range(a + 1, b):
                for x in range(c + 1, d):
                    if abs(r - cr) + abs(x - cc) <= rad:
                        out[r, x] = color
    return out


def generate_input() -> np.ndarray:
    nrows, ncols = random.choice([(2, 2), (2, 3), (3, 2)])
    inner = random.choice([5, 7])
    h, w = nrows * inner + nrows + 1, ncols * inner + ncols + 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(0, h, inner + 1):
        grid[r, :] = 9
    for c in range(0, w, inner + 1):
        grid[:, c] = 9
    for i in range(nrows):
        for j in range(ncols):
            r1, c1 = 1 + i * (inner + 1), 1 + j * (inner + 1)
            color = random.choice([1, 2, 3, 4, 6, 7, 8])
            count = random.randint(1, 3)
            spots = random.sample([(r, c) for r in range(r1, r1 + inner) for c in range(c1, c1 + inner)], count)
            for r, c in spots:
                grid[r, c] = color
    return grid
