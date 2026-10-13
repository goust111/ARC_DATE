import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _separator_lines(grid):
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == MAROON)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    return rows, cols


def _intervals(lines, limit):
    bounds = [-1] + lines + [limit]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1) if bounds[i] + 1 <= bounds[i + 1] - 1]


def main(grid: np.ndarray) -> np.ndarray:
    """Flip objects in table cells according to each cell background color.

    Input structure: Maroon separator rows and columns partition the grid into
    rectangular cells.  The top-left cell is a black/grey quadrant legend; every
    other cell has a black or grey background and one bright asymmetric object.
    Rule: Decode the legend as black background means unchanged and grey
    background means horizontal reflection.  For every non-legend cell, preserve
    separators and background; black-background objects stay in place, while
    grey-background objects are reflected across that cell's vertical middle.
    Key insight: The special top-left region tells how background color controls
    geometry; the object color itself does not choose the transform.
    """
    out = grid.copy()
    sep_rows, sep_cols = _separator_lines(grid)
    row_spans = _intervals(sep_rows, grid.shape[0])
    col_spans = _intervals(sep_cols, grid.shape[1])
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if ri == 0 and ci == 0:
                continue
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, counts = np.unique(cell, return_counts=True)
            bg = int(vals[np.argmax(counts)])
            obj = [(r, c, int(grid[r, c])) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(grid[r, c]) != bg]
            if bg != GREY:
                continue
            for r, c, _ in obj:
                out[r, c] = bg
            for r, c, color in obj:
                out[r, c0 + (c1 - c)] = color
    return out


def _stamp_obj(grid, r0, c0, color):
    pts = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    for r, c in pts:
        grid[r0 + r, c0 + c] = color


def generate_input() -> np.ndarray:
    rows, cols = random.randint(3, 4), random.randint(3, 4)
    ch, cw = random.randint(5, 6), random.randint(5, 6)
    h, w = rows * ch + rows + 1, cols * cw + cols + 1
    grid = np.full((h, w), MAROON, dtype=int)
    colors = [1, 2, 3, 4, 6, 7, 8]
    for ri in range(rows):
        for ci in range(cols):
            r0 = 1 + ri * (ch + 1)
            c0 = 1 + ci * (cw + 1)
            if ri == 0 and ci == 0:
                grid[r0 : r0 + ch, c0 : c0 + cw] = BLACK
                grid[r0 : r0 + ch // 2, c0 + cw // 2 : c0 + cw] = GREY
                grid[r0 + ch // 2 : r0 + ch, c0 : c0 + cw // 2] = GREY
                continue
            bg = random.choice([BLACK, GREY])
            grid[r0 : r0 + ch, c0 : c0 + cw] = bg
            _stamp_obj(grid, r0 + 1, c0 + 1, random.choice(colors))
    return grid
