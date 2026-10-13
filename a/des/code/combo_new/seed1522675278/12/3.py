import random
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3


def _orientation(cell: np.ndarray):
    pts = np.argwhere(cell == RED)
    if len(pts) != 3:
        return None
    rows = pts[:, 0]
    cols = pts[:, 1]
    vr = rows[np.bincount(rows).argmax()]
    hc = cols[np.bincount(cols).argmax()]
    return ("d" if vr == rows.max() else "u") + ("r" if hc == cols.max() else "l")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor blue separators green when adjacent L-shapes face one another across that separator.

    Input structure:
    - Blue full rows and columns form an equal-cell grid on a black background.
    - Some cells contain a red 3-pixel L-shape with one of four corner orientations.

    Rule:
    - For horizontal neighbors, recolor their shared vertical separator green when the
      left L points to the right and the right L points to the left.
    - For vertical neighbors, recolor their shared horizontal separator green when the
      upper L points down and the lower L points up.
    - Leave all other separators and all L-shapes unchanged.

    Key insight:
    - Separator color depends on a relational test between neighboring cell orientations,
      not on occupancy alone.
    """
    out = grid.copy()
    blue_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == BLUE)]
    blue_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLUE)]
    cell_rs = [-1] + blue_rows + [grid.shape[0]]
    cell_cs = [-1] + blue_cols + [grid.shape[1]]
    cells = {}
    for i, (ra, rb) in enumerate(zip(cell_rs, cell_rs[1:])):
        for j, (ca, cb) in enumerate(zip(cell_cs, cell_cs[1:])):
            cell = grid[ra + 1 : rb, ca + 1 : cb]
            cells[i, j] = _orientation(cell)
    for i in range(len(cell_rs) - 1):
        for j in range(len(cell_cs) - 2):
            a, b = cells[i, j], cells[i, j + 1]
            if a and b and a[1] == "r" and b[1] == "l":
                c = cell_cs[j + 1]
                out[cell_rs[i] + 1 : cell_rs[i + 1], c] = GREEN
    for i in range(len(cell_rs) - 2):
        for j in range(len(cell_cs) - 1):
            a, b = cells[i, j], cells[i + 1, j]
            if a and b and a[0] == "d" and b[0] == "u":
                r = cell_rs[i + 1]
                out[r, cell_cs[j] + 1 : cell_cs[j + 1]] = GREEN
    return out


L_SHAPES = {
    "ul": np.array([[1, 1], [1, 0]], dtype=int),
    "ur": np.array([[1, 1], [0, 1]], dtype=int),
    "dl": np.array([[1, 0], [1, 1]], dtype=int),
    "dr": np.array([[0, 1], [1, 1]], dtype=int),
}


def generate_input() -> np.ndarray:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    cell = 3
    H = rows * cell + rows - 1
    W = cols * cell + cols - 1
    grid = np.zeros((H, W), dtype=int)
    for r in range(cell, H, cell + 1):
        grid[r, :] = BLUE
    for c in range(cell, W, cell + 1):
        grid[:, c] = BLUE
    for i in range(rows):
        for j in range(cols):
            if random.random() < 0.25:
                continue
            orient = random.choice(list(L_SHAPES))
            shape = L_SHAPES[orient]
            r = i * (cell + 1)
            c = j * (cell + 1)
            grid[r : r + 2, c : c + 2][shape == 1] = RED
    return grid
