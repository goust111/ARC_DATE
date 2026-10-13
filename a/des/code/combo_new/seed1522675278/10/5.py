import random
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3


def extract_shapes(grid: np.ndarray):
    sep = next(r for r in range(grid.shape[0]) if np.all(grid[r] == BG))
    top = grid[:sep]
    blue_cols = [c for c in range(grid.shape[1]) if np.all(top[:, c] == BLUE)]
    edges = [-1] + blue_cols + [grid.shape[1]]
    shapes = []
    for a, b in zip(edges, edges[1:]):
        c0 = a + 1
        c1 = b
        if c1 - c0 <= 0:
            continue
        cell = top[:, c0:c1]
        pts = np.argwhere(cell == RED)
        r0, c0b = pts.min(axis=0)
        r1, c1b = pts.max(axis=0)
        mask = (cell[r0 : r1 + 1, c0b : c1b + 1] == RED).astype(int)
        shapes.append(mask)
    return sep, shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each data-row green-count as an index into a shape alphabet.

    Input structure:
    - A top alphabet band contains red shapes separated by blue vertical bars.
    - Below one empty black row, each data row contains scattered green marker pixels.

    Rule:
    - Count the green markers in each data row to obtain k.
    - Take the k-th alphabet shape from left to right.
    - Erase the row's green markers and redraw that shape in green with its top-left
      corner aligned to the row's leftmost former green marker.
    - Keep the alphabet band unchanged and crop only by normal grid boundaries.

    Key insight:
    - Marker position is mostly noise; the meaningful signal is the count, which selects
      a shape, while the leftmost marker only provides the placement anchor.
    """
    sep, shapes = extract_shapes(grid)
    out = grid.copy()
    for r in range(sep + 1, grid.shape[0]):
        cols = np.where(grid[r] == GREEN)[0]
        if len(cols) == 0:
            continue
        shape = shapes[len(cols) - 1]
        out[r, cols] = BG
        c0 = int(cols.min())
        h, w = shape.shape
        rr = min(grid.shape[0], r + h)
        cc = min(grid.shape[1], c0 + w)
        part = shape[: rr - r, : cc - c0]
        block = out[r:rr, c0:cc]
        block[part == 1] = GREEN
    return out


SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
]


def generate_input() -> np.ndarray:
    n = random.randint(3, 5)
    shapes = random.sample(SHAPES, n)
    cell_h = 5
    widths = [max(5, s.shape[1] + 2) for s in shapes]
    top_w = sum(widths) + (n - 1)
    rows = random.randint(3, 6)
    H = cell_h + 1 + rows
    W = max(top_w, 18)
    grid = np.zeros((H, W), dtype=int)
    c = 0
    for i, shape in enumerate(shapes):
        w = widths[i]
        if i > 0:
            grid[:cell_h, c] = BLUE
            c += 1
        sr = (cell_h - shape.shape[0]) // 2
        sc = c + (w - shape.shape[1]) // 2
        grid[sr : sr + shape.shape[0], sc : sc + shape.shape[1]][shape == 1] = RED
        c += w
    for r in range(cell_h + 1, H):
        k = random.randint(1, n)
        cols = random.sample(range(W - 5), k)
        for col in cols:
            grid[r, col] = GREEN
    return grid
