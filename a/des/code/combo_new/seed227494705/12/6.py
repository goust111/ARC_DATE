import random
import numpy as np


CYCLE = [2, 1, 3, 4]


def _mix(row_color, col_color):
    if row_color == col_color:
        return row_color
    step = (CYCLE.index(col_color) - CYCLE.index(row_color)) % len(CYCLE)
    return col_color if step <= 2 else row_color


def _stamp(out, r, c, row_color, col_color):
    if row_color == col_color:
        out[r - 1:r + 2, c - 1:c + 2] = row_color
        return
    edge = _mix(row_color, col_color)
    out[r - 1:r + 2, c - 1:c + 2] = edge
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        out[r + dr, c + dc] = col_color
    out[r, c] = row_color


def main(grid: np.ndarray) -> np.ndarray:
    """Complete all row-header and column-header intersections with 3x3 mixed-color patterns.

    Input structure: black grid with colored column headers on the top row,
    colored row headers on the left column, and some prefilled 3x3 intersection
    examples using red, blue, green, and yellow.
    Rule: for every active row/column intersection, draw a 3x3 pattern centered
    there: center is the row color, corners are the column color, and edge
    middles use the later color in the visible red-blue-green-yellow cycle.
    Key insight: edge midpoint color is derived from a cyclic order relating the
    two headers, not from either axis alone.
    """
    out = grid.copy()
    cols = [(c, int(grid[0, c])) for c in range(1, grid.shape[1] - 1) if grid[0, c] != 0]
    rows = [(r, int(grid[r, 0])) for r in range(1, grid.shape[0] - 1) if grid[r, 0] != 0]
    for r, row_color in rows:
        for c, col_color in cols:
            if 1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1:
                _stamp(out, r, c, row_color, col_color)
    return out


def generate_input() -> np.ndarray:
    h, w = 21, 21
    grid = np.zeros((h, w), dtype=int)
    row_positions = random.sample([3, 7, 11, 15, 18], random.randint(3, 5))
    col_positions = random.sample([3, 7, 11, 15, 18], random.randint(3, 5))
    for r in row_positions:
        grid[r, 0] = random.choice(CYCLE)
    for c in col_positions:
        grid[0, c] = random.choice(CYCLE)
    pairs = [(r, c) for r in row_positions for c in col_positions]
    for r, c in random.sample(pairs, random.randint(1, max(1, len(pairs) // 2))):
        _stamp(grid, r, c, int(grid[r, 0]), int(grid[0, c]))
    return grid
