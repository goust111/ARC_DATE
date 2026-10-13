import random
import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder sparse columns so the bottom color sequence becomes the left-to-right column order.

    Input structure:
    - The main area is a square sparse matrix with exactly one colored cell per row and per column.
    - The rightmost extra column repeats each row's main-area color as a marker.
    - The bottom extra row lists the desired column colors from left to right.

    Rule:
    - Read the target color order from the bottom instruction row.
    - Identify which current column contains each target color.
    - Reorder whole columns of the main area to match that color order, preserving each
      column's row positions.
    - Remove the marker column and instruction row in the output.

    Key insight:
    - The bottom colors are not data cells; they are a permutation instruction over
      columns whose identities are determined by their unique colors.
    """
    main = grid[:-1, :-1]
    target = list(grid[-1, :-1])
    color_to_col = {}
    for c in range(main.shape[1]):
        rows = np.where(main[:, c] != BG)[0]
        if len(rows) == 1:
            color_to_col[int(main[rows[0], c])] = c
    out = np.zeros_like(main)
    for new_c, color in enumerate(target):
        src_c = color_to_col[int(color)]
        out[:, new_c] = main[:, src_c]
    return out


def generate_input() -> np.ndarray:
    n = random.randint(3, 6)
    colors = random.sample(range(1, 10), n)
    perm_rows = random.sample(range(n), n)
    perm_cols = random.sample(range(n), n)
    target = random.sample(colors, n)
    grid = np.zeros((n + 1, n + 1), dtype=int)
    for color, r, c in zip(colors, perm_rows, perm_cols):
        grid[r, c] = color
        grid[r, -1] = color
    grid[-1, :-1] = target
    return grid
