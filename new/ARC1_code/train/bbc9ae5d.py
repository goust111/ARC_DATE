import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# counting, incrementing

# description:
# In the input, you will see a row with partially filled with pixels from the one color from left to right.
# To make the output: 
# 1. Take the input row
# 2. Copy it below the original row with one more colored pixel added to the sequence if there is space
# 3. Repeat until there are half as many rows as there are columns
def main(input_grid):
    """
    Build successive rows by extending a left-to-right color run one cell at a time.

    Input structure:
    - The input is a single row with one non-black color run and black remainder.
    - The row length is even.
    - The run starts at the left and grows rightward.

    Rule:
    - Copy the input row as the first output row.
    - For each following row up to half the row length, add one colored cell to the right if possible.
    - Stack the generated rows into the output grid.

    Key insight:
    - Each row represents the previous count incremented by one, capped by the row width.
    """
    # input is a single row; we grow the run of `color` to the right one pixel per row.
    row = np.array(input_grid[0], dtype=int).copy()
    L = int(row.shape[0])
    color = int(next(v for v in row.tolist() if v != Color.BLACK))

    out_rows = [row.copy()]
    for _ in range(L // 2 - 1):
        rightmost = int(np.where(row == color)[0][-1])
        if rightmost < L - 1:
            row[rightmost + 1] = color
        out_rows.append(row.copy())
    return np.stack(out_rows, axis=0)

def generate_input():
    # decide the length of the row, and make it even
    length = np.random.randint(3, 8) * 2

    # decide the color to partially fill the row with
    color = random.choice(list(Color.NOT_BLACK))

    # make a row with the color partially filled from left to right
    row = np.zeros((length,1), dtype=int)
    num_colored_pixels = np.random.randint(1, length//2 + 2)
    row[:num_colored_pixels,:] = color

    # make this row the entire grid
    grid = row

    return grid

# ============= remove below this point for prompting =============
