import numpy as np

import random
from arc_utils import Color, draw_line


# concepts:
# patterns, horizontal bars

# description:
# In the input you will see a colored pattern in a 3x3 grid.
# For each row of the input, if that row is a single color, color that row in the output grey. Otherwise, output black.

def main(input_grid):
    """
    Mark rows that are a single color with gray and leave all other rows black.

    Input structure:
    - The input is a small colored grid.
    - Rows may be uniform or contain multiple colors.
    - The output has the same dimensions as the input.

    Rule:
    - Inspect each row independently.
    - If every cell in a row matches the row's first cell, fill that output row with gray.
    - Otherwise leave that output row black.

    Key insight:
    - The transformation detects row-level uniformity only; the actual input colors do not matter.
    """
    # get input grid shape
    n, m = input_grid.shape

    # get output grid ready
    output_grid = np.zeros((n, m), dtype=int)

    # look at each row of the input grid
    for r in range(n):
        row = input_grid[r, :]
        if np.all(row == row[0]):
            output_grid[r, :] = Color.GREY

    return output_grid

def generate_input():
    # create a 3x3 array of randomly chosen, non-black, colors
    grid = np.random.choice(Color.NOT_BLACK, size=(3, 3))

    return grid
# ============= remove below this point for prompting =============
