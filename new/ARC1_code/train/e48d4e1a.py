import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# counting

# description:
# In the input, you will see a 10x10 black grid. The grid has one row and one column colored solidly in a single color, forming a cross of that color across the grid. The grid also has H grey pixels in the rightmost column at the top of the grid.
# To create the output, convert the H grey pixels to be black. Then shift the "cross" H pixels to the left and H pixels down.

def main(input_grid):
    """
    Move a colored cross left and downward by the gray counter value.

    Input structure:
    - The input is a 10x10 black grid with one solid colored row and one solid colored column.
    - The row and column share the same non-black, non-gray color.
    - The top of the rightmost column contains gray pixels that encode the shift amount.

    Rule:
    - Count gray pixels in the rightmost column.
    - Find the colored cross row, column, and color.
    - Recreate the cross on black after moving the row down and the column left by that count.

    Key insight:
    - The gray pixels are a numeric instruction only; they disappear after determining
      how far to shift the cross.
    """
    # 1. Calculate the shift amount: the number of grey pixels in the rightmost column
    H = int(np.sum(input_grid[:, -1] == Color.GREY))

    # 2. find the row and column of the cross, and its color.
    color = next(c for c in np.unique(input_grid) if c not in [Color.BLACK, Color.GREY])

    # find which row is fully colored with the color
    r = int(np.where(np.all(input_grid == color, axis=1))[0][0])
    # find which column is fully colored with the color
    c = int(np.where(np.all(input_grid == color, axis=0))[0][0])

    # shift the column and row down/left by H
    r += H
    c -= H

    # create the output grid by recreating the cross at the new location
    output_grid = np.full(input_grid.shape, Color.BLACK)
    output_grid[r, :] = color
    output_grid[:, c] = color

    return output_grid

def generate_input():
    # create a 10x10 black grid
    grid = np.full((10, 10), Color.BLACK)

    # choose a shift amount H (number of grey pixels in the rightmost column)
    H = int(np.random.randint(1, 5))

    # choose a column/row and color for the cross such that shifting by H stays in-bounds
    r = int(np.random.randint(H, 10 - H))
    # avoid the rightmost column (reserved for the grey counter)
    c = int(np.random.randint(H, 9))
    color = int(np.random.choice([c for c in Color.NOT_BLACK if c != Color.GREY]))
    grid[r, :] = color
    grid[:, c] = color

    # color the rightmost column's top H pixels grey
    grid[:H, -1] = Color.GREY

    return grid
# ============= remove below this point for prompting =============

