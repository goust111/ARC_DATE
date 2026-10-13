import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# borders

# description:
# In the input you will see an empty black grid.
# To make the output, draw a line along the border of the input with a thickness of one pixel. The border should be teal.

def main(input_grid):
    """
    Draw a one-pixel teal border around an otherwise black input canvas.

    Input structure:
    - The input is a rectangular black grid.
    - No object content needs to be preserved.
    - The output has the same dimensions as the input.

    Rule:
    - Create a black output grid with the input shape.
    - Color the first and last rows teal.
    - Color the first and last columns teal.

    Key insight:
    - The transformation depends only on the canvas dimensions.
    """
    # make the output grid
    H, W = input_grid.shape
    output_grid = np.full((H, W), Color.BLACK, dtype=int)

    # draw the border of the input grid
    output_grid[0, :] = Color.TEAL
    output_grid[-1, :] = Color.TEAL
    output_grid[:, 0] = Color.TEAL
    output_grid[:, -1] = Color.TEAL

    return output_grid
    

def generate_input():
    # make a rectangular black grid
    n = np.random.randint(3, 8)
    m = np.random.randint(3, 8)
    grid = np.zeros((n, m), dtype=int)

    return grid
# ============= remove below this point for prompting =============
