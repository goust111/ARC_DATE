import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# colors

# description:
# To create the output grid, swap the teal and grey colors in the grid.

def main(input_grid):
    """
    Swap every teal pixel with gray and every gray pixel with teal.

    Input structure:
    - The input is an arbitrary square grid of non-black colors.
    - Teal and gray may appear anywhere in the grid.
    - Other colors are not part of the swap.

    Rule:
    - Copy the input grid.
    - Recolor original gray cells to teal.
    - Recolor original teal cells to gray while leaving all other cells unchanged.

    Key insight:
    - The transformation is a pure color substitution, so positions and all non-target
      colors are preserved exactly.
    """
    output_grid = input_grid.copy()
    output_grid[input_grid == Color.GREY] = Color.TEAL
    output_grid[input_grid == Color.TEAL] = Color.GREY
    return output_grid


def generate_input():
    # make a random grid
    n = np.random.randint(3, 10)
    input_grid = np.random.choice(Color.NOT_BLACK, size=(n, n))
    return input_grid
# ============= remove below this point for prompting =============
