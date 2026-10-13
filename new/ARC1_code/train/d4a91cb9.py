import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# lines, color

# description:
# In the input you will see a red pixel and a teal pixel.
# To make the output, draw a horizontal yellow line from the red pixel to the column of the teal pixel, then draw a vertical yellow line from there to the teal pixel.

def main(input_grid):
    """
    Connect a red pixel to a teal pixel with a yellow right-angle path.

    Input structure:
    - The grid has a black background with exactly one red pixel and one teal pixel.
    - The red and teal pixels are not in the same row or column.
    - The output keeps the original endpoint pixels.

    Rule:
    - Copy the input grid.
    - Draw yellow horizontally from the red pixel's row to the teal pixel's column.
    - Draw yellow vertically along the teal pixel's column until reaching the teal pixel.

    Key insight:
    - The teal pixel supplies the destination column first, so the path is an L shape
      with its corner aligned to the red row and teal column.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the red and teal pixels
    (r_red, c_red) = map(int, np.argwhere(input_grid == Color.RED)[0])
    (r_teal, c_teal) = map(int, np.argwhere(input_grid == Color.TEAL)[0])

    # Horizontal segment: from red row to teal column (excluding red).
    if c_red < c_teal:
        output_grid[r_red, c_red + 1 : c_teal + 1] = Color.YELLOW
    else:
        output_grid[r_red, c_teal:c_red] = Color.YELLOW

    # Vertical segment: from that corner down/up to teal (excluding teal).
    if r_red < r_teal:
        output_grid[r_red + 1 : r_teal, c_teal] = Color.YELLOW
    else:
        output_grid[r_teal + 1 : r_red, c_teal] = Color.YELLOW

    return output_grid


def generate_input():
    # make a black grid as the background
    n = np.random.randint(6, 15)
    m = np.random.randint(6, 15)
    grid = np.zeros((n, m), dtype=int)

    # select a random position for the red pixel
    red_r = np.random.randint(0, n)
    red_c = np.random.randint(0, m)
    grid[red_r, red_c] = Color.RED

    # select a random position for the teal pixel
    teal_r = np.random.randint(0, n)
    teal_c = np.random.randint(0, m)
    grid[teal_r, teal_c] = Color.TEAL

    # check if the red and teal pixels are in the same row or column
    # if they are, then try again
    if red_r == teal_r or red_c == teal_c:
        return generate_input()
    
    return grid

# ============= remove below this point for prompting =============
