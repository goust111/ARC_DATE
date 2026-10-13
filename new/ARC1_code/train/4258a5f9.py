import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# surrounding

# description:
# surround every gray pixel with blue pixels

def main(input_grid):
    """
    Surround every gray pixel with a blue one-cell neighborhood while keeping the gray centers.

    Input structure:
    - The grid is mostly black.
    - Gray pixels are sparse markers to expand.
    - Markers may lie near the grid edge.

    Rule:
    - For each gray pixel, paint its clipped 3 by 3 neighborhood blue.
    - Combine overlapping neighborhoods normally.
    - Restore all original gray marker cells after painting blue.

    Key insight:
    - Each gray pixel seeds a local halo, while the marker itself remains gray in the output.
    """
    output_grid = np.zeros_like(input_grid)

    for i in range(len(input_grid)):
        for j in range(len(input_grid[i])):
            if input_grid[i, j] == Color.GRAY:
                # if the current pixel is gray, then we need to surround it with blue
                output_grid[max(0, i-1):min(len(input_grid), i+2), max(0, j-1):min(len(input_grid[i]), j+2)] = Color.BLUE

    # but we need to keep the gray center: so copy over all the gray pixels
    output_grid[input_grid == Color.GRAY] = Color.GRAY
            
    return output_grid


# create a 9x9 grid of black (0) and then sparsely populate it with gray
def generate_input():
    # create a 9x9 grid of black (0)
    grid = np.zeros((9, 9), dtype=int)
    # sparsely populate it with gray
    for r in range(9):
        for c in range(9):
            if np.random.random() < 0.05:
                grid[r, c] = Color.GRAY
    return grid
# ============= remove below this point for prompting =============
