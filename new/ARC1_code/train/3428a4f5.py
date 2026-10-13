import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# bitmasks with separator, boolean logical operations

# description:
# Compute the XOR operation of where the two grids are red, turning the output green in those locations.
# In the input, you should see two 6x5 red patterns on top and bottom separated a horizontal yellow line in the middle of the grid.
# To make the output, you have to overlap the two patterns. If the overlapping cells are the same color, then the corresponding cell is colored black; otherwise, 
# if the overlapping cells are not the same color, then the corresponding cell is colored green

def main(input_grid):
    """
    Compute the XOR of the red patterns above and below the yellow separator row.

    Input structure:
    - A full yellow horizontal row separates two equal-sized red and black masks.
    - The top and bottom masks have the same width and height.
    - Red and black encode active and inactive mask cells.

    Rule:
    - Locate the yellow separator row.
    - Split the input into top and bottom masks.
    - Output green where the two masks differ and black where they match.

    Key insight:
    - The separator only aligns the two masks; the output is their pixelwise exclusive-or.
    """

    H, W = input_grid.shape

    # Find the yellow horizontal line/bar (a full row of YELLOW).
    c_bar = None
    for r in range(H):
        if np.all(input_grid[r, :] == Color.YELLOW):
            c_bar = r
            break
    assert c_bar is not None
    
    # Extract top and bottom patterns.
    top_pattern = input_grid[:c_bar, :]
    bottom_pattern = input_grid[c_bar + 1 :, :] 

    output_grid = np.zeros_like(top_pattern)

    # applying the XOR pattern, which is where they are different
    output_grid[(top_pattern != bottom_pattern)] = Color.GREEN
    output_grid[(top_pattern == bottom_pattern)] = Color.BLACK

    return output_grid


def generate_input():
  
    # Define the grid size
    width = 5
    height = 13  # 6 top + 1 yellow line + 6 bottom

    # Initialize an empty grid
    input_grid = np.full((height, width), Color.BLACK, dtype=int)

    # Randomly assign red or black to the top and bottom patterns
    for r in range(height):
        for c in range(width):
            input_grid[r, c] = int(np.random.choice([Color.BLACK, Color.RED]))

    # Set the yellow horizontal separator row
    input_grid[height // 2, :] = Color.YELLOW

    return input_grid
# ============= remove below this point for prompting =============
