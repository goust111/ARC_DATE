import numpy as np

import random
from arc_utils import Color, bbox, bbox_anchor, detect_objects, draw_line, paste

# concepts:
# constant pattern, diagonal corners

# description:
# In the input you will see one red pixel
# To make the output grid, you should 
# 1. draw a pattern with four different colors centered at the red pixel at its diagonal corners:
#    green in the upper left, pink in the upper right, teal in the lower left, and yellow in the lower right.
# 2. remove the red pixel (equivalently start with a blank canvas and draw the pattern at the red pixel location)

def main(input_grid):
    """
    Replace the red anchor pixel with a fixed four-corner color pattern.

    Input structure:
    - The input is black except for one red pixel.
    - The red pixel marks the center of the output pattern.
    - The output keeps the input shape.

    Rule:
    - Locate the red pixel.
    - Create a blank black output canvas.
    - Paste a 3x3 pattern with colored diagonal corners centered on the red pixel.

    Key insight:
    - The red pixel is only an anchor and is not copied into the output.
    """
    # Find the red pixel position.
    coords = np.argwhere(input_grid == Color.RED)
    assert coords.shape[0] == 1
    red_r, red_c = map(int, coords[0])

    # Construct the specific pattern that is going to be drawn where the red pixel was
    pattern = np.array(
        [
            [Color.GREEN, Color.BLACK, Color.PINK],
            [Color.BLACK, Color.BLACK, Color.BLACK],
            [Color.TEAL, Color.BLACK, Color.ORANGE],
        ]
    )
    
    # Because sprites are anchored by the upper left corner, we are going to need to calculate where the pattern's upper left corner should be
    pattern_r0, pattern_c0 = red_r - 1, red_c - 1

    # The output grid is the same size of input grid
    # start with a blank canvas and then lit the pattern
    output_grid = np.full(input_grid.shape, Color.BLACK)
    output_grid = paste(output_grid, pattern, pattern_r0, pattern_c0, clip=True)

    return output_grid
    
def generate_input():
    # Generate the background grid with size of n x m.
    n, m = 5, 3
    grid = np.zeros((m, n), dtype=int)

    # Randomly select the position of the red pixel and draw it.
    r, c = np.random.randint(0, m - 1), np.random.randint(0, n - 1)
    grid[r, c] = Color.RED
    return grid
# ============= remove below this point for prompting =============
