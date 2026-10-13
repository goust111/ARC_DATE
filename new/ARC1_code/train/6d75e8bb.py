import numpy as np

import random
from arc_utils import Color, bbox, draw_line, first_free_location, paste, random_sprite

# concepts:
# shape completion

# description:
# In the input you will see an incomplete teal ractangle
# To make the output grid, you should use the red color to complete the rectangle.

def main(input_grid):
    """
    Complete the bounding rectangle around an incomplete teal shape by filling its missing cells red.

    Input structure:
    - The input is a black grid containing an incomplete teal rectangle.
    - The teal pixels define the intended rectangle's bounding box.
    - Black pixels inside that bounding box are missing rectangle cells.

    Rule:
    - Find the bounding box of all non-black pixels.
    - Replace black cells inside that box with red while preserving teal cells.
    - Paste the completed rectangle back into the original grid.

    Key insight:
    - The non-black bounding box gives the full rectangle extent; interior black cells are gaps to complete.
    """
    # Find the bounding box of the incomplete rectangle and use it to extra the sprite
    b = bbox(input_grid != Color.BLACK)
    assert b is not None
    r0, c0, r1, c1 = b
    rectangle = input_grid[r0 : r1 + 1, c0 : c1 + 1]

    # Find the missing parts of the rectangle (which are colored black) and complete it with red color
    rectangle_sprite = np.where(rectangle == Color.BLACK, Color.RED, rectangle)

    # Make the output by copying the sprite to a new canvas
    output_grid = np.copy(input_grid)
    output_grid = paste(output_grid, rectangle_sprite, r0, c0)

    return output_grid

def generate_input():
    # Generate a grid with a size of n x m
    n, m = np.random.randint(7, 15), np.random.randint(7, 15)
    grid = np.zeros((m, n), dtype=int)

    # Randomly generate a rectangle with a size of x_len x y_len, not too big nor too small
    r_len = np.random.randint(m // 2, m - 2)
    c_len = np.random.randint(n // 2, n - 2)

    # Randomly generate a rectangle with a size of x_len x y_len that is incomplete (density < 1)
    rectangle_sprite = random_sprite(m=r_len, n=c_len, color_palette=[Color.TEAL], density=0.3, connectivity=8)

    # Randomly choose a position to draw the rectangle
    _loc = first_free_location(grid, rectangle_sprite, transparent=Color.BLACK, border=1, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    grid = paste(grid, rectangle_sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
