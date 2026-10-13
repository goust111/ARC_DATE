import numpy as np

import random
from arc_utils import Color, draw_line, paste, random_sprite

# concepts:
# color guide, filling, objects

# description:
# In the input, you will see a colored object in the middle and a single pixel in the bottom left corner of a different color.
# To make the output, remove the pixel from bottom left corner and color the object in the middle with the color from the pixel you removed.

def main(input_grid):
    """
    Use the bottom-left guide pixel to recolor the central object.

    Input structure:
    - A colored object appears away from the bottom-left corner.
    - The bottom-left cell contains a single guide color.
    - Black is the background and the guide color differs from the object color.

    Rule:
    - Read the guide color from the bottom-left cell.
    - Clear that guide pixel to black.
    - Recolor every remaining foreground cell with the guide color.

    Key insight:
    - The corner pixel is a color key that transfers its color and then disappears.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # get the color of the pixel in the bottom-left corner
    color = int(output_grid[-1, 0])

    # remove the pixel from the bottom-left corner
    output_grid[-1, 0] = Color.BLACK

    # color the object in the middle with the color of the pixel from the bottom left corner
    output_grid[output_grid != Color.BLACK] = int(color)
    # could also have used flood_fill:
    # x, y = np.where(output_grid != Color.BLACK)
    # flood_fill(output_grid, x[0], y[0], color)

    return output_grid

def generate_input():
    # make 7x7 black grid with black background
    n = m = 7
    grid = np.zeros((n,m), dtype=int)

    # select a color for the sprite
    sprite_color = np.random.choice(list(Color.NOT_BLACK))

    # select a color for the corner pixel
    corner_color = np.random.choice(list(Color.NOT_BLACK))

    # check that colors are different
    # if they are the same then try again
    if sprite_color == corner_color:
        return generate_input()
    
    # make random sprite and put it in middle of grid
    sprite = random_sprite(m=m - 2, n=n - 2, symmetry="not_symmetric", color_palette=[int(sprite_color)])
    paste(grid, sprite, 1, 1, transparent=Color.BLACK, clip=True)

    # put a single pixel in the bottom left corner
    grid[-1, 0] = int(corner_color)

    return grid
# ============= remove below this point for prompting =============
