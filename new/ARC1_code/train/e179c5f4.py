import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# bouncing

# description:
# In the input you will see a single blue pixel on a black background
# To make the output, shoot the blue pixel diagonally up and to the right, having it reflect and bounce off the walls until it exits at the top of the grid. Finally, change the background color to teal.

def main(input_grid):
    """
    Trace a bouncing blue diagonal path upward and recolor the black background teal.

    Input structure:
    - The grid has a black background with exactly one blue pixel.
    - The blue pixel is the start of an upward-right diagonal ray.
    - The left and right grid edges act as reflection walls.

    Rule:
    - Starting from the blue pixel, draw blue one step up and horizontally each move.
    - Reverse horizontal direction when the next step would leave the grid.
    - Stop when the path exits through the top, then change remaining black cells to teal.

    Key insight:
    - The path behaves like a diagonal ray reflecting only off vertical walls before the
      final background recoloring.
    """
    # Plan:
    # 1. Detect the pixel
    # 2. Shoot each line of the reflection one-by-one, bouncing (changing horizontal direction) when it hits a (horizontal) wall/edge of canvas
    # 3. Change the background color to teal

    # Locate the blue pixel.
    coords = np.argwhere(input_grid == Color.BLUE)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])

    H, W = input_grid.shape

    # Shoot diagonally up-right, bouncing off left/right walls until exiting the top.
    dr, dc = -1, 1
    while 0 <= r < H and 0 <= c < W:
        input_grid[r, c] = Color.BLUE
        nr = r + dr
        nc = c + dc
        if nr < 0:
            break
        if nc < 0 or nc >= W:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
    
    old_background = Color.BLACK
    new_background = Color.TEAL
    input_grid[input_grid == old_background] = new_background
    
    return input_grid


def generate_input():
    width, height = np.random.randint(2, 15), np.random.randint(10, 30)
    grid = np.full((width, height), Color.BLACK)
    grid[0,-1] = Color.BLUE

    return grid
# ============= remove below this point for prompting =============
