import numpy as np

import random
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components

# concepts:
# bouncing

# description:
# In the input you will see a single blue pixel on a black background
# To make the output, shoot the blue pixel diagonally up and to the right, having it reflect and bounce off the walls until it exits at the top of the grid

def main(input_grid):
    """
    Trace the blue pixel diagonally upward while bouncing off side walls.

    Input structure:
    - The grid is black except for one blue pixel.
    - The blue pixel is the starting point of the path.
    - The grid boundaries act as the reflecting walls.

    Rule:
    - Move diagonally up and right from the start pixel.
    - Paint every visited cell blue.
    - Reverse horizontal direction at the left or right edge and stop when exiting the top.

    Key insight:
    - Only the horizontal direction changes on a bounce; the path always moves upward.
    """
    # Plan:
    # 1. Detect the pixel
    # 2. Shoot each line of the reflection one-by-one, bouncing (changing horizontal direction) when it hits a (horizontal) wall/edge of canvas

    # Locate the blue pixel.
    coords = np.argwhere(input_grid == Color.BLUE)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])

    H, W = input_grid.shape
    out = input_grid.copy()

    # Shoot diagonally up-right, bouncing off left/right walls until exiting the top.
    dr, dc = -1, 1
    while 0 <= r < H and 0 <= c < W:
        out[r, c] = Color.BLUE

        nr = r + dr
        nc = c + dc
        if nr < 0:
            break
        if nc < 0 or nc >= W:
            dc = -dc
            nc = c + dc
        r, c = nr, nc

    return out


def generate_input():
    width, height = np.random.randint(2, 15), np.random.randint(10, 30)
    grid = np.full((width, height), Color.BLACK)
    grid[0,-1] = Color.BLUE

    return grid
# ============= remove below this point for prompting =============
