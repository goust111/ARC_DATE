
import random

import numpy as np
from typing import *
from arc_utils import Color, draw_line

# concepts:
# geometric pattern, repetition, spiral

# description:
# In the input you will see an empty black grid.
# To make the output, you should draw a spiral pattern of green pixels starting from the top left corner and going to the right.

def main(input_grid):
    """
    Draw a clockwise green spiral into an empty black square grid.

    Input structure:
    - The input grid is entirely black.
    - The grid is square and at least large enough for a visible spiral.
    - The output keeps the same dimensions.

    Rule:
    - Start at the top-left cell and move right.
    - Turn clockwise when the next step would leave the grid or when the cell two steps ahead is already green.
    - Stop when the next step would enter an already green cell.

    Key insight:
    - The one-cell lookahead prevents the path from touching itself too early, creating the spiral gap.
    """
    # get the grid size
    H, W = input_grid.shape

    # start from the top left corner of the grid
    r, c = 0, 0
    output_grid = input_grid.copy()
    output_grid[r, c] = Color.GREEN

    # initial direction: right
    dr, dc = (0, 1)

    def turn_clockwise(dr: int, dc: int) -> tuple[int, int]:
        # rotate (dr,dc) clockwise
        return (dc, -dr)

    # continue spiralling until we cannot anymore
    while True:
        # First, check if we hit the border, if so, we turn clockwise
        if r + dr >= H or c + dc >= W or r + dr < 0 or c + dc < 0:
            dr, dc = turn_clockwise(dr, dc)
            continue

        # Then, check if the square after the current one is green, if so,
        # we are already spiralling, so stop here
        if output_grid[r + dr, c + dc] == Color.GREEN:
            break

        # Last, check if the square after the current one is green, if so, we turn clockwise
        # We do this to draw the spiral pattern
        if (
            0 <= r + 2 * dr < H
            and 0 <= c + 2 * dc < W
            and output_grid[r + 2 * dr, c + 2 * dc] == Color.GREEN
        ):
            dr, dc = turn_clockwise(dr, dc)
            continue
        
        # then we move to the next square and color it green
        r += dr
        c += dc
        output_grid[r, c] = Color.GREEN

    return output_grid

def generate_input():
    # first create a randomly sized grid, somewhere between 5x5 and 20x20
    length = random.randint(5, 20)
    grid = np.full((length, length), Color.BLACK)

    return grid
