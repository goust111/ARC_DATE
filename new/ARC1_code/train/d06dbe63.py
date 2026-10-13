
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line

# concepts:
# staircase pattern

# description:
# In the input you will see a single teal pixel.
# To make the output, draw a staircase from the teal pixel to the upper right and lower left with a step size of 2.

def main(input_grid):
    """
    Draw mirrored gray staircases of step length two from the teal anchor pixel.

    Input structure:
    - The grid is black except for one teal pixel.
    - The teal pixel is away from the border.
    - The output keeps the input dimensions.

    Rule:
    - From the teal pixel, draw gray steps upward then rightward with length two.
    - Also draw gray steps downward then leftward with length two.
    - Restore the teal pixel after drawing the staircases.

    Key insight:
    - The single teal pixel is the shared center of two opposite staircase arms.
    """
    # Find the location of the teal pixel
    teal_r, teal_c = np.argwhere(input_grid == Color.TEAL)[0]

    # staircase is gray
    staircase_color = Color.GRAY

    # we are going to draw on top of the input
    output_grid = input_grid.copy()
    H, W = input_grid.shape

    # Draw stairs from the teal pixel
    STAIR_LEN = 2
    # Upper-right staircase: go up then right (step size 2).
    r, c = teal_r, teal_c
    while 0 <= r < H and 0 <= c < W:
        # go up
        draw_line(output_grid, r, c, length=STAIR_LEN, color=staircase_color, direction=(-1, 0))
        r -= STAIR_LEN
        # go right
        draw_line(output_grid, r, c, length=STAIR_LEN, color=staircase_color, direction=(0, 1))
        c += STAIR_LEN
    
    # Lower-left staircase: go down then left (step size 2).
    r, c = teal_r, teal_c
    while 0 <= r < H and 0 <= c < W:
        # go down
        draw_line(output_grid, r, c, length=STAIR_LEN, color=staircase_color, direction=(1, 0))
        r += STAIR_LEN
        # go left
        draw_line(output_grid, r, c, length=STAIR_LEN, color=staircase_color, direction=(0, -1))
        c -= STAIR_LEN
    
    # make sure that the teal pixel stays there
    output_grid[teal_r, teal_c] = Color.TEAL

    return output_grid

def generate_input():
    # Generate grid
    width, height = np.random.randint(15, 25), np.random.randint(15, 25)
    grid = np.zeros((width, height), dtype=int)

    # Randomly place one teal pixel on the grid
    # Ensure the pixel is not on the border
    r, c = np.random.randint(width // 3, width * 2 // 3), np.random.randint(height // 3, height * 2 // 3)
    grid[r, c] = Color.TEAL

    return grid
