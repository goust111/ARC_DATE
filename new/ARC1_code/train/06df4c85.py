
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, flood_fill

# concepts:
# rectangular cells, flood fill, connecting same color

# description:
# In the input you will see horizontal and vertical bars that divide the grid into rectangular cells
# To make the output, find any pair of rectangular cells that are in the same row and column and have the same color, then color all the rectangular cells between them with that color

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Connect matching colored cells across the divider grid by filling the black cells between them.

    Input structure:
    - The input is partitioned by full horizontal and vertical bars of one divider color.
    - Rectangular cells are mostly black, with some cells filled by non-divider colors.
    - Matching colored cells may appear in the same row or column of the partition.

    Rule:
    - Identify the divider color from full rows or columns.
    - For each non-black, non-divider colored cell, search rightward and downward for the next same color.
    - Fill intervening black cells along that row or column with the matching color.

    Key insight:
    - Same-colored cells act like endpoints on the partition lattice; only the black gaps between
      aligned matching endpoints are recolored.
    """

    # find the color of the horizontal and vertical bars that divide the rectangular cells
    # this is the color of any line that extends all the way horizontally or vertically
    jail_color = None
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            color = input_grid[i][j]
            if np.all(input_grid[i, :] == color) or np.all(input_grid[:, j] == color):
                jail_color = color
                break
    
    assert jail_color is not None, "No jail color found"

    output_grid = input_grid.copy()

    # color all the cells between the same color pixels
    for r in range(input_grid.shape[0]):
        for c in range(input_grid.shape[1]):
            color = input_grid[r][c]
            if color == jail_color or color == Color.BLACK:
                continue

            # check if there is a cell with the same color in the same X value
            for y2 in range(c+1, input_grid.shape[1]):
                if input_grid[r][y2] == color:
                    for y3 in range(c+1, y2):
                        if input_grid[r][y3] == Color.BLACK:
                            output_grid[r][y3] = color
                    break

            # check if there is a cell with the same color in the same Y value
            for x2 in range(r+1, input_grid.shape[0]):
                if input_grid[x2][c] == color:
                    for x3 in range(r+1, x2):
                        if input_grid[x3][c] == Color.BLACK:
                            output_grid[x3][c] = color
                    break
                
    return output_grid

def generate_input() -> np.ndarray:

    grid_size = 32
    cell_size = 2

    # First create the array of rectangular cells, each of which is separated by horizontal and vertical bars dividing cells

    # pick a non-black color for the divider
    divider_color = random.choice(Color.NOT_BLACK)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    r_offset_r, r_offset_c = np.random.randint(0, cell_size), np.random.randint(0, cell_size)

    # make horizontal bars with cell_size gaps, but +1 because we need to include the divider, which is one pixel wide
    for r in range(r_offset_r, grid_size, cell_size+1):
        grid[r, :] = divider_color
    # make vertical bars with cell_size gaps
    for cc in range(r_offset_c, grid_size, cell_size+1):
        grid[:, cc] = divider_color

    # Second we will color some of the cells with a random color

    # random number of cells to color
    number_to_color = np.random.randint(1, 4)
    for _ in range(number_to_color):
        # pick what we're going to color the inside of the cell, which needs to be a different color from the divider
        other_color = np.random.choice([c for c in Color.ALL_COLORS if c != divider_color and c != Color.BLACK])

        # get all coords of black cells
        black_coords = np.argwhere(grid == Color.BLACK)
        # pick a random black cell
        r, cc = random.choice(black_coords)
        flood_fill(grid, r, cc, other_color)

        # sometimes skip coloring the other side of the divider
        if random.random() <= 0.2:
            continue 

        # flip a coin to decide if horizontal or vertical
        h_or_v = random.random() < 0.5
        if h_or_v:
            # horizontal
            # get all the black cells in the same row
            black_coords = np.argwhere(grid[r, :] == Color.BLACK)
            # pick a random black cell
            other_c = random.choice(black_coords)
            flood_fill(grid, r, other_c, other_color)
        else:
            # vertical
            # get all the black cells in the same column
            black_coords = np.argwhere(grid[:, cc] == Color.BLACK)
            # pick a random black cell
            other_r = random.choice(black_coords)
            flood_fill(grid, other_r, cc, other_color)

    return grid
