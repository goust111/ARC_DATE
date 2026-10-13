
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components, first_free_location, paste, randomly_spaced_indices

# concepts:
# downscaling, rectangular regions

# description:
# In the input you will see a grid consisting of monochromatic rectangular regions/partitions (chessboard pattern) of different colors.
# Each rectangular region/partition/chessboard cell is filled with a single color, and has a different size.
# To make the output, make a grid with one pixel for each cell of the chessboard.

def main(input_grid):
    """
    Downscale a colored rectangular partition grid to one pixel per partition.

    Input structure:
    - A multicolored chessboard-like sprite sits on a black background.
    - Each cell is a monochromatic rectangle and cell sizes may vary.
    - Component upper-left corners define the row and column partition positions.

    Rule:
    - Detect the non-black rectangular components.
    - Collect sorted unique top-left row and column positions.
    - Sample the input color at each position to build the compact output grid.

    Key insight:
    - The output is the partition coordinate skeleton, not a proportional resize.
    """
    # Plan:
    # 1. Find the colored regions in the input grid
    # 2. Get the coordinates of the chessboard pattern, which are the x/y positions of the regions
    # 3. Draw the output grid with one pixel for each region (cell of the chessboard)

    # 1. Input parsing
    # Find the colored objects in the input grid
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    # 2. Figuring out the chessboard pattern
    # Get the position of the objects
    positions = []
    for obj in objects:
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        positions.append(bbox_anchor(_b, anchor="upper left"))
    r_position_list = [r for r, _c in positions]
    c_position_list = [c for _r, c in positions]

    # Sort the position list, and get the unique position list since
    # the pattern is separated as chessboard
    r_position_list = sorted(np.unique(r_position_list))
    c_position_list = sorted(np.unique(c_position_list))

    # 3. Drawing the output
    # Get the size of chessboard with one pixel represents the original region
    w_color, h_color = len(r_position_list), len(c_position_list)
    output_grid = np.full((w_color, h_color), Color.BLACK)

    for r_index, r in enumerate(r_position_list):
        for c_index, c in enumerate(c_position_list):
            # Use one pixel to represent the original region
            output_grid[r_index, c_index] = input_grid[r, c]

    return output_grid

def generate_input():
    # Randomly choose the a number of vertical and horizontal partitions, each of which will have a random color
    n_regions_horizontal, n_regions_vertical = np.random.randint(2, 4), np.random.randint(2, 4)
    colors = np.random.choice(Color.NOT_BLACK, (n_regions_horizontal, n_regions_vertical), replace=True)

    # Randomly choose the size of the region of the canvas that we are going to color with a sprite
    sprite_w, sprite_h = np.random.randint(15, 20), np.random.randint(15, 20)

    # Randomly separate into a 2d grid of regions
    horizontal_endpoints = randomly_spaced_indices(max_len=sprite_w, n_indices=n_regions_horizontal - 1, border_size=1, padding=2)
    vertical_endpoints = randomly_spaced_indices(max_len=sprite_h, n_indices=n_regions_vertical - 1, border_size=1, padding=2)


    # Assign the colors to different regions
    sprite = np.zeros((sprite_w, sprite_h), dtype=int)
    for X in range(n_regions_horizontal):
        for Y in range(n_regions_vertical):
            x1 = 0 if X == 0 else horizontal_endpoints[X-1]
            x2 = horizontal_endpoints[X] if X < len(horizontal_endpoints) else sprite_w
            y1 = 0 if Y == 0 else vertical_endpoints[Y-1]
            y2 = vertical_endpoints[Y] if Y < len(vertical_endpoints) else sprite_h

            sprite[x1:x2, y1:y2] = colors[X, Y]
    
    # Randomly place the sprite in the grid
    # The grid size is larger than the sprite size
    n, m = np.random.randint(sprite_h + 1, 30), np.random.randint(sprite_w + 1, 30)
    grid = np.full((m, n), Color.BLACK)
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
