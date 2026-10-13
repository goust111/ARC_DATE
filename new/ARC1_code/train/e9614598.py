import numpy as np
import random
from arc_utils import Color, bbox, bbox_anchor, detect_objects, draw_line, paste


# concepts:
# object detection, pattern drawing

# description:
# In the input you will see two blue pixels
# To make the output grid, you should place a 3x3 green cross pattern between the two blue pixels exactly halfway between them.

def main(input_grid):
    """
    Place a green 3x3 cross centered halfway between two blue pixels.

    Input structure:
    - The grid has a black background with exactly two blue pixels.
    - The blue pixels are aligned horizontally or vertically with room between them.
    - Their midpoint can hold a centered 3x3 pattern.

    Rule:
    - Locate the two blue pixels and compute their midpoint.
    - Build a 3x3 green plus-sign sprite.
    - Paste the sprite centered at the midpoint while preserving the blue pixels.

    Key insight:
    - The two blue pixels define only the center location; the inserted object is always
      the same green cross pattern.
    """
    # Locate the two blue pixels.
    coords = np.argwhere(input_grid == Color.BLUE)
    assert coords.shape[0] == 2
    r1, c1 = map(int, coords[0])
    r2, c2 = map(int, coords[1])
    mid_r, mid_c = int((r1 + r2) / 2), int((c1 + c2) / 2)

    # Generate the 3x3 green cross pattern between the two blue pixels.
    green_cross_sprite = np.array( [[Color.BLACK, Color.GREEN, Color.BLACK],
                                [Color.GREEN, Color.GREEN, Color.GREEN],
                                [Color.BLACK, Color.GREEN, Color.BLACK]])
    output_grid = input_grid.copy()
    # Put the cross centered at the midpoint.
    upper_left_r = mid_r - 1
    upper_left_c = mid_c - 1
    output_grid = paste(output_grid, green_cross_sprite, upper_left_r, upper_left_c, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    grid_len = np.random.randint(10, 15)
    n, m = grid_len, grid_len
    grid = np.zeros((m, n), dtype=int)

    # Randomly choose if the two blue pixels are horizontal or vertical.
    if_horizontal  = np.random.choice([True, False])

    # Randomly place two blue pixels on the grid and ensure a 3x3 green cross pattern can be placed in the middle of them.
    line_interval = random.choice(range(1, 4))
    line_width = 3 + line_interval * 2

    while(line_width + 2 > grid_len):
        line_interval = random.choice(range(1, 4))
        line_width = 3 + line_interval * 2

    # Ensure there is enough space for the 3 x 3 green cross pattern and two blue pixels.
    if if_horizontal:
        r = np.random.randint(0, m - line_width - 1)
        c = np.random.randint(1, n - 1)
        grid[r, c] = Color.BLUE
        grid[r + line_width + 1, c] = Color.BLUE
    else:
        r = np.random.randint(1, m - 1)
        c = np.random.randint(0, n - line_width - 1)
        grid[r, c] = Color.BLUE
        grid[r, c + line_width + 1] = Color.BLUE

    return grid
# ============= remove below this point for prompting =============
