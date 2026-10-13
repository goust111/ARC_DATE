import numpy as np

import random
from arc_utils import Color, bbox, bbox_anchor, colors_present, draw_line

# concepts:
# pattern generation

# description:
# In the input you will see a grid with a single pixel on the bottom of the grid.
# To make the output, you should draw a geometric pattern starting outward from the pixel:
# step 1: draw vertical bars starting from the pixel and going to the right with a horizontal period of 2.
# step 2: put a grey pixel in between the vertical bars alternating between the top / bottom.

def main(input_grid):
    """
    Grow a periodic vertical-bar pattern rightward from the lone colored pixel and add alternating gray markers.

    Input structure:
    - The input is a black grid with exactly one non-black pixel.
    - The lone pixel supplies the bar color and starting column.
    - Gray is reserved for the alternating marker pixels.

    Rule:
    - Draw vertical bars upward from the seed row every two columns to the right.
    - Place gray pixels in the intervening columns.
    - Alternate the gray marker row between the top and bottom edge.

    Key insight:
    - The seed pixel determines the color and phase of a periodic pattern across the remaining columns.
    """
    # Output grid is the same size as the input grid
    output_grid = np.zeros_like(input_grid)

    # Detect the single non-black pixel.
    coords = np.argwhere(input_grid != Color.BLACK)
    assert coords.shape[0] == 1
    pixel_r, pixel_c = map(int, coords[0])
    pixel_color = int(input_grid[pixel_r, pixel_c])

    # Get the color of the pattern pixel by observation
    pattern_pixel_color = Color.GRAY
    
    # STEP 1: Draw vertical bar from bottom to top starting from the pixel and going to the right, horizontal period of 2
    horizontal_period = 2
    L = max(output_grid.shape) * 3
    for c in range(pixel_c, output_grid.shape[1], horizontal_period):
        # go up from (pixel_r,c)
        draw_line(output_grid, pixel_r, c, direction=(-1, 0), length=L, color=pixel_color, clip=True)
    
    # STEP 2: put a grey pixel in between the vertical bars alternating between the top / bottom.
    cur_r = output_grid.shape[0] - 1 if pixel_r == 0 else 0
    for c in range(pixel_c + 1, output_grid.shape[1], horizontal_period):
        output_grid[cur_r, c] = pattern_pixel_color
        # alternate between top and bottom
        cur_r = 0 if cur_r == output_grid.shape[0] - 1 else output_grid.shape[0] - 1

    return output_grid

def generate_input():
    # Generate the background grid
    n, m = np.random.randint(10, 20, size=2)
    grid = np.zeros((m, n), dtype=int)

    # Randomly choose the color of the line
    pattern_pixel_color = Color.GRAY
    color = np.random.choice([color for color in Color.NOT_BLACK if color != pattern_pixel_color])

    # Randomly place the pixel on the bottom of the grid
    r = np.random.randint(0, m)
    grid[r, -1] = color

    return grid
# ============= remove below this point for prompting =============
