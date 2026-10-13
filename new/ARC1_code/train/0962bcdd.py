
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, draw_line, find_components, first_free_location, paste

# concepts:
# pixel manipulation, growing

# description:
# In the input you will see some number of colored crosses, each of which is 3 pixels tall, 3 pixels wide, and has a single pixel in the center of the cross that is a different color.
# Make the output by growing the cross by 1 pixel north/south/east/west, and growing the center pixel by 2 pixels along each of the 4 diagonals.

def main(input_grid):
    """
    Expand each two-color cross into a larger cross-and-diagonal motif.

    Input structure:
    - The input contains small 3x3 cross objects on a black background.
    - Each cross has one arm color and a different center color.
    - Crosses are independent connected components.

    Rule:
    - Locate the center and bounding box of each cross.
    - Extend the arm color one step farther north, south, east, and west.
    - Extend the center color along all four diagonals within the expanded 5x5 footprint.

    Key insight:
    - The two colors have different growth roles: the arm color grows orthogonally, while the
      center color grows diagonally.
    """

    # extract the 3x3 crosses
    crosses = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="objects")

    output_grid = input_grid.copy()

    for cross in crosses:
        # find the center
        _b = bbox(cross != Color.BLACK)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1
        center_r, center_c = r + w//2, c + h//2

        # extract the relevant colors
        center_color = cross[center_r, center_c]
        cross_color = cross[cross != Color.BLACK][0]

        # grow the cross
        for output_r in range(r-1, r+w+1):
            for output_c in range(c-1, c+h+1):
                # skip if out of bounds
                if output_r < 0 or output_c < 0 or output_r >= input_grid.shape[0] or output_c >= input_grid.shape[1]:
                    continue
                
                # grow the cross north/south/east/west
                if output_r == center_r or output_c == center_c:
                    output_grid[output_r, output_c] = cross_color
                
                # grow the center diagonally
                if (output_r - center_r) == (output_c - center_c) or (output_r - center_r) == (center_c - output_c):
                    output_grid[output_r, output_c] = center_color

    return output_grid



def generate_input():
    input_grid = np.zeros((20, 20), dtype=int)

    # create 2 crosses
    for cross_number in range(2):
        # create a separate sprite
        cross = np.zeros((3, 3), dtype=int)
        cross_color, center_color = random.sample(list(Color.NOT_BLACK), 2)
        cross[1, :] = cross_color
        cross[:, 1] = cross_color
        cross[1, 1] = center_color

        # find a free place on the grid
        _loc = first_free_location(input_grid, cross, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc

        # blit the cross to the canvas
        paste(input_grid, cross, r, c, transparent=Color.BLACK, clip=True)

    return input_grid
