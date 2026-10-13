import numpy as np

import random
from arc_utils import Color, bbox, draw_line, find_components, flood_fill

# concepts:
# objects, flood fill, connectivity

# description:
# In the input, you will see a black grid with a red line that starts in the top left corner and bounces off the borders of the grid until it reaches the right side of the grid.
# To make the output, find the black regions separated by the red lines, then, starting with the first region from the left, color every third region yellow.

def main(input_grid):
    """
    Color every third black region between a red bouncing divider line yellow.

    Input structure:
    - The input is a black grid split by a red line.
    - The red line acts as a boundary between connected black regions.
    - Regions are ordered from left to right by their bounding boxes.

    Rule:
    - Find connected regions using red as the separating background.
    - Sort the regions from left to right.
    - Flood-fill regions with indices 0, 3, 6, and so on with yellow.

    Key insight:
    - The red path partitions the grid; the fill pattern is determined by region order.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the black regions in the input grid
    bg = Color.RED
    black_regions = find_components(
        input_grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="objects",
    )

    # sort the regions from left to right using the coordinates of their bounding boxes
    black_regions = sorted(black_regions, key=lambda region: bbox(region != bg)[1])

    # color every third black region yellow using flood fill
    for i, region in enumerate(black_regions):
        if i % 3 == 0:
            rr, cc = np.argwhere(region != bg)[0]
            flood_fill(output_grid, int(rr), int(cc), Color.YELLOW, connectivity=4)

    return output_grid



def generate_input():
    # make a black grid that is 3 cells tall and between 10 and 20 cells wide
    grid = np.zeros((np.random.randint(10,20), 3), dtype=int)

    # make a red line that starts in the top left corner and bounces off the borders of the grid until it reaches the right side of the grid
    r, c = 0, 0
    dr, dc = 1, 1
    # go until we reach the right side of the grid
    while r < grid.shape[0]:
        # make the current cell red
        grid[r, c] = Color.RED
        # if we are at the top or bottom of the grid, bounce off the top or bottom border
        if c + dc >= grid.shape[1]:
            dc = -1
        elif c + dc < 0:
            dc = 1
        r += dr
        c += dc

    return grid
# ============= remove below this point for prompting =============
