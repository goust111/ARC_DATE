import numpy as np

import random
from arc_utils import Color, draw_line, find_components, paste

# concepts:
# filling

# description:
# The input consists of a black grid. The grid is divided with red lines into black rectangles of different sizes.
# To produce the output grid, fill in the smallest black rectangles with teal, and fillin in the largest black rectangles with blue.

def main(input_grid):
    """
    Fill smallest red-bounded black rectangles teal and largest ones blue.

    Input structure:
    - Red divider lines partition a black grid into rectangular black regions.
    - The red lines remain as separators.
    - There may be multiple regions tied for smallest or largest area.

    Rule:
    - Treat red as the background and find the black rectangular components.
    - Compute each component's black area.
    - Recolor smallest-area regions teal and largest-area regions blue, leaving red dividers intact.

    Key insight:
    - The output depends only on relative region area within the red partition, not on
      the regions' positions.
    """
    # to get the black rectangles, find connected components with red as background
    objects = find_components(input_grid, bg=Color.RED, connectivity=4, monochromatic=True, return_format="objects")

    # get object areas
    object_areas = [np.sum(obj == Color.BLACK) for obj in objects]

    # find the smallest and largest areas
    smallest_area = min(object_areas)
    largest_area = max(object_areas)

    # fill in the smallest rectangles with teal, and the largest rectangles with blue
    new_objects = []
    for obj in objects:
        area = int(np.sum(obj == Color.BLACK))
        if area == smallest_area:
            obj[obj == Color.BLACK] = Color.TEAL
        elif area == largest_area:
            obj[obj == Color.BLACK] = Color.BLUE
        new_objects.append(obj)

    # create an output grid to store the result
    output_grid = np.full(input_grid.shape, Color.RED)

    # blit the objects back into the grid
    for obj in new_objects:
        output_grid = paste(output_grid, obj, 0, 0, transparent=Color.RED)

    return output_grid


def generate_input():
    # create a grid of size 10-20x10-20
    n = np.random.randint(10, 21)
    m = np.random.randint(10, 21)
    grid = np.full((n, m), Color.BLACK)

    num_lines = np.random.randint(3, 15)

    for i in range(num_lines):
        # add a red line to divide the grid somewhere
        r, c = np.random.randint(2, n-1), np.random.randint(2, m-1)
        # make sure we're not neighboring a red line already
        if Color.RED in [grid[r, c+1], grid[r, c-1], grid[r+1, c], grid[r-1, c]]:
            continue

        horizontal = np.random.choice([True, False])
        if horizontal:
            draw_line(grid, r, c, direction=(1, 0), color=Color.RED, stop_at=[Color.RED])
            draw_line(grid, r-1, c, direction=(-1, 0), color=Color.RED, stop_at=[Color.RED])
        else:
            draw_line(grid, r, c, direction=(0, 1), color=Color.RED, stop_at=[Color.RED])
            draw_line(grid, r, c-1, direction=(0, -1), color=Color.RED, stop_at=[Color.RED])

    return grid
# ============= remove below this point for prompting =============
