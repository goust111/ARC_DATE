import numpy as np

import random
from arc_utils import Color, check_between_objects, draw_line, find_components, first_free_location, paste

# concepts:
# connected components

# description:
# In the input you will see two rectangles separated by a gap.
# To make the output, you need to connect the two rectangles with a teal line.

def main(input_grid):
    """
    Connect two separated rectangles by coloring the gap between them teal.

    Input structure:
    - The grid has a black background with two monochromatic rectangle objects.
    - The rectangles are separated by at least a small black gap.
    - Teal is reserved for the connecting bridge.

    Rule:
    - Detect the two non-black connected components.
    - Scan black cells and test which ones lie between the two rectangles.
    - Recolor those between-object cells teal.

    Key insight:
    - The bridge is not drawn from a fixed endpoint; it is the set of background cells
      geometrically between the two object footprints.
    """
    # Copy the input grid as output
    output_grid = input_grid.copy()

    # Detect the objects
    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=4, monochromatic=True, return_format="objects")

    for r, c in np.argwhere(input_grid == bg):
        # Check if the current position is between the two objects
        # Also ensure it is not between the borders of the objects (padding=1)
        if check_between_objects(obj1=objects[0], obj2=objects[1], r=int(r), c=int(c), padding=1, background=bg):
            output_grid[int(r), int(c)] = Color.TEAL
    
    return output_grid


def generate_input():
    # Generate the background grid
    n, m = np.random.randint(10, 20, size=2)
    grid = np.full((m, n), Color.BLACK)

    # Choose two colors for the pattern
    color_connect = Color.TEAL
    colors = np.random.choice([color for color in Color.NOT_BLACK if color != color_connect], 2, replace=False)
    color1, color2 = colors

    # Generate two rectangles
    n1, m1 = np.random.randint(3, 10, size=2)
    n2, m2 = np.random.randint(3, 10, size=2)

    rectangle1 = np.full((n1, m1), color1)
    rectangle2 = np.full((n2, m2), color2)

    # Place the rectangles on the grid
    _loc = first_free_location(grid, rectangle1, transparent=Color.BLACK, border=0, padding=2, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    x1, y1 = _loc
    paste(grid, rectangle1, x1, y1, transparent=Color.BLACK, clip=True)
    try:
        # Check if there is enough space for the second rectangle
        _loc = first_free_location(grid, rectangle2, transparent=Color.BLACK, border=0, padding=2, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        x2, y2 = _loc
        paste(grid, rectangle2, x2, y2, transparent=Color.BLACK, clip=True)
        # Ensure the two rectangles can be connected
        main(grid)
    except Exception as e:
        # If not, regenerate the input
        return generate_input()

    return grid
# ============= remove below this point for prompting =============
