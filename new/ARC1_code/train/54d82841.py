import numpy as np

import random
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components, paste, random_free_location, random_sprite

# concepts:
# gravity, falling

# description:
# In the input you will see various monochromatic objects
# To make the output, make each object drop a single yellow pixel below it, centered with the middle of the object

def main(input_grid):
    """
    Drop a yellow marker to the bottom row under each object's horizontal center.

    Input structure:
    - The grid contains separated monochromatic objects on a black background.
    - Each object has a bounding box with a well-defined center column.
    - The bottom row is available for the dropped markers.

    Rule:
    - Detect each foreground object.
    - Compute the center column of its bounding box.
    - Paint a yellow pixel in the last row at that center column.

    Key insight:
    - Only the object's horizontal center matters; the marker falls all the way to the
      canvas bottom rather than stopping at other objects.
    """
    # Plan:
    # 1. Detect the objects
    # 2. Drop yellow pixels which land in the final row of the grid, centered with the middle of the object

    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=4, monochromatic=True, return_format="objects")

    output_grid = input_grid.copy()

    for obj in objects:
        b = bbox(obj != bg)
        if b is None:
            continue
        _r, c = bbox_anchor(b, anchor="center")
        bottom_r = output_grid.shape[0] - 1
        output_grid[bottom_r, c] = Color.YELLOW
    
    return output_grid


def generate_input():
    width, height = np.random.randint(15, 30), np.random.randint(15, 30)
    grid = np.full((width, height), Color.BLACK)
    
    n_objects = np.random.randint(1, 5)
    for _ in range(n_objects):
        # Make a random sprite with odd width, so that there will be a unique horizontally center pixel 
        widths = [3, 5, 7]
        heights = [1,2,3,4]
        sprite = random_sprite(widths, heights, color_palette=[random.choice(Color.NOT_BLACK)])
        # Find a place for it, but put some padding along the borders so that it's not at the bottom
        try:
            r, c = random_free_location(grid, sprite, transparent=Color.BLACK, border=2, padding=2, padding_connectivity=8)
        except ValueError:
            break
        # Put it down
        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
