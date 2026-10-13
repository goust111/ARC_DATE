import numpy as np

import random
from arc_utils import Color, bbox, bbox_h, bbox_w, detect_objects, draw_line

# concepts:
# object detection, color change

# description:
# In the input you will see a grid with a red pattern
# To make the output grid, you should find out any single isolated red objects with size of 1x1 and change them to blue.

def main(input_grid):
    """
    Recolor isolated 1x1 red components blue while leaving larger red components alone.

    Input structure:
    - The grid contains red components on a black background.
    - Components are detected with 4-connectivity.
    - Some red components may be single isolated pixels.

    Rule:
    - Detect all red components.
    - For each component with a 1x1 bounding box, recolor that pixel blue.
    - Leave all other pixels unchanged.

    Key insight:
    - Object size, not location or surrounding shape, determines which red pixels change.
    """
    # Detect all the red objects in the grid, ignoring objects of other colors
    red_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4)

    # Convert 1x1 objects (isolated pixels) into blue
    output_grid = input_grid.copy()
    for obj in red_objects:
        b = bbox(obj != Color.BLACK)
        if b is None:
            continue
        r0, c0, _r1, _c1 = b
        length = bbox_h(b)
        width = bbox_w(b)
        # Find out the single isolated red object with size of 1x1 and change it to blue.
        if length == 1 and width == 1:
            output_grid[r0, c0] = Color.BLUE

    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    n, m = np.random.randint(3, 6), np.random.randint(3, 6)
    grid = np.zeros((m, n), dtype=int)

    colored = 0
    # Randomly scatter density of red pixels on the grid.
    density = 0.4
    while colored < density * m * n:
        r = np.random.randint(0, m)
        c = np.random.randint(0, n)
        if grid[r, c] == Color.BLACK:
            grid[r, c] = Color.RED
            colored += 1

    # Ensure there is at least one 1x1 single isolated red object in the grid.
    red_objects = detect_objects(grid=grid, colors=[Color.RED], monochromatic=True, connectivity=4)
    if not any(np.sum(object != Color.BLACK) == 1 for object in red_objects):
        return generate_input()
    
    return grid
# ============= remove below this point for prompting =============
