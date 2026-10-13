
import numpy as np
from typing import *
from arc_utils import Color, detect_objects, draw_line
import random

# concepts:
# bouncing

# description:
# In the input you will see a short diagonal teal line pointing at a red rectangle on a black background.
# To make the output, shoot outward from the teal line, but change the color to green. Reflect off the red rectangle.

def main(input_grid):
    """
    Shoot green rays from both ends of a teal diagonal line and reflect them off the red rectangle.

    Input structure:
    - The grid contains one short teal diagonal line on a black background.
    - A single red rectangle or bar lies in the ray path.
    - The scene may be rotated.

    Rule:
    - Detect the teal line endpoints and the red rectangle.
    - Draw green in both outward diagonal directions from the teal line.
    - When a ray hits red, flip the appropriate direction component and keep drawing.

    Key insight:
    - The teal line gives the initial ray direction, and the red object acts as a reflecting
      obstacle for the green path.
    """
    # Plan:
    # 1. Detect the objects
    # 2. Determine the orientation of the teal line, and its endpoints
    # 3. Shoot a green line outward until it hits the red rectangle
    # 4. Reflect the green line off the red rectangle, continuing in green

    teal_line = detect_objects(input_grid, colors=[Color.TEAL], monochromatic=True, connectivity=8)
    assert len(teal_line) == 1, "There should be exactly one teal line"
    teal_line = list(teal_line)[0]
    red_rectangle = detect_objects(input_grid, colors=[Color.RED], monochromatic=True, connectivity=8)
    assert len(red_rectangle) == 1, "There should be exactly one red rectangle"
    red_rectangle = list(red_rectangle)[0]

    output_grid = input_grid.copy()

    # To get the orientation of a line, find the endpoints and compare their x and y coordinates
    x1, y1 = max( (r, c) for r, c in np.argwhere(teal_line == Color.TEAL) )
    x2, y2 = min( (r, c) for r, c in np.argwhere(teal_line == Color.TEAL) )
    direction12 = (int(np.sign(x2 - x1)), int(np.sign(y2 - y1)))
    direction21 = (-direction12[0], -direction12[1])

    # Try both (direction, x2, y2) and (-direction, x1, y1) as starting points
    for (dr,dc), start_r, start_c in [ (direction12, x2, y2), (direction21, x1, y1) ]:
        start_r += dr
        start_c += dc
        # Loop, shooting lines off of red things, until we run out of the canvas
        while 0 <= start_r < input_grid.shape[0] and 0 <= start_c < input_grid.shape[1]:
            stop_r, stop_c = draw_line(output_grid, start_r, start_c, direction=(dr,dc), color=Color.GREEN, stop_at=[Color.RED])

            # reflection geometry depends on if we hit the red rectangle on our left/right/up/down
            # did we hit the red rectangle on our right? 
            if stop_r+1 < output_grid.shape[0] and output_grid[stop_r+1, stop_c] != Color.BLACK:
                dr = -dr
            # did we hit the red rectangle on our left?
            elif stop_r-1 >= 0 and output_grid[stop_r-1, stop_c] != Color.BLACK:
                dr = -dr
            # did we hit the red rectangle on our bottom?
            elif stop_c+1 < output_grid.shape[1] and output_grid[stop_r, stop_c+1] != Color.BLACK:
                dc = -dc
            # did we hit the red rectangle on our top?
            elif stop_c-1 >= 0 and output_grid[stop_r, stop_c-1] != Color.BLACK:
                dc = -dc
            else:
                # didn't do any reflections, so stop
                break

            start_r, start_c = stop_r + dr, stop_c + dc
    
    return output_grid


def generate_input():
    # Make a grid with a red rectangle on the bottom and a teal diagonal line at the top pointing at it
    # Then randomly rotate to get a variety of orientations

    width, height = np.random.randint(10, 25), np.random.randint(8, 10)
    grid = np.full((width, height), Color.BLACK)

    red_height = np.random.randint(2, 5)
    grid[:, -red_height:] = Color.RED

    # Make a diagonal line, which always begins at the top
    line_r = np.random.randint(0, width//2)
    # always begins at the top
    line_c = 0

    draw_line(grid, line_r, line_c, direction=(1, 1), color=Color.TEAL, length=2)

    # randomly rotate to get a variety of orientations
    grid = np.rot90(grid,-(np.random.randint(0, 4)))

    return grid
