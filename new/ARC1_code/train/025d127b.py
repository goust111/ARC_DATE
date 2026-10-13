
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, find_components, first_free_location, paste

# concepts:
# objects, pixel manipulation

# description:
# In the input you will see a set of objects, each consisting of a horizontal top/bottom and diagonal left/right edges (but that structure is not important)
# To make the output shift right each pixel in the object *except* when there are no other pixels down and to the right

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Shift each object pixel one step right unless it is on the object's lower-right frontier.

    Input structure:
    - The input contains one or more same-color objects on a black background.
    - Objects are connected components and may have diagonal or slanted edges.
    - Each object is processed independently in its original position.

    Rule:
    - For every non-black pixel in an object, look for any other object pixel down and to the right.
    - If such a pixel exists, move the current pixel one column to the right.
    - If not, leave the pixel in place.

    Key insight:
    - The local move is controlled by the object's southeast support: pixels with something beyond
      them down-right slide, while the terminal southeast contour stays fixed.
    """
    # find the connected components, which are monochromatic objects
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")

    output_grid = np.zeros_like(input_grid)

    for obj in objects:
        transformed_object = np.zeros_like(obj)

        for r in range(obj.shape[0]):
            for c in range(obj.shape[1]):
                if obj[r, c] != Color.BLACK:
                    # check that there are other colored pixels down and to the right
                    down_and_to_the_right = obj[r+1:, c+1:]
                    if np.any(down_and_to_the_right != Color.BLACK):
                        transformed_object[r, c+1] = obj[r, c]
                    else:
                        transformed_object[r, c] = obj[r, c]

        paste(output_grid, transformed_object, 0, 0, transparent=Color.BLACK, clip=True)

    return output_grid


def generate_input():
    n, m = np.random.randint(10, 30), np.random.randint(10, 30)
    grid = np.full((m, n), Color.BLACK)

    n_objects = np.random.randint(1, 3)

    for _ in range(n_objects):
        color = random.choice(Color.NOT_BLACK)

        H, W = grid.shape
        # The object will have shape (bar_width + side_height, side_height) and must fit within the border.
        max_side = min(W - 4, (H - 4) - 3)  # bar_width >= 3
        if max_side < 3:
            continue
        side_height = np.random.randint(3, max_side + 1)
        max_bar = min((H - 4) - side_height, H // 2)
        if max_bar < 3:
            continue
        bar_width = np.random.randint(3, max_bar + 1)

        width, height = bar_width + side_height, side_height
        obj = np.zeros((width, height), dtype=int)

        # make the horizontal top edge
        obj[:bar_width+1, 0] = color
        # make the horizontal bottom edge
        obj[-bar_width:, -1] = color
        # make the diagonal left edge
        for i in range(side_height):
            obj[i, i] = color
        # make the diagonal right edge
        for i in range(side_height-1):
            obj[bar_width+i+1, i] = color

        # place the object randomly on the grid, assuming we can find a spot
        try:
            _loc = first_free_location(grid, obj, transparent=Color.BLACK, border=2, padding=2, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except:
            continue

        paste(grid, obj, r, c, transparent=Color.BLACK, clip=True)

    # Make sure that we actually generated something
    if np.all(grid == Color.BLACK):
        return generate_input()
    
    return grid
