import numpy as np

import random
from arc_utils import Color, bbox, crop_bbox, draw_line, first_free_location, paste, random_sprite

# concepts:
# cropping

# description:
# In the input you will see a single colored shape, around 4x6 in size, floating in a 12x12 grid of black.
# To make the output, crop the background out of the image - so the output grid has the same dimensions as the shape.

def main(input_grid):
    """
    Crop away the black background and return the tight box around the single colored object.

    Input structure:
    - The grid has a black background.
    - Exactly one non-black sprite is placed somewhere inside the grid.
    - The sprite may use any non-black color.

    Rule:
    - Find the bounding box of all non-black cells.
    - Crop the input grid to that bounding box.
    - Return the cropped sprite with its original colors.

    Key insight:
    - The output is determined only by the non-black support; the surrounding black margin is discarded.
    """
    b = bbox(input_grid != Color.BLACK)
    assert b is not None
    return crop_bbox(input_grid, b)

def generate_input():
    # create a roughly 12x12 input grid
    n = np.random.randint(10, 15)
    m = np.random.randint(10, 15)
    grid = np.full((n, m), Color.BLACK, dtype=int)

    # create a small random sprite with shape (2-7)x(2-7)
    w = np.random.randint(2, 8)
    h = np.random.randint(2, 8)
    sprite = random_sprite(w, h, color_palette=[np.random.choice(Color.NOT_BLACK)])

    # blit the spite onto a random location on the grid
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    return grid
# ============= remove below this point for prompting =============
