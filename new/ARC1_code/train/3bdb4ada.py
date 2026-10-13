
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste

# concepts:
# punching holes, geometric pattern

# description:
# In the input you will see a grid with some rectangles. Each rectangle is three pixels along its short side, and is monochromatic.
# To make the output, you should draw black pixels along the middle of each rectangle with a period of 2 starting one pixel in.
# Effectively punching holes in the middle of each rectangle, skipping alternating pixels.

def main(input_grid):
    """
    Punch alternating black holes through the center line of each monochromatic 3-wide rectangle.

    Input structure:
    - The grid contains one or more separated non-black monochromatic rectangles.
    - Each rectangle has short side length 3 and an odd long side.
    - Rectangles may be horizontal or vertical on a black background.

    Rule:
    - Detect each rectangle and rotate vertical ones into a horizontal working orientation.
    - Set every other pixel along the middle line, starting one pixel in, to black.
    - Restore the original orientation and paste the edited rectangle back onto a black output.

    Key insight:
    - The hole pattern is defined along the rectangle's long axis, so orientation is handled
      by canonicalizing the rectangle before editing it.
    """
    # Plan:
    # 1. Extract the rectangles from the input grid
    # 2. Canonicalize the rectangles: ensure that they are horizontal (remember to rotate back otherwise)
    # 3. Draw black pixels inside each rectangle (horizontally, skipping every other pixel)
    # 4. Rotate the rectangle back if it was not originally horizontal

    # 1. Extract the rectangles from the input grid
    rectangle_objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    output_grid = np.full_like(input_grid, Color.BLACK)
    for rectangle_object in rectangle_objects:
        # 2. Canonicalize the rectangle sprite
        _b = bbox(rectangle_object != Color.BLACK)
        assert _b is not None
        original_r, original_c, _r1, _c1 = _b
        width = _r1 - original_r + 1
        height = _c1 - original_c + 1
        # crop to convert object to sprite
        _b = bbox(rectangle_object != Color.BLACK)
        assert _b is not None
        rectangle_sprite = crop_bbox(rectangle_object, _b)

        # Flip it to be horizontal if it isn't already
        is_horizontal = width > height
        if not is_horizontal:
            rectangle_sprite = np.rot90(rectangle_sprite)
            width, height = height, width

        # 3. Punch holes through the middle of the rectangle
        # The inner row is y=height//2
        for r in range(1, width, 2):
            rectangle_sprite[r, height//2] = Color.BLACK
        
        # 4. Rotate back if it was originally vertical, and then draw it to the output grid
        if not is_horizontal:
            rectangle_sprite = np.rot90(rectangle_sprite, k=-1)
        
        # draw it back in its original location
        paste(output_grid, rectangle_sprite, original_r, original_c, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Generate the background grid
    width, height = np.random.randint(10, 30), np.random.randint(10, 20)
    grid = np.zeros((width, height), dtype=int)

    # Randomly choose the number of rectangles on the grid
    num_rectangles = np.random.randint(1, 4)
    # Randomly choose colors for the rectangles
    colors = np.random.choice(Color.NOT_BLACK, num_rectangles, replace=False)

    for color in colors:
        # Randomly choose the width of the rectangle, should not be too short and should be odd
        w, h = np.random.randint(2, 6) * 2 + 1, 3
        rectangle_sprite = np.full((w, h), color)

        # randomly flipped to create different orientations
        if np.random.rand() < 0.5:
            rectangle_sprite = np.rot90(rectangle_sprite, k=-1)

        # Randomly choose the position of the rectangle
        try:
            _loc = first_free_location(grid, rectangle_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except:
            # If there is no free location for the rectangle, retry
            return generate_input()
        paste(grid, rectangle_sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
