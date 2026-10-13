
import numpy as np
from typing import *
import random

from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, paste, random_free_location


# concepts:
# patterns, growing, horizontal/vertical bars

# description:
# In the input you will see a 30x30 grid with at least two rectangles, each with at least one special pixel of a different color, and a crosshair-type pattern outside these rectangles.
# For each of these special pixels, apply the crosshair pattern and extend the points inside the rectangle that special pixel is in, until it reaches the edge.

def main(input_grid):
    """
    Apply the external crosshair pattern to each special pixel inside the rectangles.

    Input structure:
    - The most common color is the background.
    - The smallest non-background component is a crosshair instruction sprite.
    - Larger components are rectangles containing one or more special pixels of another color.

    Rule:
    - Crop the crosshair sprite and infer whether it extends horizontally, vertically, or both.
    - For each rectangle, paste the crosshair centered on every special pixel.
    - Extend the crosshair point color through rectangle-colored cells along the enabled row or column.

    Key insight:
    - The small outside crosshair is an instruction object; each special pixel becomes the center
      of that same growth pattern inside its rectangle.
    """
    # first get the grid size
    n, m = input_grid.shape

    # figure out background color by assuming it is the most common color
    background_color = np.bincount(input_grid.flatten()).argmax()

    # with this background color, find the polychromatic objects
    objects = find_components(input_grid, bg=background_color, connectivity=8, monochromatic=False, return_format="objects")

    # sort the objects by their size in terms of area, the crosshair is the smallest object
    sorted_objects = sorted(objects, key=lambda r: np.count_nonzero(r != background_color))
    crosshair_object = sorted_objects[0]
    rectangles = sorted_objects[1:]

    # now find the crosshair sprite coordinates
    _b = bbox(crosshair_object != background_color)
    assert _b is not None
    crosshair_sprite = crop_bbox(crosshair_object, _b)
    width, height = crosshair_sprite.shape
    
    # if the crosshair is wider than it is tall, it extends horizontally, vertically if taller, both if square
    horizontal = True
    vertical = True
    if width > height:
        vertical = False
        point_color = crosshair_sprite[0, height // 2]
    elif height > width:
        horizontal = False
        point_color = crosshair_sprite[width // 2, 0]
    else:
        point_color = crosshair_sprite[width // 2, 0]

    # now we prepare the output grid
    output_grid = np.full_like(input_grid, background_color)

    # for each rectangle, crop it to just the rectangle, find the special pixel, extend the crosshair pattern from it, then add it back to the grid
    for rectangle in rectangles:
        # crop the rectangle to just the rectangle, while preserving its position in the grid
        _b = bbox(rectangle != background_color)
        assert _b is not None
        rec_r, rec_c, _r1, _c1 = _b
        w = _r1 - rec_r + 1
        h = _c1 - rec_c + 1
        _b = bbox(rectangle != background_color)
        assert _b is not None
        cropped_rectangle = crop_bbox(rectangle, _b)

        # find the special color, it is the least common color in the rectangle
        colors, counts = np.unique(cropped_rectangle, return_counts=True)
        # colors are sorted by their frequency, so choose least common as the special color
        special_color = colors[-1]
        rectangle_color = colors[-2]

        # for each special pixel, extend the crosshair pattern
        for r, c in np.argwhere(cropped_rectangle == special_color):
            # first color the special pixel with the crosshair sprite centered on it
            cropped_rectangle = paste(cropped_rectangle, crosshair_sprite, r - width // 2, c - height // 2, transparent=background_color, clip=True)

            # then extend the points in the crosshair pattern until they reach the edge of the rectangle
            if horizontal:
                for x0 in range(w):
                    if cropped_rectangle[x0, c] == rectangle_color:
                        cropped_rectangle[x0, c] = point_color
            if vertical:
                for y0 in range(h):
                    if cropped_rectangle[r, y0] == rectangle_color:
                        cropped_rectangle[r, y0] = point_color
        
        # add the rectangle back to the grid
        paste(output_grid, cropped_rectangle, rec_r, rec_c, transparent=background_color, clip=True)

    return output_grid


def generate_input():
    H = W = 30
    background = Color.BLACK
    grid = np.full((H, W), background, dtype=int)

    # Colors chosen so that `np.unique` ordering matches the solver's "special_color = colors[-1]" logic.
    rectangle_color = Color.BLUE
    special_color = Color.MAROON
    point_color = Color.TEAL

    # Small crosshair object (smallest component).
    crosshair = np.full((3, 3), background, dtype=int)
    crosshair[1, :] = point_color
    crosshair[:, 1] = point_color
    paste(grid, crosshair, 1, 1, transparent=background, clip=False)

    # Place 2 rectangles, each with one special pixel of `special_color`.
    for _ in range(2):
        rh = random.randint(6, 10)
        rw = random.randint(6, 10)
        rect = np.full((rh, rw), rectangle_color, dtype=int)
        rect[rh // 2, rw // 2] = special_color
        r0, c0 = random_free_location(grid, rect, transparent=background, border=1, padding=1, padding_connectivity=8)
        paste(grid, rect, r0, c0, transparent=background, clip=True)

    return grid
