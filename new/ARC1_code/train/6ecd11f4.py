
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, contact, draw_line, find_components, is_contiguous, paste, random_free_location, random_sprite

# concepts:
# objects, color guide, masking, scaling

# description:
# In the input you will see a large monochromatic object, and a smaller square object with many colors.
# To make the output, first downscale the large object so that it is the same size as the small multicolor square, and then use the large object as a binary mask to zero out pixels in the square. Return the resulting masked sprite.

def main(input_grid):
    """
    Use a downscaled monochrome pattern as a mask on the multicolor square.

    Input structure:
    - The input is a black grid with one large single-color pattern object.
    - A smaller dense multicolor square appears elsewhere.
    - The pattern is an integer scale-up of a square mask.

    Rule:
    - Separate the monochrome pattern from the multicolor square.
    - Crop both objects and downsample the pattern to the square's size.
    - Keep square pixels where the downsampled pattern is non-black and set the rest to black.

    Key insight:
    - The large object is a binary stencil that aligns with the square after integer downscaling.
    """
    # find all the objects in the input grid
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=False, return_format="objects")

    # figure out which object is the pattern and which object is the multi-colored square
    pattern = square = None
    for obj in objects:
        # if the object has only one color and black, it is the pattern. Otherwise, it is the multi-colored square.
        if len(set(obj.flatten())) == 2:
            pattern = obj
        else:
            square = obj
    
    # get the square location and cut out the bounding box
    _b = bbox(square != Color.BLACK)
    assert _b is not None
    r, c, _r1, _c1 = _b
    width = _r1 - r + 1
    height = _c1 - c + 1
    square_sprite = square[r:r+width, c:c+height]

    # cut out the bounding box of the pattern
    _b = bbox(pattern != Color.BLACK)
    assert _b is not None
    x2, y2, _r1, _c1 = _b
    width2 = _r1 - x2 + 1
    height2 = _c1 - y2 + 1
    # make sure the pattern is square
    if width2 != height2:
        width2 = height2 = max(width2, height2)
    pattern_sprite = pattern[x2:x2+width2, y2:y2+height2]

    # figure out how much bigger the pattern is than the square
    scale = width2 // width

    # scale down the pattern to fit the square
    scaled_pattern = np.zeros_like(square_sprite) 
    for i in range(width):
        for j in range(height):
            scaled_pattern[i,j] = pattern_sprite[i * scale, j * scale]
    
    # if the pixel is in the pattern, keep the color from the square otherwise make it black in the output grid
    output_grid = np.where(scaled_pattern, square_sprite, Color.BLACK)

    return output_grid


def generate_input():
    # decide how big the multi-colored square will be
    size = np.random.randint(3, 5)

    # make the multi-colored square with all colors except black
    square = random_sprite(size, size, density=1, symmetry="not_symmetric", color_palette=Color.NOT_BLACK)
    
    # make sure the square has at least 2 non-background colors
    if len(set(map(int, np.unique(square))) - {Color.BLACK}) < 2:
        return generate_input()

    # make a random pattern that is the same size as the multi-colored square but only uses one color 
    color = np.random.choice(list(Color.NOT_BLACK))
    pattern = random_sprite(size, size, density=.7, symmetry="not_symmetric", color_palette=[color], connectivity=8)

    # check that pattern is continuous, if not then try again
    if not is_contiguous(pattern, connectivity=8):
        return generate_input()

    # decide how much to scale up the pattern
    scale = np.random.randint(2, 5)

    # scale up the pattern
    pattern = np.repeat(np.repeat(pattern, scale, axis=1), scale, axis=0)

    # make a grid large enough to fit both the pattern and the multi-colored square without touching
    n = m = (size * scale) + (2 * size) + np.random.randint(2, 5)
    grid = np.zeros((n, m), dtype=int)

    # put the pattern on the grid randomly
    r, c = np.random.randint(0, n - size * scale), np.random.randint(0, m - size * scale)
    paste(grid, pattern, r, c, transparent=Color.BLACK, clip=True)

    # put the multi-colored square on the grid randomly but not touching the pattern
    try:
        x2, y2 = random_free_location(grid, square, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
    except ValueError:
        return generate_input()
    paste(grid, square, x2, y2, transparent=Color.BLACK, clip=True)

    return grid


