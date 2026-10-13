
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, find_components, paste

# concepts:
# sliding objects

# description:
# In the input grid, you will see a chevron-shaped object of one color in a black grid, with pixels of another color scattered around the grid.
# To produce the output grid, take all pixels located underneath the chevron. For each of these pixels, extend a vertical line of the same color up and down, until reaching the bottom of the grid or the boundary of the chevron.

def main(input_grid):
    """
    Extend colored pixels downward from the underside of a chevron when the chevron is above them.

    Input structure:
    - The input is a black grid with one large chevron-shaped object.
    - A second color appears as scattered pixels.
    - Only scattered pixels in columns beneath the chevron affect the output.

    Rule:
    - Identify the chevron as the largest object and find its color.
    - For each pixel of the other color, search upward in its column for the chevron.
    - If found, fill that column from just below the chevron to the bottom with the pixel color.

    Key insight:
    - The chevron acts as an overhead boundary that activates vertical fills only in covered columns.
    """
    # 1. find the chevron: it is the largest object by size.
    # 2. get the color of the chevron
    # 3. get the color of the colored pixels in the grid.
    # 4. for each colored pixel, check if the chevron is above it. if so, extend a line of the same color above and below it until we reach the bottom of the grid or the boundary of the chevron.

    # get the chevron
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    chevron = max(objects, key=lambda o: np.count_nonzero(o))

    # get the color of the chevron
    chevron_color = chevron[chevron != Color.BLACK][0]

    # get the color of the colored pixels (the other color in the grid)
    colors = np.unique(input_grid)
    colors = [c for c in colors if c not in [Color.BLACK, chevron_color]]
    assert len(colors) == 1
    pixel_color = colors[0]

    # for each colored pixel, check if chevron is above it in the same column.
    # If so, fill that column from just below the chevron down to the bottom.
    H, W = input_grid.shape
    output_grid = input_grid.copy()
    for r in range(H):
        for c in range(W):
            if int(input_grid[r, c]) != int(pixel_color):
                continue

            rr = r
            while rr >= 0 and int(input_grid[rr, c]) != int(chevron_color):
                rr -= 1
            if rr < 0:
                continue

            output_grid[rr + 1 :, c] = int(pixel_color)

    return output_grid


def generate_input():
    # create a 20x20 black grid
    input_grid = np.full((20, 20), Color.BLACK)

    # choose a chevron color and pixel color
    chevron_color, pixel_color = np.random.choice(Color.NOT_BLACK, 2, replace=False)

    # create the chevron
    chevron = np.full((7, 4), Color.BLACK)
    # Create coordinate arrays
    r, c = np.indices(chevron.shape)
    # fill in the chevron
    chevron[np.logical_and(c == 0, r == 3)] = chevron_color
    chevron[np.logical_and(c == 1, np.logical_and(r >= 2, r <= 4))] = chevron_color
    chevron[np.logical_and(c == 2, np.logical_and(r >= 1, r <= 5))] = chevron_color
    chevron[np.logical_and(c == 3, np.logical_or(r == 0, r == 6))] = chevron_color
    # put the chevron at a random location in the upper half of the grid
    r, c = np.random.randint(0, 20 - 7), np.random.randint(0, 10 - 4)
    paste(input_grid, chevron, r, c, transparent=Color.BLACK, clip=True)

    # generate 5-25 pixels at random (unfilled) spots.
    n_pixels = np.random.randint(5, 26)
    r_choices, c_choices = np.where(input_grid == Color.BLACK)
    location_choices = list(zip(r_choices, c_choices))
    pixel_locations = random.sample(location_choices, n_pixels)
    for r, c in pixel_locations:
        input_grid[r, c] = pixel_color

    return input_grid
