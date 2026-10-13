
import numpy as np
from typing import *
import random
from arc_utils import Color, detect_rotational_symmetry, draw_line, first_free_location, paste, random_sprite

# concepts:
# symmetry detection, occlusion

# description:
# In the input you will see an object that is almost rotationally symmetric, except that some of it has been removed (covered in black pixels)
# To make the output fill in the missing parts of the object to make it rotationally symmetric


def main(input_grid):
    """
    Fill missing cells so the visible object becomes rotationally symmetric.

    Input structure:
    - The input contains one partly occluded non-black object on a black background.
    - The completed object has 90-degree rotational symmetry.
    - Existing non-black pixels already agree with that symmetry.

    Rule:
    - Detect the object's center of rotational symmetry.
    - Rotate every colored pixel through the other three quarter-turn positions.
    - Fill any black rotated positions with the source pixel's color.

    Key insight:
    - Every visible pixel implies three rotational counterparts; black counterparts are missing
      pieces that can be restored directly.
    """
    # Plan:
    # 1. Find the center of rotation
    # 2. Rotate each colored pixel (4 times, rotating around the center of rotation) and fill in any missing pixels
    output_grid = input_grid.copy()

    # Find the rotational symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # Find the colored pixels
    colored_pixels = np.argwhere(input_grid != Color.BLACK)

    # Do the rotations and fill in the missing colors
    for r, c in colored_pixels:
        # Get the color, which is going to be copied to the rotated positions
        color = input_grid[r, c]
        
        # Loop over all rotations, going 90 degrees each time (so four times)
        for i in range(1, 4):
            # Calculate rotated coordinate
            rotated_r, rotated_c = sym.apply(r, c, iters=i)

            # Fill in the missing pixel
            if output_grid[rotated_r, rotated_c] == Color.BLACK:
                output_grid[rotated_r, rotated_c] = color
            else:
                assert output_grid[rotated_r, rotated_c] == color, "The object is not rotationally symmetric"

    return output_grid


def generate_input():
    # Initialize 10x10 grid
    grid = np.zeros((10, 10), dtype=int)

    # Create a single-color 5x5 radially symmetric sprite, so the completed object
    # is rotationally symmetric in color (not just shape).
    obj_color = int(random.choice(Color.NOT_BLACK))
    sprite = random_sprite(5, 5, density=0.3, symmetry="radial", color_palette=[obj_color], background=Color.BLACK)

    # Randomly remove pixels from sprite
    for i in range(sprite.shape[0]):
        for j in range(sprite.shape[1]):
            if random.random() < 0.2:
                sprite[i, j] = Color.BLACK

    # Place sprite randomly onto the grid
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
