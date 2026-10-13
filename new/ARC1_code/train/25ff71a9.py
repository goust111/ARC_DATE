import numpy as np
import random
from arc_utils import Color, draw_line, find_components, paste, random_sprite, translate



# concepts:
# sliding objects

# description:
# In the input you will see a 3x3 grid with a contiguous shape on it.
# Slide the shape down by one pixel.

def main(input_grid):
    """
    Slide the single colored object down by one pixel on a black 3 by 3 grid.

    Input structure:
    - The grid contains one monochromatic connected object.
    - The background is black.
    - The object has room to move down one row.

    Rule:
    - Detect the object's connected component.
    - Translate that component by one row downward.
    - Return the translated object on a black background.

    Key insight:
    - The whole connected shape moves rigidly; no cells are added, removed, or recolored.
    """
    # find the connected component, which is a monochromatic object
    objects = find_components(
        input_grid,
        bg=Color.BLACK,
        connectivity=8,
        monochromatic=True,
        return_format="objects",
    )
    obj = objects[0]

    # translate the object down by one pixel
    output_grid = translate(obj, dr=1, dc=0, background=Color.BLACK)

    return output_grid


def generate_input():
    # first create blank grid
    grid = np.zeros((3, 3), dtype=int)

    # while the grid is empty
    while grid.sum() == 0:

        # generate a random up to 3x3 sprite
        n = random.randint(1, 3)
        m = random.randint(1, 2)
        random_sprite_to_add = random_sprite(n, m, symmetry='not_symmetric', color_palette=[random.choice(Color.NOT_BLACK)])

        # choose a random location on the grid, ensuring the bottom row is empty
        r = random.randint(0, 2)
        c = random.randint(0, 2 - m)

        # place the sprite onto the grid
        paste(grid, random_sprite_to_add, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
