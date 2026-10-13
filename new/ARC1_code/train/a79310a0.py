import numpy as np

import random
from arc_utils import Color, draw_line, find_components, first_free_location, paste, random_sprite, translate

# concepts:
# translation, color change

# description:
# In the input you will see a grid with a teal object.
# To make the output grid, you should translate the teal object down by 1 pixel and change its color to red.

def main(input_grid):
    """
    Move the single teal object down one row and recolor it red.

    Input structure:
    - One teal object appears on a black background.
    - The object has room to move down one pixel.
    - No other foreground objects are expected.

    Rule:
    - Extract the single foreground object.
    - Recolor every foreground cell red.
    - Translate the object one row downward on a black output canvas.

    Key insight:
    - The object shape is preserved exactly; only color and vertical position change.
    """
    # Plan:
    # 1. Find the object (it's the only one)
    # 2. Change its color to red
    # 3. Translate it downward by 1 pixel

    
    # Get the single teal object
    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=4, monochromatic=False, return_format="objects")
    assert len(objects) == 1
    teal_object = objects[0]

    # Make a blank output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Change its color to red
    teal_object = teal_object.copy()
    teal_object[teal_object != bg] = Color.RED

    # Translate it downward by 1 pixel
    teal_object = translate(teal_object, dr=1, dc=0, background=bg)

    # Blit the teal object onto the output grid
    output_grid = paste(output_grid, teal_object, 0, 0, transparent=bg)

    return output_grid

def generate_input():
    # Generate the background grid with size of n x n.
    grid_len = np.random.randint(4, 8)
    grid = np.zeros((grid_len, grid_len), dtype=int)

    # Randomly generate the teal object and place it on the grid.
    sprite_width, sprite_height = np.random.randint(1, grid_len - 1), np.random.randint(1, grid_len -1)
    sprite = random_sprite(n=sprite_width, m=sprite_height, color_palette=[Color.TEAL], density=0.5)
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    grid = paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    return grid
# ============= remove below this point for prompting =============
