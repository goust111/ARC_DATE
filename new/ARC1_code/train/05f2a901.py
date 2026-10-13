import numpy as np
import random

from arc_utils import Color, contact, first_free_location, paste, random_sprite, translate

# concepts:
# collision detection, sliding objects

# description:
# In the input you will see a teal 2x2 square and a red object (the red object might be irregular in its shape)
# Slide the red object in any of the four directions until it just touches the teal square

def main(input_grid):
    """
    Slide the red object in a cardinal direction until it first touches the fixed teal square.

    Input structure:
    - The input contains one teal 2x2 square and one red object on a black background.
    - The red object is aligned so a horizontal or vertical slide can make contact.
    - The teal square remains fixed in the output.

    Rule:
    - Keep only the teal square as the starting output.
    - Try cardinal translations of the red object from shortest to longest.
    - Return the first translation where the red object touches the teal square by 4-connectivity.

    Key insight:
    - The task is a minimal-contact slide: the red shape moves straight toward the teal anchor and
      stops as soon as their edges touch.
    """

    # get just the teal object
    teal_object = np.zeros_like(input_grid)
    teal_object[input_grid == Color.TEAL] = Color.TEAL

    # get just the red object
    red_object = np.zeros_like(input_grid)
    red_object[input_grid == Color.RED] = Color.RED

    # the output grid starts with just the teal object
    output_grid = np.copy(teal_object)
    
    # consider sliding in the 4 cardinal directions, and consider sliding as far as possible
    possible_displacements = [
        (slide_distance * dr, slide_distance * dc)
        for slide_distance in range(max(input_grid.shape))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
    ]
    for dr, dc in possible_displacements:
        translated_red_object = translate(red_object, dr=dr, dc=dc, background=Color.BLACK)
        if contact(object1=teal_object, object2=translated_red_object, background=Color.BLACK, connectivity=4):
            return paste(output_grid, translated_red_object, 0, 0, transparent=Color.BLACK)
            
    assert 0, "No valid slide found"

def generate_input():
    # Make a black grid first as background.
    n, m = random.randint(5, 20), random.randint(5, 20)
    grid = np.full((m, n), Color.BLACK)

    # Make a 2x2 teal square, put it somewhere on the grid.
    square_sprite = np.full((2, 2), Color.TEAL)
    _loc = first_free_location(grid, square_sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
    if _loc is None:
        return generate_input()
    r, c = _loc
    paste(grid, square_sprite, r, c, transparent=Color.BLACK)

    # Make a random sprite of [3,4] x [3,4] with symmetry disabled and the color red.
    sprite = random_sprite(m=[3, 4], n=[3, 4], symmetry="not_symmetric", color_palette=[Color.RED])

    # Put the sprite somewhere on the grid (non-overlapping, with a gap).
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
    if _loc is None:
        return generate_input()
    r, c = _loc
    paste(grid, sprite, r, c, transparent=Color.BLACK)

    # check that we could slide the object either vertically or horizontally in order to touch the red square
    # this will be true if there is a row or column that has both red and blue
    H, W = grid.shape
    for r in range(H):
        if Color.TEAL in grid[r, :] and Color.RED in grid[r, :]:
            return grid
    for c in range(W):
        if Color.TEAL in grid[:, c] and Color.RED in grid[:, c]:
            return grid
    
    # if not, try again
    return generate_input()


# ============= remove below this point for prompting =============
