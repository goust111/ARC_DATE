
import numpy as np
from typing import *
import random
from arc_utils import Color, collision, detect_objects, draw_line, first_free_location, paste, translate

# concepts:
# alignment, copy to object border

# description:
# In the input you will see a teal object on a black background, and several colored pixels on the border of canvas.
# To make the output grid, you should copy the colored pixels horizontally/vertically so that they are just barely overlapping/colliding with the teal object.

def main(input_grid):
    """
    Project each colored border pixel inward until it collides with the teal object.

    Input structure:
    - A black grid contains one teal object.
    - Several one-cell non-black, non-teal objects sit on the canvas border.
    - Each border pixel can reach the teal object by sliding horizontally or vertically.

    Rule:
    - Detect the teal object and all other single-pixel colored objects.
    - Try cardinal translations for each colored pixel.
    - Paste the first translated pixel position that overlaps the teal object.

    Key insight:
    - The border pixels act like projected markers: their colors are copied along a row or column
      to the first position that touches the teal shape.
    """
    # Plan:
    # 1. Detect the teal object
    # 2. Detect the colored pixels on the border
    # 3. Slide the colored pixels in the 4 cardinal directions until we find how to make them overlapping with the teal object

    output_grid = np.copy(input_grid)

    # Detects the rectangle in the input grid that is TEAL
    teal_objects = detect_objects(grid=input_grid, colors=[Color.TEAL], monochromatic=True, connectivity=4)
    
    # There should only be one rectangle of the color TEAL has been detected in the grid.
    assert len(teal_objects) == 1
    teal_object = teal_objects[0]

    # colored pixels are NOT black and NOT TEAL.
    colors_except_teal = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    
    # Detects all other colored pixels in the grid 
    pixels = detect_objects(grid=input_grid,
                            # Exclude teal from the search
                            colors=colors_except_teal, 
                            # only consider single pixels
                            allowed_dimensions=[(1,1)], 
                            monochromatic=True, connectivity=4)

    # Copy the colored pixels to the teal object by moving them either vertically or horizontally.
    for pixel in pixels:
        # consider translating the pixel in the 4 cardinal directions, and consider translating as far as possible
        possible_displacements = [ (slide_distance*dr, slide_distance*dc)
                                   # We could slide as far as the maximum grid extent
                                   for slide_distance in range(max(input_grid.shape))
                                   # (dx, dy) ranges over 4 cardinal directions
                                   for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)] ]
        for dr, dc in possible_displacements:
            # check if the objects are colliding/overlapping after translating
            translated_pixel = translate(pixel, dr, dc, background=Color.BLACK)
            if collision(object1=teal_object, object2=translated_pixel):
                # put the red object where it belongs
                paste(output_grid, translated_pixel, 0, 0, transparent=Color.BLACK, clip=True)
                break
    
    return output_grid


def generate_input():
    # Initialize a 10x10 grid representing a black background.
    n = m = 10
    grid = np.zeros((n, m), dtype=int)
    
    # Randomly determine the width and height of the TEAL rectangle between 2 and 5.
    width, height = np.random.randint(2, 6), np.random.randint(2, 6)
    teal_sprite = np.full((width, height), Color.TEAL)

    # Find a free location for this sprite and blit it to the grid
    _loc = first_free_location(grid, teal_sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, cc = _loc
    paste(grid, teal_sprite, r, cc, transparent=Color.BLACK, clip=True)

    # list to hold the available positions, which are all on the border of the canvas
    border_locations = [ (0, cc) for cc in range(n) ] + [ (m-1, cc) for cc in range(n) ] + [ (r, 0) for r in range(1, m-1) ] + [ (r, n-1) for r in range(1, m-1) ]
    
    # Put a random number of colored pixels on the border of the canvas
    for _ in range(np.random.randint(3, 8)):
        # Pick a random location on the border
        r, cc = random.choice(border_locations)
        # Pick a random color that is not black or teal
        color = random.choice([c for c in Color.NOT_BLACK if c != Color.TEAL])

        grid[r, cc] = color
    
    return grid 
