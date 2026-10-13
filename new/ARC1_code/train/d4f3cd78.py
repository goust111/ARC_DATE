
import random

import numpy as np
from typing import *
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components, paste, random_free_location

# concepts:
# cups, filling

# description:
# In the input you will see grey cups, meaning an almost-enclosed shape with a small opening on one of its sides, and empty space (black pixels) inside.
# To make the output grid, you should fill the interior of each cup with teal, then shoot teal out of the opening of the cup straight out in a line.

def main(input_grid):
    """
    Fill the inside of a cup and extend teal outward through its opening.

    Input structure:
    - The grid contains one non-black cup-shaped object on a black background.
    - The cup is an almost closed rectangular outline with one boundary opening.
    - Black cells inside the cup and at the opening define the fill and exit path.

    Rule:
    - Find the cup's bounding box and mark the black cells inside it.
    - Fill those interior cells with teal.
    - Detect the opening side and draw teal straight outward from every opening cell.

    Key insight:
    - The only boundary gap identifies both the cup interior and the direction for the
      emitted teal line.
    """
    # Plan:
    # 1. Detect the cup
    # 2. Find the mask of the inside of the cup
    # 3. Find the mask of the opening of the cup (on one of its sides)
    # 4. Fill the cup with teal
    # 5. Shoot pixels outward from the opening (straight out)
    
    # 1. Detect cup
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    assert len(objects) == 1, "There should be exactly one cup"
    obj = list(objects)[0]

    output_grid = input_grid.copy()

    # 2. Extract what's inside the cup (as its own object), which is everything in the bounding box that is not the object itself
    _b = bbox(obj != Color.BLACK)
    assert _b is not None
    cup_r0, cup_c0, cup_r1, cup_c1 = _b
    inside_cup_mask = np.zeros_like(input_grid, dtype=bool)
    inside_cup_mask[cup_r0 : cup_r1 + 1, cup_c0 : cup_c1 + 1] = True
    inside_cup_mask = inside_cup_mask & (obj == Color.BLACK)

    # 3. Extract the hole in the cup, which is what's inside and on the boundary of the bounding box
    # what's inside...
    hole_mask = inside_cup_mask.copy()
    # ...and then we need to remove anything not on the boundary
    hole_mask[cup_r0 + 1 : cup_r1, cup_c0 + 1 : cup_c1] = False

    # 4. Fill the cup with teal
    output_grid[inside_cup_mask] = Color.TEAL

    # 5. Shoot pixels outward from the opening (straight out)
    # Find the direction of the opening, which is the unit vector that points away from the interior
    hole_bbox = bbox(hole_mask)
    assert hole_bbox is not None
    hole_r, hole_c = bbox_anchor(hole_bbox, anchor="center")

    direction: tuple[int, int] | None = None
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        rr = hole_r - dr
        cc = hole_c - dc
        if 0 <= rr < input_grid.shape[0] and 0 <= cc < input_grid.shape[1] and inside_cup_mask[rr, cc]:
            direction = (dr, dc)
            break
    assert direction is not None
    # Loop over every boundary pixel and shoot outward
    for r, c in np.argwhere(hole_mask):
        draw_line(output_grid, int(r), int(c), direction=direction, color=Color.TEAL)
    return output_grid

def generate_input():
    # Generate the grid with random size
    width = np.random.randint(12, 30)
    height = np.random.randint(12, 30)
    grid = np.full((width, height), Color.BLACK)

    # Pick a random width/height for this cup
    cup_width = np.random.randint(4, 8)
    cup_height = np.random.randint(3, 8)

    # Make a sprite, which is just going to be a grey outline of a rectangle with a hole on one of its sides
    sprite = np.full((cup_width, cup_height), Color.BLACK)
    color = random.choice([ color for color in Color.NOT_BLACK if color != Color.TEAL ])
    sprite[0, :] = color
    sprite[-1, :] = color
    sprite[:, 0] = color
    sprite[:, -1] = color

    # Make the hole randomly somewhere on the edges (but not in the corners)
    edges_but_not_corners = [ (0, c) for c in range(1, cup_height-1) ] + [ (cup_width-1, c) for c in range(1, cup_height-1) ] + [ (r, 0) for r in range(1, cup_width-1) ] + [ (r, cup_height-1) for r in range(1, cup_width-1) ]
    hole_r, hole_c = random.choice(edges_but_not_corners)
    sprite[hole_r, hole_c] = Color.BLACK

    # Find a random free location for it
    free_r, free_c = random_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
    paste(grid, sprite, free_r, free_c, transparent=Color.BLACK, clip=True)

    return grid    
