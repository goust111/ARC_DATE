
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components, paste, random_free_location

# concepts:
# cups, filling

# description:
# In the input you will see a cups, meaning an almost-enclosed shape with a small opening on one of its sides, and empty space (black pixels) inside.
# To make the output grid, you should fill the interior of each cup with yellow, then shoot yellow out of the opening of the cup both straight out and diagonally from the edges of the opening.
# 

def main(input_grid):
    """
    Fill the cup interior with yellow and emit yellow rays from its opening.

    Input structure:
    - The grid contains one non-black cup shape on black background.
    - The cup encloses black interior cells and has an opening on one side.
    - The cup may be rotated to face any direction.

    Rule:
    - Find the cup bounding box and the black cells inside it.
    - Fill those interior cells yellow.
    - Draw yellow straight out through the opening and diagonally from the opening endpoints until blocked.

    Key insight:
    - The opening position relative to the cup center determines the outward direction of the fill.
    """
    # Plan:
    # 1. Detect the cup
    # 2. Find the mask of the inside of the cup
    # 3. Find the mask of the opening of the cup (on one of its sides)
    # 4. Fill the cup with yellow
    # 5. Shoot pixels outward from the opening (straight out)
    # 6. Shoot pixels outward from the opening (diagonally out, from the edges)
    
    # 1. Detect cup
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    assert len(objects) == 1, "There should be exactly one cup"
    obj = list(objects)[0]

    output_grid = input_grid.copy()

    # 2. Extract what's inside the cup (as its own object), which is everything in the bounding box that is not the object itself
    _b = bbox(obj != Color.BLACK)
    assert _b is not None
    cup_r, cup_c, _r1, _c1 = _b
    cup_width = _r1 - cup_r + 1
    cup_height = _c1 - cup_c + 1
    inside_cup_mask = np.zeros_like(input_grid, dtype=bool)
    inside_cup_mask[cup_r:cup_r+cup_width, cup_c:cup_c+cup_height] = True
    inside_cup_mask = inside_cup_mask & (obj == Color.BLACK)

    # 3. Extract the hole in the cup, which is what's inside and on the boundary of the bounding box
    # what's inside...
    hole_mask = inside_cup_mask.copy()
    # ...and then we need to remove anything not on the boundary
    hole_mask[cup_r+1 : cup_r+cup_width-1, cup_c+1 : cup_c+cup_height-1] = False

    # 4. Fill the cup with yellow
    output_grid[inside_cup_mask] = Color.YELLOW

    # 5. Shoot pixels outward from the opening (straight out)
    # Find the direction of the opening, which is the unit vector that points from the center of the cup to the hole
    _b = bbox(hole_mask != Color.BLACK)
    assert _b is not None
    hole_r, hole_c = bbox_anchor(_b, anchor='center')
    _b = bbox(obj != Color.BLACK)
    assert _b is not None
    cup_r, cup_c = bbox_anchor(_b, anchor='center')
    direction = (int(np.sign(hole_r - cup_r)), int(np.sign(hole_c - cup_c)))
    # Loop over every boundary pixel and shoot outward
    for r, c in np.argwhere(hole_mask):
        draw_line(output_grid, r, c, direction=direction, color=Color.YELLOW)

    # 6. Shoot pixels outward from the opening (diagonally out, from the edges)
    # Find the two extremal points on the boundary of the hole, which are the points farthest away from each other
    points_on_boundary = np.argwhere(hole_mask)
    pt1, pt2 = max({ ( (x1,y1), (x2,y2) ) for x1,y1 in points_on_boundary for x2,y2 in points_on_boundary },
                   key=lambda pair: np.linalg.norm(np.array(pair[0]) - np.array(pair[1])))
    
    # For each of those points, shoot diagonal lines in all directions, but stop as soon as you hit something that's not black
    for pt in [pt1, pt2]:
        for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw_line(output_grid, pt[0]+direction[0], pt[1]+direction[1], direction=direction, color=Color.YELLOW, stop_at=Color.NOT_BLACK)
    
    return output_grid

def generate_input():
    # Generate the grid with random size
    # Pick a random width/height for this cup
    cup_width = np.random.randint(4, 8)
    cup_height = np.random.randint(3, 8)
    width = np.random.randint(max(8, cup_width + 3), 30)
    height = np.random.randint(max(8, cup_height + 3), 30)
    grid = np.full((width, height), Color.BLACK)

    # Make a sprite, which is just going to be a blue outline of a rectangle with a hole at the top
    sprite = np.full((cup_width, cup_height), Color.BLACK)
    color = np.random.choice([ color for color in Color.NOT_BLACK if color != Color.YELLOW ])
    sprite[0, :] = color
    sprite[-1, :] = color
    sprite[:, 0] = color
    sprite[:, -1] = color

    # Make an opening on one side (not in the corners).
    hole_c = 0
    hole_len = int(np.random.randint(1, max(2, cup_width - 2)))
    hole_r0 = int(np.random.randint(1, cup_width - 1 - hole_len + 1))
    sprite[hole_r0 : hole_r0 + hole_len, hole_c] = Color.BLACK

    # Find a random free location for it
    free_r, free_c = random_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
    paste(grid, sprite, free_r, free_c, transparent=Color.BLACK, clip=True)

    # Do a random rotation, so the cup can face any direction
    grid = np.rot90(grid,-(np.random.randint(0, 4)))

    return grid    
