
import numpy as np
from typing import *
import random
from arc_utils import Color, MirrorSymmetry, bbox, bbox_anchor, colors_present, contact, draw_line, find_components, first_free_location, paste, random_sprite

# concepts:
# symmetry, mirror

# description:
# In the input you will see big objects with a primary color and some single pixels of different colors attached to it.
# To make the output, mirror the primary colored part of each object over the differently colored pixels attached to it, changing the primary color to match the other color.

def main(input_grid):
    """
    Mirror each primary object across its attached indicator pixel and recolor the mirror copy.

    Input structure:
    - Larger primary-color objects sit on a black background.
    - One-pixel indicators of other colors touch the primary objects.
    - Indicators may touch sides or corners of the primary object.

    Rule:
    - Pair each indicator with every primary object it touches.
    - Recolor a copy of the primary object to the indicator color.
    - Mirror that copy across the side or corner axis implied by the indicator position.

    Key insight:
    - Attached single pixels provide both the mirror axis and the color for the reflected copy.
    """
    # Plan
    # 1. Parse the input into primary regions and their associated single pixel indicators
    # 2. For each primary region and each attached indicator pixel, change color and mirror.

    # 1. Input parsing
    background = Color.BLACK
    objects = find_components(input_grid, bg=background, connectivity=8, monochromatic=True, return_format="objects")
    indicator_pixels = [ obj for obj in objects if np.sum(obj != background) == 1 ]
    primary_objects = [ obj for obj in objects if np.sum(obj != background) > 1 ]

    # 2. Output generation

    # Copy the input because we draw on top of it
    output_grid = input_grid.copy()

    # loop over primary objects and every pixel that they are in contact with
    for primary_object in primary_objects:
        for indicator_pixel in indicator_pixels:
            if not contact(object1=primary_object, object2=indicator_pixel, background=background, connectivity=8): continue

            # Recolor
            indicator_color = colors_present(indicator_pixel, exclude={ background })[0]
            recolored_object = np.copy(primary_object)
            recolored_object[primary_object != background] = indicator_color

            # Build the mirroring object
            _b = bbox(indicator_pixel != background)
            assert _b is not None
            indicator_r, indicator_c = bbox_anchor(_b, anchor="upper left")
            _b = bbox(primary_object != background)
            assert _b is not None
            primary_r1, primary_c1 = bbox_anchor(_b, anchor="upper left")
            _b = bbox(primary_object != background)
            assert _b is not None
            primary_x2, primary_y2 = bbox_anchor(_b, anchor="lower right")
            # If it's in the corners, we mirror diagonally (over both x and y)
            # If it's on the left/right side, we mirror horizontally
            # If it's on the top/bottom side, we mirror vertically
            mirror_r, mirror_c = None, None
            if indicator_r == primary_r1-1: mirror_r = primary_r1-0.5
            if indicator_r == primary_x2+1: mirror_r = primary_x2+0.5
            if indicator_c == primary_c1-1: mirror_c = primary_c1-0.5
            if indicator_c == primary_y2+1: mirror_c = primary_y2+0.5
            symmetry = MirrorSymmetry(mirror_r=mirror_r, mirror_c=mirror_c)

            # Mirror the primary object over the indicator pixel
            for r, c in np.argwhere(primary_object != background):
                x2, y2 = symmetry.apply(r, c)
                if 0 <= x2 < output_grid.shape[0] and 0 <= y2 < output_grid.shape[1]:
                    output_grid[x2, y2] = recolored_object[r, c]

    return output_grid

def generate_input():
    # Create monochromatic objects and then attach differently colored indicator pixels at one of their corners
    width, height = np.random.randint(15, 30), np.random.randint(15, 30)
    grid = np.zeros((width, height), dtype=int)

    num_objects = np.random.randint(1, 4)
    for _ in range(num_objects):
        # we'll attach indicators to the upper left and then randomly rotate the sprite before putting it on the canvas
        # at most three indicators, so we'll choose a primary color and then choose the indicator colors from the rest5
        max_indicators = 3
        num_indicators = np.random.randint(1, max_indicators+1)
        primary_color, *indicator_colors = np.random.choice(Color.NOT_BLACK, size=num_indicators+1, replace=False)

        # create sprite with indicators attached to it in the upper left. it must be contiguous, so we loop until we find one that is
        while True:
            sprite = random_sprite([3,4,5,6], [3,4,5,6], color_palette=[primary_color], symmetry="not_symmetric")
            # Clear place for indicators in the upper left
            sprite[:1, :] = Color.BLACK
            sprite[:, :1] = Color.BLACK
            # three indicator locations in the upper left corner
            indicator_locations = [(0, 0), (0, 1), (1, 0)]
            random.shuffle(indicator_locations)
            for color, (r, c) in zip(indicator_colors, indicator_locations):
                sprite[r, c] = color
            # make sure that the upper left corner is connected to the rest of the sprite
            if sprite[1, 1] == primary_color: break
        
        # randomly rotate to get variety of orientations
        sprite = np.rot90(sprite, k=-(np.random.randint(4)))
        
        # randomly place the sprite on the grid
        _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=max(sprite.shape), padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid

            
