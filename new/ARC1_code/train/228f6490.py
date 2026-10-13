
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, object_interior, paste, random_free_location, random_sprite

# concepts:
# holes, objects, distracters, topology, puzzle piece

# description:
# In the input you will see multiple grey rectangular objects, each with a black hole inside of it. There are also monochromatic objects somewhere exactly the same shape as each hole, and random other distracter objects (distracters all the same color). 
# To make the output, check to see if each non-distracter object perfectly fits inside the black hole inside of a gray object, like it is a puzzle piece. If it does, place it inside the black hole. If it doesn't, leave it where it is.

def main(input_grid):
    """
    Move each non-distractor puzzle piece into the matching black hole inside a gray object.

    Input structure:
    - Gray rectangular objects contain black interior holes.
    - Colored objects outside the rectangles may match those hole shapes.
    - Distractor objects share the most common non-gray object color.

    Rule:
    - Extract gray objects and their black interior masks.
    - Treat the most common non-gray object color as the distractor color.
    - For every other colored sprite, compare its mask to each hole mask and paste exact matches into the hole.

    Key insight:
    - Fit is determined by binary shape equality, not by the piece color or its original position.
    """
    # Plan:
    # 1. Parse the input into objects, sprites, and black holes inside the grey objects
    # 2. Identify color of distracter objects.
    # 3. Turn each object into a sprite
    # 4. Check if each sprite can be moved into a black hole, and if so, move it there
    
    # Parse, separating greys from other objects
    grey_input = input_grid.copy()
    grey_input[input_grid != Color.GREY] = Color.BLACK
    grey_objects = find_components(grey_input, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    # extracting a mask for the black region is tricky, because black is also the color of the background
    # get the black region inside the object by getting the interior mask, then just the black pixels
    interior_black_regions = [ object_interior(obj, background=Color.BLACK) & (obj == Color.BLACK)
                               for obj in grey_objects ]

    not_grey_input = input_grid.copy()
    not_grey_input[input_grid == Color.GREY] = Color.BLACK
    not_grey_objects = find_components(not_grey_input, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    # Get the sprites
    not_grey_sprites = []
    for obj in not_grey_objects:
        _b = bbox(obj != Color.BLACK)
        if _b is None:
            continue
        not_grey_sprites.append(crop_bbox(obj, _b))

    # Get the color of the distracter objects
    # This is the most common color of the non-grey objects
    color_of_each_object = [ np.unique(obj[obj!=Color.BLACK])[0] for obj in not_grey_objects ]
    distracter_color = max(set(color_of_each_object), key=color_of_each_object.count)
    
    # Check if each sprite perfectly fits in a black hole/black interior region
    # do this by checking if it has the same shape as a black interior region
    # if it does, place it there
    output_grid = np.copy(input_grid)
    for sprite, obj, color in zip(not_grey_sprites, not_grey_objects, color_of_each_object):
        # Try to find a perfect fit (if it is not the distracter color)
        if color == distracter_color:
            continue
        
        for interior_obj_mask in interior_black_regions:
            # check the sprite masks are the same, meaning that they have the same shape
            # to convert a sprite to a mask you check if it is not background (black)
            sprite_mask = sprite != Color.BLACK
            # to convert an object to a spright you crop it
            _b = bbox(interior_obj_mask)
            if _b is None:
                continue
            interior_sprite_mask = crop_bbox(interior_obj_mask, _b)
            perfect_fit = np.array_equal(sprite_mask, interior_sprite_mask)

            if perfect_fit:
                # remove the object from its original location
                object_mask = obj != Color.BLACK
                output_grid[object_mask] = Color.BLACK

                # place the sprite in the black hole by blitting it
                _b = bbox(interior_obj_mask)
                assert _b is not None
                interior_r, interior_c, _r1, _c1 = _b
                interior_width = _r1 - interior_r + 1
                interior_height = _c1 - interior_c + 1
                paste(output_grid, sprite, interior_r, interior_c, transparent=Color.BLACK, clip=True)
                break

    return output_grid            
    
    


def generate_input():
    n, m = np.random.randint(20, 30, size=2)
    input_grid = np.full((m, n), Color.BLACK)

    distracter_color = np.random.choice(Color.NOT_BLACK)

    n_grey_objects = np.random.randint(1, 3)
    for k in range(n_grey_objects):
        grey_width, grey_height = np.random.randint(5, 8, size=2)
        grey_sprite = np.full((grey_width, grey_height), Color.GREY)

        # make a black hole in the grey object
        # the black hole should be a sprite whole primary color is black, and whose background color is grey
        # it should be smaller than the grey object
        hole_width, hole_height = np.random.randint(1, grey_width-2), np.random.randint(1, grey_height-2)
        hole_sprite = random_sprite(hole_width, hole_height, color_palette=[Color.BLACK], background=Color.GREY, symmetry="not_symmetric")
        try:
            hole_r, hole_c = random_free_location(grey_sprite, hole_sprite, transparent=Color.GREY, border=1, padding=0, padding_connectivity=8)
        except ValueError:
            break
        paste(grey_sprite, hole_sprite, hole_r, hole_c, transparent=Color.GREY, clip=True)

        # place the grey object in the input grid
        try:
            r, cc = random_free_location(input_grid, grey_sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(input_grid, grey_sprite, r, cc, transparent=Color.BLACK, clip=True)

        # each hole has a corresponding colored object somewhere on the grid waiting for it with the same shape as the whole
        # this is going to have the perfect fit, so it needs to have the same mask as the hole but a different color (but not the special colors of gray or the distracter)
        color = np.random.choice([c for c in Color.NOT_BLACK if c != Color.GREY and c != distracter_color])
        object_sprite = np.full((hole_width, hole_height), Color.BLACK)
        object_sprite[hole_sprite == Color.BLACK] = color

        # place the object sprite in the input grid
        try:
            r, cc = random_free_location(input_grid, object_sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(input_grid, object_sprite, r, cc, transparent=Color.BLACK, clip=True)

    # place small distracter objects
    n_distracter_objects = np.random.randint(5, 10)
    for k in range(n_distracter_objects):
        distracter_width, distracter_height = np.random.randint(1, 3, size=2)
        distracter_sprite = random_sprite(distracter_width, distracter_height, color_palette=[distracter_color], background=Color.BLACK)
        try:
            r, cc = random_free_location(input_grid, distracter_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(input_grid, distracter_sprite, r, cc, transparent=Color.BLACK, clip=True)

    return input_grid
