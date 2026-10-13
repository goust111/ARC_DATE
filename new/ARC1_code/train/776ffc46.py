import numpy as np

import random
from arc_utils import Color, bbox, colors_present, crop_bbox, draw_line, find_components, first_free_location, object_interior, paste, random_sprite

# concepts:
# same/different, color change, containment

# description:
# In the input you will see some monochromatic objects. Some of them will be contained by a grey box, and some of them will not.
# To make the output, take each shape contained inside a grey box, and find any other shapes with the same shape (but a different color), and change their color to match the color of the shape inside the grey box.

def main(input_grid):
    """
    Recolor unboxed shape copies to match the same-shaped object enclosed by a gray box.

    Input structure:
    - The input is a black grid with gray outline boxes and colored sprites.
    - Some colored sprites are contained inside gray boxes.
    - Other sprites outside the boxes may share the same shape in different colors.

    Rule:
    - Separate gray box objects from non-gray sprite objects.
    - Identify sprites contained inside gray interiors.
    - Recolor outside sprites with matching shape masks to the contained sprite's color.

    Key insight:
    - Gray boxes mark reference examples whose colors transfer to same-shaped free copies.
    """
    # Plan:
    # 1. Extract the objects and separate them according to if they are grey or not
    # 2. Determine if each non-grey shape is contained by a grey shape
    # 3. Check to see which objects (among grey contained shapes) have another object which has the same shape (not contained by grey)
    # 4. Do the color change when you find these matching objects

    # 1. Extract the objects, separating them by if they are grey or not
    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=8, monochromatic=True, return_format="objects")
    grey_objects = [obj for obj in objects if Color.GREY in colors_present(obj, exclude={bg})]
    non_grey_objects = [obj for obj in objects if Color.GREY not in colors_present(obj, exclude={bg})]

    # 2. Determine if each non-grey shape is contained by a grey shape
    # Divide the non-grey objects into two groups: those contained by grey, and those not contained by grey
    # Make a helper function for checking of one object is contained by another
    def object_contains_another_object(inside_object, outside_object):
        # Using bounding boxes:
        # inside_x, inside_y, inside_width, inside_height = bounding_box(inside_object)
        # outside_x, outside_y, outside_width, outside_height = bounding_box(outside_object)
        # return inside_x >= outside_x and inside_y >= outside_y and inside_x + inside_width <= outside_x + outside_width and inside_y + inside_height <= outside_y + outside_height
        # Using topology+masks:
        inside_object_mask = inside_object != bg
        outside_interior_mask = object_interior(outside_object, background=bg)
        return np.all(outside_interior_mask >= inside_object_mask)
    objects_contained_by_grey = [ non_grey for non_grey in non_grey_objects
                                  if any( object_contains_another_object(non_grey, grey) for grey in grey_objects ) ]
    objects_not_contained_by_gray = [ non_grey for non_grey in non_grey_objects
                                     if not any( object_contains_another_object(non_grey, grey) for grey in grey_objects ) ]
    
    # 3. Check to see which objects (among grey contained shapes) have another object which has the same shape (not contained by grey)
    output_grid = input_grid.copy()

    # Helper function to check if two objects have the same shape
    def objects_have_same_shape(obj1, obj2):
        b1 = bbox(obj1 != bg)
        b2 = bbox(obj2 != bg)
        if b1 is None or b2 is None:
            return False
        mask1 = crop_bbox(obj1, b1) != bg
        mask2 = crop_bbox(obj2, b2) != bg
        return np.array_equal(mask1, mask2)
    
    for grey_contained in objects_contained_by_grey:
        for non_grey in objects_not_contained_by_gray:
            if objects_have_same_shape(grey_contained, non_grey):
                # 4. Do the color change
                target_color = colors_present(grey_contained, exclude={bg})[0]
                non_grey_mask = non_grey != bg
                output_grid[non_grey_mask] = target_color

    return output_grid

def generate_input():
    # Make some sprites
    # One of those sprites is going to be contained by a grey box, and occur more than once (but in different colors)
    # The other sprites are not going to be contained by the grey box, and might occur multiple times anyway

    width, height = random.choice(range(17, 30+1)), random.choice(range(17, 30+1))
    grid = np.full((width, height), Color.BLACK)

    possible_sprite_dimensions = [1,2,3,4]
    contained_color = random.choice([ color for color in Color.NOT_BLACK if color != Color.GREY ])
    contained_sprite = random_sprite(possible_sprite_dimensions, possible_sprite_dimensions,
                                     color_palette=[contained_color])

    # put down the contained sprite at a random spot, and then put a gray box around it
    _loc = first_free_location(grid, contained_sprite, transparent=Color.BLACK, border=2, padding=2, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, contained_sprite, r, c, transparent=Color.BLACK, clip=True)
    contained_w, contained_h = contained_sprite.shape

    # make the gray box (outline of a rectangle)
    grey_box = np.full((contained_w+4, contained_h+4), Color.GREY)
    grey_box[1:-1, 1:-1] = Color.BLACK
    paste(grid, grey_box, r-2, c-2, transparent=Color.BLACK, clip=True)

    # put down some other sprites which are the same shape as the contained sprite, but in different colors
    n_duplicate_sprites = np.random.randint(1, 3)
    for _ in range(n_duplicate_sprites):
        duplicate_sprite = contained_sprite.copy()
        duplicate_sprite[duplicate_sprite != Color.BLACK] = random.choice([ color for color in Color.NOT_BLACK if color != Color.GREY and color != contained_color ])
        _loc = first_free_location(grid, duplicate_sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, duplicate_sprite, r, c, transparent=Color.BLACK, clip=True)
    
    # put down some other sprites which are not the same shape as the contained sprite, which might be duplicated occasionally
    n_non_contained_sprites = np.random.randint(1, 3)
    for _ in range(n_non_contained_sprites):
        sprite_color = random.choice([ color for color in Color.NOT_BLACK if color != Color.GREY ])
        sprite = random_sprite(possible_sprite_dimensions, possible_sprite_dimensions,
                               color_palette=[sprite_color])
        _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

        # Maybe duplicate?
        if np.random.rand() < 0.5:
            _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
            paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return grid
# ============= remove below this point for prompting =============
