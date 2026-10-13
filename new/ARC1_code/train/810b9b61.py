import numpy as np

import random
from arc_utils import Color, draw_line, find_components, first_free_location, object_interior, paste

# concepts:
# objects, topology

# description:
# In the input grid, you will see various blue objects. Some are "hollow" and contain a fully-enclosed region, while others do not have a middle that is separate from outside the object, and fully enclosed.
# To create the output grid, copy the input grid. Then, change the color of all "hollow" shapes to be green.

def main(input_grid):
    """
    Recolor hollow blue objects green while preserving non-hollow blue objects.

    Input structure:
    - The input contains multiple blue objects on a black background.
    - Some objects fully enclose one or more black interior cells.
    - Other blue objects are incomplete or open and do not enclose an interior region.

    Rule:
    - Find each blue connected component.
    - Test whether the component has black cells inside its enclosed interior.
    - Recolor hollow components green and paste all other objects unchanged.

    Key insight:
    - The decision is topological: a closed boundary that traps background cells is the feature
      that distinguishes objects to recolor.
    """
    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=4, monochromatic=True, return_format="objects")
    output_grid = input_grid.copy()
    for obj in objects:
        if is_hollow(obj):
            obj = obj.copy()
            obj[obj != bg] = Color.GREEN
        output_grid = paste(output_grid, obj, 0, 0, transparent=bg)

    return output_grid

def is_hollow(obj):
    # to check if it contains a fully enclosed region, find everything that is enclosed by the object (in its interior), but not actually part of the object
    interior_mask = object_interior(obj)
    object_mask = obj != Color.BLACK
    hollow_mask = interior_mask & ~object_mask
    return np.any(hollow_mask)


def generate_input():
    n = np.random.randint(10, 28)
    input_grid = np.full((n, n), Color.BLACK)
    # create a bunch of random objects. all objects are either (1) hollow, in which case they are the border of a rectangle of some size, or (2) not hollow, in which case they are a subset of a border of a rectangle of some size.
    # make sure we place at least one hollow and nonhollow object. then add random objects until somewhat full.

    def random_hollow_object():
        n, m = np.random.randint(3, 7), np.random.randint(3, 7)
        obj = np.full((n, m), Color.BLUE)
        obj[1:n-1, 1:m-1] = Color.BLACK
        return obj

    def random_nonhollow_object():
        obj = random_hollow_object()
        # remove a random number of dots from it
        size = np.count_nonzero(obj)
        new_size = np.random.randint(1, size)
        xs, ys = np.where(obj != Color.BLACK)
        for i in range(size - new_size):
            obj[xs[i], ys[i]] = Color.BLACK

        return obj

    try:
        # add one hollow and one nonhollow object, then add random objects until somewhat full.
        obj = random_hollow_object()
        _loc = first_free_location(input_grid, obj, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(input_grid, obj, r, c, transparent=Color.BLACK, clip=True)

        obj = random_nonhollow_object()
        _loc = first_free_location(input_grid, obj, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(input_grid, obj, r, c, transparent=Color.BLACK, clip=True)
    except ValueError:
        return generate_input()

    while True:
        obj = random_hollow_object() if np.random.rand() < 0.5 else random_nonhollow_object()
        try:
            _loc = first_free_location(input_grid, obj, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
            paste(input_grid, obj, r, c, transparent=Color.BLACK, clip=True)
        except ValueError:
            return input_grid
# ============= remove below this point for prompting =============
