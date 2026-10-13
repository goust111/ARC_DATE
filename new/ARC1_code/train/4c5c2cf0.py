
import numpy as np
from typing import *

import random
from arc_utils import Color, detect_mirror_symmetry, draw_line, find_components, orbit, paste, random_free_location, random_sprite

# concepts:
# symmetry, reflection

# description:
# In the input you will see a small monochromatic pattern with reflectional symmetry (horizontal and vertical), as well as another monochromatic object in one of its quadrants.
# To make the output, reflect the other object across the axes of reflectional symmetry of the small pattern.

def main(input_grid):
    """
    Reflect the non-symmetric object across the horizontal and vertical symmetry axes of the separate symmetric marker.

    Input structure:
    - The grid contains exactly two monochromatic non-black objects.
    - One object has both horizontal and vertical mirror symmetry.
    - The other object sits in one quadrant relative to those symmetry axes.

    Rule:
    - Detect both objects and identify the one with both mirror axes.
    - Use those axes as symmetries for the whole output grid.
    - For every pixel of the other object, paint all reflected orbit positions with its color.

    Key insight:
    - The symmetric object defines the reflection axes; the non-symmetric object supplies
      the pixels to replicate.
    """
    # Plan:
    # 1. Detect the pair of objects and divide them by whether or not they are already symmetric
    # 2. Reflect the non-symmetric object across the axes of symmetry of the symmetric object

    # 1. Object detection and setup
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")
    assert len(objects) == 2

    # Find the object that is symmetric.
    #
    # Important: the *other* object can sometimes have >=2 detected mirror symmetries
    # (e.g. if it happens to be symmetric about one axis), so we specifically look for
    # an object that has BOTH a horizontal-only and a vertical-only mirror symmetry.
    best_obj = None
    best_syms = None
    best_score = (-1, -1)  # (has_both_axes, symmetry_count)
    for obj in objects:
        syms = detect_mirror_symmetry(obj, ignore_colors=[], background=Color.BLACK)
        has_r = any(getattr(s, "mirror_r", None) is not None and getattr(s, "mirror_c", None) is None for s in syms)
        has_c = any(getattr(s, "mirror_r", None) is None and getattr(s, "mirror_c", None) is not None for s in syms)
        score = (1 if (has_r and has_c) else 0, len(syms))
        if score > best_score:
            best_score = score
            best_obj = obj
            best_syms = syms

    assert best_obj is not None and best_syms is not None, "There should be a symmetric object"
    symmetric_object = best_obj
    symmetries = best_syms

    # Find the object that is not symmetric
    non_symmetric_object = next(obj for obj in objects if obj is not symmetric_object)

    # 2. Reflect the non-symmetric object across the axes of symmetry of the symmetric object
    output_grid = input_grid.copy()

    for r, c in np.argwhere(non_symmetric_object != Color.BLACK):
        original_color = non_symmetric_object[r, c]
        for transformed_r, transformed_c in orbit(output_grid, r, c, symmetries=symmetries):
            output_grid[transformed_r, transformed_c] = original_color
    
    return output_grid
        



def generate_input():
    
    # Make the canvas large enough that placing the symmetric object with the required border is always feasible.
    grid = np.full((random.choice(range(25, 30 + 1)), random.choice(range(25, 30 + 1))), Color.BLACK)

    # We are going to generate a grid with a symmetric pattern and a non-symmetric object in one of its quadrants
    
    # Generate the symmetric pattern
    symmetric_sprite = random_sprite(range(3, 6), range(3, 6),
                                     color_palette=[random.choice(Color.NOT_BLACK)], connectivity=8, 
                                     symmetry="mirror")    

    # Generate the non-symmetric object
    non_symmetric_sprite = random_sprite(range(7,10), range(7,10),
                                         color_palette=[random.choice(Color.NOT_BLACK)], connectivity=8, 
                                         symmetry="not_symmetric")
    
    # Place the symmetric object so there's guaranteed space for the other object in the lower-right quadrant.
    H, W = grid.shape
    sym_h, sym_w = symmetric_sprite.shape
    ns_h, ns_w = non_symmetric_sprite.shape
    gap = 1
    r_symmetric = random.randint(0, H - sym_h - gap - ns_h)
    c_symmetric = random.randint(0, W - sym_w - gap - ns_w)
    paste(grid, symmetric_sprite, r_symmetric, c_symmetric, transparent=Color.BLACK, clip=False)

    # Put the other sprite in the lower right quadrant, and then randomly rotate at the end to get a variety of possible placements
    r_non_symmetric = r_symmetric + symmetric_sprite.shape[0] + 1
    c_non_symmetric = c_symmetric + symmetric_sprite.shape[1] + 1
    paste(grid, non_symmetric_sprite, r_non_symmetric, c_non_symmetric, transparent=Color.BLACK, clip=False)

    # random rotation
    grid = np.rot90(grid,-(np.random.randint(0, 4)))

    return grid
