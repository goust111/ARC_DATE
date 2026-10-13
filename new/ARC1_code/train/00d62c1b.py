import numpy as np
import random
from arc_utils import Color, draw_line, first_free_location, object_boundary, object_interior, paste, random_sprite


# concepts:
# topology

# description:
# The input grid is a square grid with black and green pixels. The input grid should have regions that are enclosed by the green pixels. 
# To produce the output, you need to find the enclosed regions in the input grid, and then color them yellow. 
                
def main(input_grid):
    """
    Fill black regions enclosed by green boundaries with yellow while preserving the original boundary.

    Input structure:
    - The input is a square black grid containing one or more green closed shapes.
    - Green pixels form the visible boundaries of enclosed regions.
    - Black pixels may appear both inside and outside those boundaries.

    Rule:
    - Detect the interior of the green objects.
    - Exclude the green boundary pixels themselves.
    - Recolor enclosed black interior pixels yellow and leave all other pixels unchanged.

    Key insight:
    - The transformation depends on topology rather than shape identity: only black cells
      trapped inside a green boundary are changed.
    """
    # Create initial output grid template based on input grid.
    output_grid = input_grid.copy()

    # Find enclosed regions
    interior_mask = object_interior(input_grid)
    boundary_mask = object_boundary(input_grid)
    inside_but_not_on_edge = interior_mask & ~boundary_mask

    # Color enclosed regions
    for r, c in np.argwhere(inside_but_not_on_edge):
        if output_grid[r, c] == Color.BLACK:
            output_grid[r, c] = Color.YELLOW

    return output_grid


def generate_input():
    # Generate a square grid of arbitrary size with black background, size from 5x5 to 20x20
    n = random.randint(10, 20)
    grid = np.zeros((n, n), dtype=int)

    # Generate some random green sprites, and then hollow out the interior
    n_objects = random.randint(1, 3)
    for _ in range(n_objects):
        n, m = random.randint(4, 10), random.randint(4, 10)
        sprite = random_sprite(n, m, color_palette=[Color.GREEN], connectivity=8)
        interior_mask = object_interior(sprite)
        boundary_mask = object_boundary(sprite)
        interior_but_not_edges = interior_mask & ~boundary_mask
        sprite[interior_but_not_edges] = Color.BLACK

        try:
            _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except:
            continue

        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return grid 
# ============= remove below this point for prompting =============
