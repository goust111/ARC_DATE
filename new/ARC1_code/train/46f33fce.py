import numpy as np
import random
from arc_utils import Color, bbox, bbox_anchor, crop_bbox, detect_objects, draw_line, paste, scale_sprite

# concepts:
# scaling

# description:
# In the input you will see a grid with different colors of pixels scattered on the grid
# To make the output grid, you should first only scale the pixels by 2 times,
# then scale in whole grid 2 times.

def main(input_grid):
    """
    Expand each isolated colored pixel into a 2 by 2 block, then scale the entire grid by 2.

    Input structure:
    - The grid is black except for isolated non-black single pixels.
    - Colored pixels are placed with enough spacing for local expansion.
    - Multiple colors may appear independently.

    Rule:
    - Detect every non-black 1 by 1 object.
    - Replace each pixel with a 2 by 2 block anchored one cell up and left from the original.
    - Scale the whole resulting grid by 2.

    Key insight:
    - The transformation is two-stage: first local pixel expansion, then a global doubling
      of the entire canvas.
    """
    # Plan:
    # 1. Detect all the colored pixels
    # 2. Rescale each such sprite
    # 3. Blit the rescaled sprite onto the output grid, taking care to anchor it correctly
    # 4. Rescale the output grid (2x)

    # Detect all the colored pixels in the input grid
    pixel_objects = detect_objects(grid=input_grid, colors=list(Color.NOT_BLACK),
                            # These are single pixels, so they are 1x1
                            allowed_dimensions=[(1, 1)],
                            monochromatic=True, connectivity=4, background=Color.BLACK)

    # Initialize the output grid with the same size as the input grid
    output_grid = np.copy(input_grid)

    scale_factor = 2
    for obj in pixel_objects:
        # Get the position of each colored pixel, and crop it to produce a sprite
        b = bbox(obj != Color.BLACK)
        assert b is not None
        r0, c0, _r1, _c1 = b
        single_pixel_sprite = crop_bbox(obj, b)

        # Scale the sprite by `scale_factor` times
        scaled_sprite = scale_sprite(single_pixel_sprite, scale_factor)

        # The coordinate of the scaled pattern (anchored at the upper left)
        new_r0, new_c0 = r0 - scale_factor + 1, c0 - scale_factor + 1

        # Put the scaled pattern on the output grid
        output_grid = paste(output_grid, scaled_sprite, new_r0, new_c0, transparent=Color.BLACK, clip=True)
    
    # Scale the whole grid by scale_factor times
    output_grid = scale_sprite(output_grid, scale_factor)

    return output_grid

def generate_input():
    # Generate the background grid with size of n x n.
    grid_size = 10
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # randomly scatter the pixels on the grid with only every scaling_factor coordinates
    # Leave enough space for scaling
    density = 0.4
    scaling_factor = 2
    colors = Color.NOT_BLACK
    for r in range(1, grid_size, scaling_factor):
        for c in range(1, grid_size, scaling_factor):
            # Randomly scatter the pixels on the grid
            if np.random.rand() < density:
                grid[r, c] = np.random.choice(colors)

    return grid
# ============= remove below this point for prompting =============
