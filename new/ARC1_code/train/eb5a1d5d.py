import numpy as np
import random
from arc_utils import Color, bbox, colors_present, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite


# concepts:
# downscaling, nesting

# description:
# In the input you will see a grid consisting of nested shapes of different colors.
# To make the output, make a grid with one pixel for each layer of the shapes, 
# with the color from outermost layer to the innermost layer in the same order they appear in the input.

def main(input_grid):
    """
    Compress nested colored shapes into a small nested-square color summary.

    Input structure:
    - The input contains nested monochromatic non-black shapes on a black background.
    - Outer layers have larger bounding boxes than inner layers.
    - Each layer has a distinct color.

    Rule:
    - Detect the colored objects and sort them from largest bounding box to smallest.
    - Create a square output with one ring per detected layer.
    - Fill nested squares from outermost to innermost using the corresponding layer colors.

    Key insight:
    - The output preserves only nesting order and colors; geometric details of the
      original shapes are reduced to one-pixel-thick square layers.
    """
    # Plan:
    # 1. Parse the input into objects and order them from outermost to innermost by area
    # 2. Draw nested rectangles with the colors of the input objects, each layer has only one pixel length

    # 1. input parsing
    # Find the objects in the input grid
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    # Sort the objects from outermost to innermost, using area to determine the order
    def _area(o: np.ndarray) -> int:
        b = bbox(o != Color.BLACK)
        if b is None:
            return 0
        r0, c0, r1, c1 = b
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    objects.sort(key=_area, reverse=True)

    # 2. drawing the output
    # Leave only one layer of each color shape
    grid_len = len(objects) * 2 - 1
    output_grid = np.full((grid_len, grid_len), Color.BLACK)

    # Calculate each layer's length, which starts at the outermost layer
    current_len = grid_len

    # Draw nested shapes with the colors of the input objects. Each layer has only one pixel length of the current color
    for i, obj in enumerate(objects):
        # Get the color of the current layer
        cols = set(map(int, np.unique(obj))) - {Color.BLACK}
        assert cols
        color = next(iter(cols))
        # Fill the region with the current color
        cur_shape = np.full((current_len, current_len), color)
        # Place the current shape in the output grid
        # Make sure each layer has only one pixel length of the current color
        output_grid = paste(output_grid, cur_shape, i, i)
        current_len -= 2

    return output_grid

def generate_input():
    # Get the number of colors
    n_colors = random.randint(2, 6)
    colors = np.random.choice(Color.NOT_BLACK, n_colors, replace=False)

    # Get the innermost shape dimensions
    cur_width, cur_height = np.random.randint(3, 6), np.random.randint(3, 6)
    
    # Draw a nested shape with the colors
    previous_shape = None
    for color in colors:
        # Fill the current shape with the current color
        cur_shape = np.full((cur_width, cur_height), color)
        if previous_shape is not None:
            # Place the previous shape in a random location in the current shape
            _loc = first_free_location(cur_shape, previous_shape, transparent=color, border=1, padding=0, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
            paste(cur_shape, previous_shape, r, c, transparent=color, clip=True)

        # Update the previous shape
        previous_shape = cur_shape
        
        # Update the dimensions of the next shape, should be larger than the previous shape
        next_layer_width, next_layer_height = np.random.randint(cur_width + 2, cur_width + 6), np.random.randint(cur_height + 2, cur_height + 6)
        cur_width, cur_height = next_layer_width, next_layer_height
    
    # The final shape is the last shape with all inner shapes inside each other
    grid = cur_shape
    return grid
# ============= remove below this point for prompting =============
