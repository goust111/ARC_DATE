
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste

# concepts:
# Expanding, Framing, Growing

# description:
# In the input you will see some squares of different sizes and colors.
# To make the output, you need to:
# 1. Expand the squares down to the bottom of the grid using the color BLUE.
# 2. Draw a green frame around the squares. The green square should be twice as long as the original square.
# 3. Put the original square back to the center of the green square.

def main(input_grid):
    """
    Add blue downward extensions and larger green square frames around each input square.

    Input structure:
    - The grid contains multiple monochromatic square objects on a black background.
    - The square objects may have different sizes but share the same generated color.
    - Blue and green are output-only construction colors.

    Rule:
    - Detect each square object and clear the output background to black.
    - Fill blue below each square's original footprint down to the grid bottom.
    - Draw a green frame twice the square size around it, then paste the original square back.

    Key insight:
    - Each square contributes two independent structures: a vertical blue projection from
      its base and a centered doubled-size green frame around the original square.
    """
    # Plan:
    # 1. Extract the square objects from the input grid
    # 2. Expand the squares down to the bottom using the color BLUE
    # 3. Draw the frame
    # 4. Put the original squares back

    # 1. Input parsing and setup
    # Extract the squares in the input grid
    square_objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    # Note the frame color and the expansion down color
    frame_color = Color.GREEN
    expand_color = Color.BLUE

    # The output grid is the same size as the input grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # 2. Expand the squares down to the bottom using `expand_color`.
    for square_obj in square_objects:
        _b = bbox(square_obj != Color.BLACK)
        assert _b is not None
        r0, c0, r1, c1 = map(int, _b)
        # Fill below the square's footprint.
        if r1 + 1 < output_grid.shape[0]:
            output_grid[r1 + 1 :, c0 : c1 + 1] = int(expand_color)
    
    # 3. Draw a green square frame around the original squares, the green square should be twice as big as original were
    for square_obj in square_objects:
        # The square can be partly outside the canvas
        # This math is to get the (x,y) of the top-left corner of the square, even if it's outside the canvas
        _b = bbox(square_obj != Color.BLACK)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1
        square_len = max(w, h)
        r -= (square_len - w)
        c -= (square_len - h)
        # Make and draw the frame
        frame_len = square_len * 2
        green_frame = np.full((frame_len, frame_len), frame_color)
        paste(output_grid, green_frame, r - square_len // 2, c - square_len // 2, transparent=Color.BLACK, clip=True)
    
    # 4. Put the original square back to the center of the green square
    for square_obj in square_objects:
        _b = bbox(square_obj != Color.BLACK)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1
        _b = bbox(square_obj != Color.BLACK)
        assert _b is not None
        square_obj = crop_bbox(square_obj, _b)
        paste(output_grid, square_obj, r, c, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Generate the background grid
    width = np.random.randint(15, 30)
    height = width
    grid = np.full((width, height), Color.BLACK)

    # Randomly select the number of squares
    square_num = np.random.randint(2, 5)
    square_color = Color.MAROON

    for _ in range(square_num):
        # Randomly select the size of the squares
        square_size = np.random.randint(1, 4)
        # Ensure the size of square is even
        square_size = square_size * 2

        square = np.full((square_size, square_size), square_color)
        try:
            # Place the square on the grid
            _loc = first_free_location(grid, square, transparent=Color.BLACK, border=1, padding=2, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except:
            # If there is no space, regenerate the input
            return generate_input()
        paste(grid, square, r, c, transparent=Color.BLACK, clip=True)

    return grid
