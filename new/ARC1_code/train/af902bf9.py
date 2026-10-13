import numpy as np

import random
from arc_utils import Color, draw_line, first_free_location, paste

# concepts:
# filling, surrounding

# description:
# In the input you will see several yellow pixels arranged in groups of 4 so that each group outlines a rectangular shape.
# To make the output, fill the corresponding inner rectangular regions with red.

def main(input_grid):
    """
    Fill red rectangles whose corners are marked by groups of four yellow pixels.

    Input structure:
    - Yellow pixels mark outside corners of one or more hidden rectangles.
    - Rectangle interiors are black in the input.
    - The four yellow markers share two rows and two columns.

    Rule:
    - Search yellow pairs that can form opposite rectangle corners.
    - Confirm the other two yellow corners by row and column alignment.
    - Fill the open interior of each such rectangle red.

    Key insight:
    - Yellow pixels are geometric corner constraints; the red output is the implied interior.
    """
    # Detect the rectangular regions by finding groups of four surrounding yellow pixels
    surrounding_color = Color.YELLOW
    rectangle_color = Color.RED
    
    output_grid = np.copy(input_grid) 

    # loop over all the yellows...
    for r, c in np.argwhere(input_grid == surrounding_color):
        # ...and find the other matching yellows forming a rectangle: (x, y), (x, y'), (x', y), (x', y')
        for other_r, other_c in np.argwhere(input_grid == surrounding_color):
            if input_grid[r, other_c] == surrounding_color and input_grid[other_r, c] == surrounding_color:
                # fill the rectangle with red
                output_grid[r+1:other_r, c+1:other_c] = rectangle_color

    return output_grid

def generate_input():
    # Create the background grid
    n, m = np.random.randint(10, 15), np.random.randint(10, 15)
    grid = np.zeros((m, n), dtype=int)

    rectangle_num = np.random.randint(1, 4)

    # Draw rectangles on the grid
    rectangle_color = Color.RED
    surrounding_color = Color.YELLOW

    for _ in range(rectangle_num):
        rectangle_len = np.random.randint(1, 5)
        rectangle_width = np.random.randint(1, 5)
        
        # Draw the rectangle with rectangle color
        rectangle_sprite = np.full((rectangle_len, rectangle_width), rectangle_color)
        try:
            _loc = first_free_location(grid, rectangle_sprite, transparent=Color.BLACK, border=1, padding=2, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except:
            continue

        paste(grid, rectangle_sprite, r, c, transparent=Color.BLACK, clip=True)

        # Place four surrounding colors around the rectangle right outside its corners
        min_r, min_c, max_r, max_c = r, c, r+rectangle_len, c+rectangle_width
        grid[min_r-1, min_c-1] = surrounding_color
        grid[min_r-1, max_c] = surrounding_color
        grid[max_r, c-1] = surrounding_color
        grid[max_r, max_c] = surrounding_color

    # Remove the inner rectangle color, only keep the surrounding color
    grid[grid == rectangle_color] = Color.BLACK
    return grid
# ============= remove below this point for prompting =============
