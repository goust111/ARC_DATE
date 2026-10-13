import numpy as np

import random
from arc_utils import Color, contact, draw_line, first_free_location, paste, random_sprite

# concepts:
# counting

# description:
# In the input you will see multiple 2x2 blue squares, multiple 2x2 red squares, and some red or blue dots sprinkled about.
# To make the output, fill a 1x5 grid with blue pixels from left to right for each 2x2 blue square in the input (counting the number of blue 2x2 squares).

def main(input_grid):
    """
    Count the blue 2 by 2 squares and encode that count as a blue bar in a 1 by 5 grid.

    Input structure:
    - The input contains separated 2 by 2 blue squares and 2 by 2 red squares.
    - Single red or blue pixels may appear as distractors.
    - The number of blue squares fits in five output cells.

    Rule:
    - Scan every 2 by 2 window of the input.
    - Count windows filled entirely with blue.
    - Fill that many cells from the left of a 1 by 5 black output with blue.

    Key insight:
    - Only solid 2 by 2 blue blocks count; color-matched singleton noise and red blocks are ignored.
    """
    # make a counter to count the number of blue squares
    blue_square_count = 0

    # scan the grid for blue squares and count them up
    for r in range(input_grid.shape[0]-1):
        for c in range(input_grid.shape[1]-1):
            if input_grid[r,c] == input_grid[r+1,c] == input_grid[r,c+1] == input_grid[r+1,c+1] == Color.BLUE:
                blue_square_count += 1
    
    # make a 1x5 output grid
    output_grid = np.zeros((1, 5), dtype=int)

    # add the number of blue squares to the array from left to right with each pixel representing one blue block
    output_grid[0, :blue_square_count] = Color.BLUE

    return output_grid

def generate_input():
    # make 9x9 black background grid first
    n = m = 9
    grid = np.zeros((n,m), dtype=int)

    # make a random number of blue square sprites and place them at random places on the grid but don't have them touch
    blue_square = np.full((2,2), Color.BLUE, dtype=int)
    for _ in range(np.random.randint(2,6)):
        _loc = first_free_location(grid, blue_square, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        if not contact(object1=grid, object2=blue_square, r2=r, c2=c): # only place new squares that won't touch old ones
          paste(grid, blue_square, r, c, transparent=Color.BLACK, clip=True)
    
    # make a random number of red square sprites and place them at random places on the grid but don't have them touch
    red_square = np.full((2,2), Color.RED, dtype=int)
    for _ in range(np.random.randint(2,6)):
        _loc = first_free_location(grid, red_square, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        if not contact(object1=grid, object2=red_square, r2=r, c2=c): # only place new squares that won't touch old ones
          paste(grid, red_square, r, c, transparent=Color.BLACK, clip=True)
        
    # make a random number of blue pixel sprites and place them at random places on the grid
    blue_pixel = random_sprite(1, 1, density=1, color_palette=[Color.BLUE])
    for _ in range(np.random.randint(4)):
        _loc = first_free_location(grid, blue_pixel, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, blue_pixel, r, c, transparent=Color.BLACK, clip=True)

    # make a random number of red pixel sprites and place them at random places on the grid
    red_pixel = random_sprite(1, 1, density=1, color_palette=[Color.RED])
    for _ in range(np.random.randint(4)):
        _loc = first_free_location(grid, red_pixel, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, red_pixel, r, c, transparent=Color.BLACK, clip=True)

    return grid


# ============= remove below this point for prompting =============
