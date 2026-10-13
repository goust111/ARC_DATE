import numpy as np

import random
from arc_utils import Color, draw_line, first_free_location, paste, randomly_scatter_points

# concepts:
# rectangle detection, background shape detection

# description:
# In the input you will see a grid with scattered one color pixels
# To make the output grid, you should detect the 3x3 black square in the random color pattern
# and replace it with a 3x3 blue square

def main(input_grid):
    """
    Replace every fully black 3x3 gap in a noisy colored field with a blue 3x3 square.

    Input structure:
    - The input is a fixed-size grid with many scattered non-blue colored pixels.
    - One or more fully black 3x3 squares are embedded in the pattern.
    - Existing colored pixels should remain unchanged.

    Rule:
    - Scan all 3x3 windows in the input.
    - Select windows that are entirely black.
    - Fill each still-black matching window with blue.

    Key insight:
    - The targets are uniform black square voids amid the colored scatter.
    """
    # Plan: 
    # 1. Detect the 3x3 regions that are all black
    # 2. Draw a blue 3x3 in those regions

    # 1. Detect the 3x3 regions that are all black
    region_len = 3
    output_grid = np.copy(input_grid)
    matching_regions = [(r, c) for r in range(len(input_grid) - (region_len - 1)) for c in range(len(input_grid[0]) - (region_len - 1)) if np.all(input_grid[r:r + region_len, c:c + region_len] == Color.BLACK)]

    # 2. Draw a blue 3x3 in those regions
    for r, c in matching_regions:
        # Check if the region is all black
        if np.all(output_grid[r:r+region_len, c:c+region_len] == Color.BLACK):
            output_grid[r:r+region_len, c:c+region_len] = Color.BLUE

    return output_grid    

def generate_input():
    # Generate the background grid with size of n x m.
    n, m = 20, 20
    grid = np.zeros((m, n), dtype=int)

    # Get the random scatter color pixels on the grid.
    avaliable_colors = [c for c in Color.NOT_BLACK if c != Color.BLUE]
    background_color = np.random.choice(avaliable_colors)
    
    # Generate random color pixels on the grid.
    randomly_scatter_points(grid, color=background_color, density=0.7)

    # Randomly generate the number of black squares.
    square_num = np.random.randint(1, 4)

    # Generate the black squares on the grid.
    square_len = 3
    for _ in range(square_num):
        square = np.full((square_len, square_len), Color.BLACK)
        try:
            # Get the random free location for the black square.
            _loc = first_free_location(grid, square, transparent=background_color, border=0, padding=0, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, cc = _loc
        except:
            continue
        # Blit the black square on the grid.
        grid = paste(grid, square, r, cc, transparent=background_color, clip=True)
     
    return grid
# ============= remove below this point for prompting =============
