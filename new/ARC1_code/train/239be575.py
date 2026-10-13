import numpy as np

import random
from arc_utils import Color, contact, draw_line, find_components, first_free_location, paste

# concepts:
# connectivity

# description:
# In the input image you will see several teal pixels and two 2x2 red squares on the black background.
# If the 2x2 red squares are connected by a path of teal pixels, then output a 1x1 teal grid, otherwise, output a 1x1 black grid. 

def main(input_grid):
    """
    Output teal if the two red squares belong to one non-black connected component, otherwise black.

    Input structure:
    - The grid has a black background.
    - Two separated 2 by 2 red squares are present.
    - Teal pixels may form a 4-connected path between the red squares.

    Rule:
    - Build connected components over all non-black cells, ignoring color differences.
    - Check whether any component contains every red-square cell.
    - Return a 1 by 1 teal grid if connected, or a 1 by 1 black grid if not.

    Key insight:
    - The task is a connectivity test where teal pixels can bridge the two red objects.
    """
    # make output grid
    output_grid = np.zeros((1,1), dtype=int)

    # get just the red squares
    red_squares = np.zeros_like(input_grid)
    red_squares[input_grid == Color.RED] = Color.RED

    # get all components that are connected, regardless of color
    connected_components = find_components(
        input_grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="objects",
    )

    # check each connected component to see if it contains both red squares
    for connected_component in connected_components:
        # if it contains both red squares, output teal grid
        if np.all(connected_component[red_squares == Color.RED] == Color.RED):         
            output_grid[:,:] = Color.TEAL
            return output_grid

    # if none of the connected components contain both red squares, output black grid
    output_grid[:,:] = Color.BLACK
    return output_grid


def generate_input():
    # make a black grid first as background
    n, m = np.random.randint(5, 10), np.random.randint(5, 10)
    grid = np.zeros((m, n), dtype=int)

    # make a 2x2 red square sprite
    red_square_sprite = np.full((2,2), Color.RED, dtype=int)

    # place a red square sprite at two random places on the grid
    _loc = first_free_location(grid, red_square_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    x1, y1 = _loc
    paste(grid, red_square_sprite, x1, y1, transparent=Color.BLACK, clip=True)
    _loc = first_free_location(grid, red_square_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    x2, y2 = _loc
    paste(grid, red_square_sprite, x2, y2, transparent=Color.BLACK, clip=True)

    # check that the red squares do not touch
    # if they do, then try again
    if contact(object1=red_square_sprite, object2=red_square_sprite, r1=x1, c1=y1, r2=x2, c2=y2):
        return generate_input()

    # sprinkle teal pixels over the black parts of the grid so they cover roughly a third of it
    for _ in range(m * n // 3):
        r, c = random.randint(0, m - 1), random.randint(0, n - 1)
        # do not sprinkle teal pixels on top of red ones
        if grid[r, c] != Color.RED:
            grid[r, c] = Color.TEAL
    
    return grid
# ============= remove below this point for prompting =============
