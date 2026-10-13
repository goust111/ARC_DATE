import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two blue bitmasks separated by a grey vertical bar
# To make the output, color teal the red that are set in both bitmasks (logical AND)

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Convert two blue bitmasks separated by a gray bar into their pixelwise overlap.

    Input structure:
    - The input contains two same-sized blue/black masks.
    - A full-height gray vertical bar separates the left and right masks.
    - Blue marks active cells and black marks inactive cells.

    Rule:
    - Locate the gray separator column.
    - Compare the masks cell by cell.
    - Output red exactly where both corresponding mask cells are blue; output black elsewhere.

    Key insight:
    - The gray bar is only a layout delimiter; after splitting on it, the task is a logical AND
      between the two aligned masks.
    """
    # Find the grey vertical bar (a full column of GREY).
    H, W = input_grid.shape
    c_bar = None
    for c in range(W):
        if np.all(input_grid[:, c] == Color.GREY):
            c_bar = c
            break
    assert c_bar is not None

    left_mask = input_grid[:, :c_bar]
    right_mask = input_grid[:, c_bar + 1 :]

    output_grid = np.zeros_like(left_mask)
    output_grid[(left_mask == Color.BLUE) & (right_mask == Color.BLUE)] = Color.RED
    
    return output_grid


def generate_input() -> np.ndarray:
    # create a pair of equally sized maroon bitmasks
    width, height = np.random.randint(2, 10), np.random.randint(2, 10)

    grid1 = np.zeros((width, height), dtype=int)
    grid2 = np.zeros((width, height), dtype=int)

    for r in range(width):
        for c in range(height):
            grid1[r, c] = np.random.choice([Color.BLUE, Color.BLACK])
            grid2[r, c] = np.random.choice([Color.BLUE, Color.BLACK])
    
    # separator grey vertical bar (full column)
    bar = np.full((width, 1), Color.GREY, dtype=int)

    grid = np.concatenate((grid1, bar, grid2), axis=1)

    return grid
# ============= remove below this point for prompting =============
