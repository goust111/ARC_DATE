import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two maroon bitmasks separated by a blue vertical bar
# To make the output, color teal the pixels that are not set in either bitmasks (logical NOR)

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Compute the logical NOR of two maroon masks separated by a blue bar.

    Input structure:
    - The input contains two same-sized maroon/black masks.
    - A full-height blue vertical bar separates the masks.
    - Maroon marks active cells and black marks inactive cells.

    Rule:
    - Locate the blue separator column.
    - Compare the left and right masks cell by cell.
    - Output teal exactly where neither corresponding mask cell is maroon; output black elsewhere.

    Key insight:
    - After the separator identifies the two aligned masks, the transformation is a pixelwise NOR.
    """
    # Find the blue vertical bar (a full column of BLUE).
    H, W = input_grid.shape
    c_bar = None
    for c in range(W):
        if np.all(input_grid[:, c] == Color.BLUE):
            c_bar = c
            break
    assert c_bar is not None

    left_mask = input_grid[:, :c_bar]
    right_mask = input_grid[:, c_bar + 1 :]

    output_grid = np.zeros_like(left_mask)
    output_grid[(left_mask != Color.MAROON) & (right_mask != Color.MAROON)] = Color.TEAL
    
    return output_grid


def generate_input() -> np.ndarray:
    # create a pair of equally sized maroon bitmasks
    width, height = np.random.randint(2, 10), np.random.randint(2, 10)

    grid1 = np.zeros((width, height), dtype=int)
    grid2 = np.zeros((width, height), dtype=int)

    for r in range(width):
        for c in range(height):
            grid1[r, c] = np.random.choice([Color.MAROON, Color.BLACK])
            grid2[r, c] = np.random.choice([Color.MAROON, Color.BLACK])
    
    # separator blue vertical bar (full column)
    bar = np.full((width, 1), Color.BLUE, dtype=int)

    grid = np.concatenate((grid1, bar, grid2), axis=1)

    return grid
# ============= remove below this point for prompting =============
