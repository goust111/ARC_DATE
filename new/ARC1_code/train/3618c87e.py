
import random

import numpy as np
from typing import *
from arc_utils import Color, draw_line

# concepts:
# color, falling

# description:
# In the input, you should see a gray baseline at the bottom. For each gray baseline pixel, there may or may not be gray pixels above it. If there are gray pixels above it, you can see a blue pixel above the gray pixels.
# To make the output, make the blue pixels fall downward, falling through the gray baseline until they hit the bottom.

def main(input_grid):
    """
    Drop each non-background, non-baseline pixel to the lowest available cell in its column.

    Input structure:
    - The bottom row is a uniform baseline color.
    - The most common color is the background.
    - Other colored pixels may sit above baseline or background cells in their columns.

    Rule:
    - Identify the baseline color from the bottom row and the background by frequency.
    - Scan each column from bottom to top.
    - Move every non-background, non-baseline pixel to the lowest background or baseline cell below it.

    Key insight:
    - The baseline is treated as passable landing space, so falling pixels can replace it at the bottom.
    """
    output_grid = np.copy(input_grid)

    H, W = output_grid.shape

    # Find the color of the bottom baseline
    baseline = output_grid[H - 1, :]
    baseline_colors = np.unique(baseline)
    assert len(baseline_colors) == 1
    baseline_color = baseline_colors[0]

    # Find the color of the background, which is the most common color
    background_color = np.argmax(np.bincount(output_grid.flatten()))

    # Now make all the other colors fall down (along rows).
    for c in range(W):
        for r in range(H - 2, -1, -1):
            if output_grid[r, c] != background_color and output_grid[r, c] != baseline_color:
                # Find the lowest background/baseline spot below it.
                possible_r_values = [
                    rr
                    for rr in range(r + 1, H)
                    if output_grid[rr, c] == background_color or output_grid[rr, c] == baseline_color
                ]
                if possible_r_values:
                    rr_dest = max(possible_r_values)
                    output_grid[rr_dest, c] = output_grid[r, c]
                    output_grid[r, c] = background_color                  
  
    return output_grid


def generate_input():
    H = W = 5
    input_grid = np.zeros((H, W), dtype=int)

    # bottom baseline row is grey
    input_grid[H - 1, :] = Color.GREY

    # randomly select 1 or 2 columns to have a blue pixel above a grey stack
    num_blue_squares = random.choice([1, 2])
    cols = random.sample(range(W), num_blue_squares)

    for c in cols:
        input_grid[H - 2, c] = Color.GREY
        input_grid[H - 3, c] = Color.BLUE

    return input_grid
