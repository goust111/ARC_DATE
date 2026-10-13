
import numpy as np
from typing import *

import random
from arc_utils import Color, bbox, draw_line

# concepts:
# pixel patterns, pyramid, color alternation

# description:
# In the input you will see a single orange line that connects to the top of the grid.
# To make the output, you should draw a pyramid pattern outward from the orange line.
# The pattern is expanded from the orange line to the left and right of the grid.
# Each line of the pattern is one cell shorter than the previous one, and the color alternates between orange and teal.

def main(input_grid):
    """
    Expand a vertical colored line into a centered alternating-color pyramid.

    Input structure:
    - The input contains one non-black line segment on a black background.
    - The line's dominant column gives the pyramid center column.
    - The line length determines the pyramid height and widest row.

    Rule:
    - Identify the line color, center column, and lowest occupied row.
    - Build a black output grid of the same size.
    - For each row above the line base, draw a centered run whose colors alternate by distance from the center.

    Key insight:
    - The original line is the pyramid axis; moving upward shortens the run, while
      horizontal parity controls the alternating colors.
    """
    # Plan (from the original seed solver):
    # 1. Parse the input
    # 2. Draw the left side of the pyramid
    # 3. Draw the right side of the pyramid
    #
    # Note: the legacy seed version ran under a transpose wrapper. This migrated solver
    # runs canonical `(r,c)`, so some low-level details differ, but the intent is the same.

    coords = np.argwhere(input_grid != Color.BLACK)
    if coords.size == 0:
        return input_grid.copy()

    # Line parameters
    r_max = int(coords[:, 0].max())
    cols, counts = np.unique(coords[:, 1], return_counts=True)
    c0 = int(cols[int(np.argmax(counts))])
    line_color = int(input_grid[int(coords[0, 0]), int(coords[0, 1])])

    if line_color == Color.ORANGE:
        alt_color = Color.TEAL
    elif line_color == Color.TEAL:
        alt_color = Color.ORANGE
    else:
        alt_color = Color.TEAL

    H, W = input_grid.shape
    out = np.full((H, W), Color.BLACK, dtype=int)

    # For each row of the line, draw a centered run with alternating colors.
    for r in range(r_max + 1):
        half = r_max - r
        for dc in range(-half, half + 1):
            c = c0 + int(dc)
            if 0 <= c < W:
                out[r, c] = int(line_color) if (abs(int(dc)) % 2 == 0) else int(alt_color)

    return out

def generate_input():
    # Generate the background grid
    width, height = np.random.randint(5, 10, size=2)
    grid = np.full((width, height), Color.BLACK)

    # Randomly generate one orange line connecting to the top
    # The line should be at least 4 cells long
    length = np.random.randint(3, height - 1)

    # Randomly choose one position to start the line
    r = np.random.randint(0, width)
    line_color = Color.ORANGE
    draw_line(grid=grid, r=r, c=0, direction=(0, 1), length=length, color=line_color)

    return grid
