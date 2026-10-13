
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line

# concepts:
# magnetism, lines

# description:
# In the input, you will see a horizontal grey line on a black background, with red and blue pixels scattered on either side of the line.
# To make the output, draw vertical lines from each of the blue and red pixels, with lines from the red pixels going toward the grey line and lines from the blue pixels going away from the grey line. 
# These lines should stop when they hit the grey line or the edge of the grid.

def main(input_grid):
    """
    Draw red segments toward the gray divider and blue segments away from it.

    Input structure:
    - The input is a black grid with one straight gray row or column.
    - Red and blue pixels appear on both sides of the gray divider.
    - The gray divider stops lines that are drawn toward it.

    Rule:
    - Detect whether the gray divider is horizontal or vertical.
    - Draw red lines from red pixels toward the divider.
    - Draw blue lines from blue pixels away from the divider toward the edge.

    Key insight:
    - The gray line is an axis of attraction for red and repulsion for blue.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # Find the grey line and determine whether it's horizontal (a row) or vertical (a column).
    grey_coords = np.argwhere(output_grid == Color.GREY)
    grey_rows = np.unique(grey_coords[:, 0])
    grey_cols = np.unique(grey_coords[:, 1])

    red_coords = np.argwhere(output_grid == Color.RED)
    blue_coords = np.argwhere(output_grid == Color.BLUE)

    if len(grey_rows) == 1:
        # Horizontal line at row = grey_r
        grey_r = int(grey_rows[0])

        # Red pixels: draw toward the line (down if above, up if below).
        for r, c in red_coords:
            r = int(r)
            c = int(c)
            dr = 1 if r < grey_r else -1
            draw_line(output_grid, r, c, length=None, color=Color.RED, direction=(dr, 0), stop_at={Color.GREY})

        # Blue pixels: draw away from the line (up if above, down if below).
        for r, c in blue_coords:
            r = int(r)
            c = int(c)
            dr = -1 if r < grey_r else 1
            draw_line(output_grid, r, c, length=None, color=Color.BLUE, direction=(dr, 0), stop_at={Color.GREY})

    elif len(grey_cols) == 1:
        # Vertical line at col = grey_c
        grey_c = int(grey_cols[0])

        # Red pixels: draw toward the line (right if left, left if right).
        for r, c in red_coords:
            r = int(r)
            c = int(c)
            dc = 1 if c < grey_c else -1
            draw_line(output_grid, r, c, length=None, color=Color.RED, direction=(0, dc), stop_at={Color.GREY})

        # Blue pixels: draw away from the line (left if left, right if right).
        for r, c in blue_coords:
            r = int(r)
            c = int(c)
            dc = -1 if c < grey_c else 1
            draw_line(output_grid, r, c, length=None, color=Color.BLUE, direction=(0, dc), stop_at={Color.GREY})

    else:
        raise ValueError("Expected a single straight grey line (one row or one column).")

    return output_grid

def generate_input():
    # make a 10x10 black grid for the background
    n = m = 10
    grid = np.zeros((n,m), dtype=int)

    # make a horizontal grey line on a random row about halfway down the grid
    row = np.random.randint(m//3, 2*(m//3))
    grid[:, row] = Color.GREY

    # scatter a random number of blue and red pixels on either side of the grey line so that no pixel is in the same column as any other pixel on its side of the grey line
    # select columns for the pixels above the grey line
    cols = np.random.choice(np.arange(m), size=np.random.randint(3, 7), replace=False)
    for col in cols:
      # randomly choose whether to make the pixel red or blue
      if np.random.rand() < 0.5:
        grid[col, np.random.randint(row-1)] = Color.RED
      else:
        grid[col, np.random.randint(row-1)] = Color.BLUE
    # select columns for the pixels below the grey line
    cols = np.random.choice(np.arange(m), size=np.random.randint(3, 7), replace=False)
    for col in cols:
      # randomly choose whether to make the pixel red or blue
      if np.random.rand() < 0.5:
        grid[col, np.random.randint(row+1, m)] = Color.RED
      else:
        grid[col, np.random.randint(row+1, m)] = Color.BLUE 
    
    return grid

