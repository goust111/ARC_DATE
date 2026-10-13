import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# patterns, lines

# description:
# In the input, you will see a top row with a sequence of colored pixels, and right below it is a grey line.
# To make the output, copy the first two rows of the input. 
# Then, starting below the grey line, draw rows one color at a time in the order of the colors in the top row from left to right, with the color of each row matching the color of the corresponding pixel in the top row. 
# Repeat this pattern until you reach the bottom of the grid.

def main(input_grid):
    """
    Repeat the top-row color sequence as full rows below the gray separator.

    Input structure:
    - The top row contains the color sequence.
    - A full gray row separates the sequence from the fill area.
    - Rows below the separator are the target area.

    Rule:
    - Copy the input grid.
    - Locate the first full gray row.
    - Fill every row below it with the top-row colors cycling from left to right.

    Key insight:
    - The gray row marks where expansion starts; the top row is the repeating row-color schedule.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # get the colors from the top row
    colors = input_grid[0, :]

    # get the number of colors
    num_colors = len(set(colors))

    # get the row index of the grey line (a full GREY row)
    grey_rows = np.where((input_grid == Color.GREY).all(axis=1))[0]
    assert len(grey_rows) > 0
    grey_line = int(grey_rows[0])

    # draw the rows below the grey line
    for i in range(input_grid.shape[0] - grey_line - 1):
        rr = grey_line + i + 1
        output_grid[rr, :] = int(colors[i % num_colors])

    return output_grid





def generate_input():
    # decide how many colors to use
    num_colors = np.random.randint(2, 6)

    # select colors for the sequence
    colors = np.random.choice(list(Color.NOT_BLACK), num_colors, replace=False)

    # make a grid that will fit the colors along the top row, a grey line below it, and 2 times the number of colors rows below that
    n = num_colors
    m = 2 * num_colors + 2
    grid = np.zeros((m, n), dtype=int)

    # put the colors in the top row
    grid[0, :] = colors

    # put a grey line below the top row
    grid[1, :] = Color.GREY

    return grid
# ============= remove below this point for prompting =============
