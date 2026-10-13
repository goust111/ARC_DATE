import numpy as np
from arc_utils import Color, bbox, bbox_anchor, colors_present, draw_line
import random


# concepts:
# geometric pattern

# description:
# In the input you will see a grid with a single coloured pixel.
# To make the output, move the colored pixel down one pixel and draw a yellow line from the pixel to the top of the grid.
# Finally repeat the yellow line by repeating it horizontally left/right with a period of 2 pixels.

def main(input_grid):
    """
    Move the single colored marker down one row and draw alternating yellow columns above it.

    Input structure:
    - The input is a black grid with exactly one non-black, non-yellow marker pixel.
    - The marker is not on the bottom row.
    - The marker's column defines the parity of the stripe pattern.

    Rule:
    - Move the marker color one row downward and clear its original cell.
    - From the top through the marker's original row, color yellow every other column aligned with the marker column.
    - Leave all other cells black.

    Key insight:
    - The marker supplies both a dropped object and a phase reference for the repeated vertical
      yellow stripe pattern.
    """
    # Plan:
    # 1. Extract the pixel from the input grid
    # 2. Move the pixel one pixel down
    # 3. Draw a yellow line from the pixel to the top of the grid, repeating it horizontally left/right with a period of 2 pixels

    # Locate the single non-black pixel.
    coords = np.argwhere(input_grid != Color.BLACK)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])
    pixel_color = int(input_grid[r, c])

    output_grid = input_grid.copy()

    # Move the pixel down by one row.
    output_grid[r + 1, c] = pixel_color
    output_grid[r, c] = Color.BLACK

    # Draw repeating yellow vertical bars (period=2) from the top down to the original row r.
    for cc in range(0, output_grid.shape[1]):
        if (cc - c) % 2 == 0:
            output_grid[0 : r + 1, cc] = Color.YELLOW
    return output_grid

def generate_input():
    # Generate the background grid
    width, height = np.random.randint(5, 30, size=2)
    grid = np.zeros((width, height), dtype=int)

    # Randomly choose one color
    color = np.random.choice([color for color in Color.NOT_BLACK if color != Color.YELLOW])

    # Randomly place the pixel on the grid
    r, c = np.random.randint(0, width - 1), np.random.randint(0, height - 1)
    grid[r, c] = color

    return grid
# ============= remove below this point for prompting =============
