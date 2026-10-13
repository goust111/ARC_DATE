import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# vertical lines, growing

# description:
# In the input you will see individual pixels sprinkled on a black background that are either red, green, or blue
# Turn each red pixel into a vertical bar, and each green or blue pixel into a horizontal bar

def main(input_grid):
    """
    Grow red seed pixels into full vertical bars and green or blue seeds into full horizontal bars.

    Input structure:
    - The input is a black grid with scattered single red, green, and blue pixels.
    - Each colored pixel is an independent seed.
    - Only these three colors define output bars.

    Rule:
    - For every red pixel, fill its entire column red.
    - For every green pixel, fill its entire row green.
    - For every blue pixel, fill its entire row blue.

    Key insight:
    - Seed color chooses bar orientation: red expands vertically, while green and blue expand
      horizontally across their rows.
    """

    # extract the pixels of each color
    red_pixels = (input_grid == Color.RED)
    green_pixels = (input_grid == Color.GREEN)
    blue_pixels = (input_grid == Color.BLUE)

    # prepare a blank output grid
    output_grid = np.zeros_like(input_grid)

    # Turn red pixels into vertical bars (full columns).
    red_locations = np.argwhere(red_pixels)
    for r, c in red_locations:
        output_grid[:, int(c)] = Color.RED
    
    # Turn green and blue pixels into horizontal bars (full rows).
    green_locations = np.argwhere(green_pixels)
    blue_locations = np.argwhere(blue_pixels)
    for r, c in green_locations:
        output_grid[int(r), :] = Color.GREEN
    for r, c in blue_locations:
        output_grid[int(r), :] = Color.BLUE
    
    return output_grid



def generate_input():
    # make a black grid, of a random size
    n, m = np.random.randint(10, 20), np.random.randint(10, 20)
    grid = np.zeros((m, n), dtype=int)

    # sprinkle some red, green, and blue pixels
    num_red, num_green, num_blue = np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)
    for _ in range(num_red):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        grid[r, c] = Color.RED
    for _ in range(num_green):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        grid[r, c] = Color.GREEN
    for _ in range(num_blue):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        grid[r, c] = Color.BLUE

    return grid
# ============= remove below this point for prompting =============
