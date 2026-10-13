
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line

# concepts:
# repetition, horizontal/vertical bars

# description:
# In the input you will see a pair of colored pixels
# Make each pixel into a horizontal/vertical bar by connecting it to the other side of the canvas
# Then, repeat the bars indefinitely in the same direction: either downward (for horizontal bars) or rightward (for vertical bars)

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Turn two seed pixels into alternating full-length bars repeated at twice their separation.

    Input structure:
    - The input contains exactly two non-black pixels on a black grid.
    - Their placement determines whether the bars run across rows or columns.
    - The two seed colors are preserved as the alternating bar colors.

    Rule:
    - If either seed lies on a side edge, draw full row bars through the two seed rows.
    - Otherwise, draw full column bars through the two seed columns.
    - Repeat both bars along the perpendicular axis using a stride of twice their separation.

    Key insight:
    - The two seeds define both the orientation and period of a two-color stripe pattern.
    """
    # find the individual coloured pixels
    colored_pixels = np.argwhere(input_grid != Color.BLACK)

    # there should be exactly two colored pixels
    assert len(colored_pixels) == 2

    # find the two pixels
    pixel1, pixel2 = colored_pixels
    x1, y1 = pixel1
    x2, y2 = pixel2
    color1, color2 = input_grid[x1, y1], input_grid[x2, y2]

    # make the horizontal/vertical bars
    output_grid = np.copy(input_grid)

    # check if they should be horizontal (constant y) or vertical (constant x)
    # if they are on the top or bottom, they should be vertical
    if y1 == 0 or y1 == input_grid.shape[1] - 1 or y2 == 0 or y2 == input_grid.shape[1] - 1:
        # vertical bars: constant x
        output_grid[x1, :] = color1
        output_grid[x2, :] = color2

        # repeat the vertical bars indefinitely
        # first, figure out how far apart they are.
        dr = abs(x2 - x1)
        # next, each repetition needs to be separated by twice that amount, because we have two lines
        dr = 2 * dr
        # finally, repeat the vertical bars indefinitely, using the same colors and using a list slice to repeat the bars
        output_grid[x1::dr, :] = color1
        output_grid[x2::dr, :] = color2
    else:
        # horizontal bars: constant y
        output_grid[:, y1] = color1
        output_grid[:, y2] = color2

        # repeat the horizontal bars indefinitely
        # first, figure out how far apart they are.
        dc = abs(y2 - y1)
        # next, each repetition needs to be separated by twice that amount, because we have two lines
        dc = 2 * dc
        # finally, repeat the horizontal bars indefinitely, using the same colors and using a list slice to repeat the bars
        output_grid[:, y1::dc] = color1
        output_grid[:, y2::dc] = color2

    return output_grid



def generate_input() -> np.ndarray:
    # make a grid that is narrow and tall. This is for horizontal lines, and we will flip a coin at the end to transpose everything to make things vertical
    n = np.random.randint(5, 10)
    m = np.random.randint(max(n + 1, 7), 30)
    grid = np.zeros((n, m), dtype=int)

    # Place two colored pixels.
    # Keep y strictly away from the border so the non-transposed case uses the "horizontal bars" branch,
    # and ensure the two pixels have different x so dr != 0 in the other branch.
    x1 = random.choice([0, n - 1])
    x2 = (n - 1) if x1 == 0 else 0
    y1 = np.random.randint(1, m // 2)
    y2 = np.random.randint(y1 + 2, m - 1)
    color1 = random.choice(Color.NOT_BLACK)
    color2 = random.choice([c for c in Color.NOT_BLACK if c != color1])

    grid[x1, y1] = color1
    grid[x2, y2] = color2

    # decide to either do vertical or horizontal
    vertical = np.random.choice([True, False])
    if vertical: 
        # transpose the grid
        grid = grid.T

    return grid
