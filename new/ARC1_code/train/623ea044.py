import numpy as np
import random
from arc_utils import Color, bbox, draw_line


# concepts:
# diagonal lines

# description:
# In the input you will see one colored pixel on a black background.
# To make the output, make two diagonal lines that are the same color as the colored pixel and intersect at the location of the colored pixel.

def main(input_grid):
    """
    Draw both diagonals through the single colored pixel using that pixel's color.

    Input structure:
    - The grid is black except for exactly one non-black pixel.
    - The input grid is square or rectangular.
    - The lone pixel's color is the output line color.

    Rule:
    - Locate the single non-black pixel.
    - Draw in all four diagonal directions from that location.
    - Keep drawing to the canvas edge with clipping.

    Key insight:
    - The lone pixel provides both the intersection point and the color for the two full
      diagonal lines.
    """
    # make output grid
    output_grid = np.copy(input_grid)

    # locate the colored pixel (the only non-black pixel)
    coords = np.argwhere(input_grid != Color.BLACK)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])
    color = int(input_grid[r, c])

    # draw diagonals
    L = max(output_grid.shape) * 3
    for dr, dc in [(1, -1), (-1, 1), (-1, -1), (1, 1)]:
        draw_line(output_grid, r, c, length=L, color=color, direction=(dr, dc), clip=True)

    return output_grid

    



def generate_input():
    # make a square black grid for the background first
    n = m = np.random.randint(5, 20)
    grid = np.zeros((n, m), dtype=int)

    # put a randomly colored pixel at a random point on the grid
    color = random.choice(list(Color.NOT_BLACK))
    r, c = np.random.randint(0, n-1), np.random.randint(0, m-1)
    grid[r,c] = color

    return grid
# ============= remove below this point for prompting =============
