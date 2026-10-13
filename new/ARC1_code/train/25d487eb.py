
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, paste


# concepts:
# patterns, growing, horizontal/vertical bars

# description:
# In the input you will see a colored triangle with a single pixel centered in the base of the triangle that is a different color.
# Grow a bar out of and away from the triangle, as if shot out of the tip opposite the differently colored pixel. This bar is the same color as that base pixel.

def main(input_grid):
    """
    Grow a same-colored bar from the triangle tip away from its unique base pixel.

    Input structure:
    - A colored filled triangle sits on a black background.
    - One centered base pixel has a unique color different from the triangle color.
    - The triangle may be rotated to any cardinal orientation.

    Rule:
    - Identify the unique base pixel as the least frequent color.
    - Find the side of that pixel that is outside the triangle.
    - Move in the opposite direction through the triangle and color black cells beyond the tip with the base color.

    Key insight:
    - The unique base pixel reveals both the bar color and, by its exposed outside side,
      the direction from the base through the opposite tip.
    """
    # get output grid ready
    output_grid = input_grid

    # find the differently colored pixel
    colors, counts = np.unique(input_grid, return_counts=True)
    base_color = colors[np.argmin(counts)]

    # find the base location and coordinates from it
    base = np.argwhere(input_grid == base_color).flatten()
    [base_r, base_c] = base

    # find which side of the base is not in the triangle
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        # check if this direction would be outside the grid
        x0, y0 = base_r + dr, base_c + dc
        if 0 <= x0 < len(input_grid) and 0 <= y0 < len(input_grid[0]):
            # check if this direction is in the triangle or not
            if input_grid[x0, y0] != Color.BLACK:
                continue
        # this direction is the opposite of the tip, so grow the bar in the opposite direction from the first black pixel
        x0, y0 = base_r - dr, base_c - dc
        while 0 <= x0 < len(input_grid) and 0 <= y0 < len(input_grid[0]):
            if input_grid[x0, y0] == Color.BLACK:
                output_grid[x0, y0] = base_color
            x0, y0 = x0 - dr, y0 - dc

        return output_grid

    assert 0, "No valid slide found"


def generate_input():
    # first choose a random size for the triangle
    size = random.randint(2, 4)
    width = size*2 + 1
    height = size + 1

    # choose a random color for the triangle
    triangle_color = np.random.choice(Color.NOT_BLACK)

    # choose a random color for the base pixel that is not the same as the triangle color
    base_color = np.random.choice(Color.NOT_BLACK)
    while base_color == triangle_color:
        base_color = np.random.choice(Color.NOT_BLACK)

    # create just the triangle on a black grid for the background
    triangle_sprite = np.zeros((height, width), dtype=int)

    # filled triangle with apex at top row, base at bottom row
    for i in range(height):
        c0 = size - i
        c1 = size + i
        triangle_sprite[i, c0 : c1 + 1] = triangle_color

    # recolor the centered base pixel
    triangle_sprite[height - 1, size] = base_color

    # randomly choose the triangle's orientation by rotating 0 to 3 times
    rotations = random.randint(0, 3)
    triangle_sprite = np.rot90(triangle_sprite, k=rotations)
    th, tw = triangle_sprite.shape

    # now make a larger black grid for the background (ensure placement with 2px margin)
    n = random.randint(max(10, tw + 5), 20)
    m = random.randint(max(10, th + 5), 20)
    grid = np.zeros((m, n), dtype=int)

    # choose a random location on the grid, ensuring at least 2 empty pixels on each edge
    r = random.randint(2, m - th - 2)
    c = random.randint(2, n - tw - 2)

    # place the triangle onto the grid
    paste(grid, triangle_sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
