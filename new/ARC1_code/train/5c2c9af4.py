
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, in_bounds

# concepts:
# repeating pattern, connecting colors

# description:
# In the input grid, you will see an all black grid with three dots of the same color in a perfect 45 degree diagonal, but equally spaced apart from each other.
# To create the output grid, connect the outer two of the three dots with a square border shape. The square border contains the two dots as corners, and is centered on the third center dot. Then make another square border that is the same distance (number of background cells) from the existing border as the existing border is from the center dot. Repeat making square borders of the same distance outwards until the grid is filled.

def main(input_grid):
    """
    Grow concentric square borders from three equally spaced diagonal seed pixels.

    Input structure:
    - The grid is black except for exactly three same-colored pixels.
    - The pixels lie on a 45-degree diagonal with equal spacing.
    - The middle pixel is the center for the square borders.

    Rule:
    - Sort the three seed pixels and measure the spacing to the center pixel.
    - Draw a square border centered on the middle pixel using that spacing.
    - Repeat outward at integer multiples of the spacing until no new cells are added.

    Key insight:
    - The outer seed pixels define the first square's diagonal corners, and the same spacing
      drives the repeating larger borders.
    """
    # Plan:
    # 1. get the dots
    # 2. get the center dot, and the two outer dots
    # 3. calculate the distance from the center dot of the outer dots.
    # 4. make a helper function for drawing a square of a certain distance from the center dot
    # 5. repeat making squares of multiples of that distance until no new cells are filled in on the grid.

    # get a list of locations
    pixel_xs, pixel_ys = np.where(input_grid != Color.BLACK)
    pixel_locations = list(zip(list(pixel_xs), list(pixel_ys)))
    assert len(pixel_locations) == 3
    
    # sort by x coordinate
    pixel0, pixel1, pixel2 = sorted(pixel_locations, key=lambda l: l[0])
    color = input_grid[pixel0[0], pixel0[1]]
    width = pixel1[0] - pixel0[0]

    def in_bounds(grid, r, c):
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

    def draw_square_border(grid, r, c, w, h, color):
        # x, y is the top left corner
        for dr in range(w+1):
            # top border
            if in_bounds(grid, r+dr, c):
                grid[r+dr, c] = color
            # bottom border
            if in_bounds(grid, r+dr, c+h):
                grid[r+dr, c+h] = color
        for dc in range(h+1):
            # left border
            if in_bounds(grid, r, c+dc):
                grid[r, c+dc] = color
            # right border
            if in_bounds(grid, r+w, c+dc):
                grid[r+w, c+dc] = color

    output_grid = input_grid.copy()
    i = 1
    while True:
        top_left_r = pixel1[0] - width * i
        top_left_c = pixel1[1] - width * i
        w = 2 * (width * i)
        old_grid = output_grid.copy()
        draw_square_border(output_grid, top_left_r, top_left_c, w, w, color)
        if not np.any(old_grid != output_grid):
            break
        i += 1

    return output_grid

def generate_input():
    # 1. make a 28x28 black grid
    # 2. choose a distance from the center between 1 and 10
    # 3. make the three initial dots on a grid
    # 4. blit onto a random spot on the black grid
    input_grid = np.full((28,28), Color.BLACK)
    distance = np.random.randint(2, 11)
    color = np.random.choice(Color.NOT_BLACK)
    center_r = np.random.randint(distance, 28 - distance)
    center_c = np.random.randint(distance, 28 - distance)
    input_grid[center_r, center_c] = color
    input_grid[center_r - distance, center_c - distance] = color
    input_grid[center_r + distance, center_c + distance] = color
    # randomly rotate the input grid sometimes
    if np.random.rand() < 0.55:
        input_grid = np.rot90(input_grid, k=-1)

    return input_grid
