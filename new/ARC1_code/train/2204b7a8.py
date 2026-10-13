
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line

# concepts:
# proximity, color change, horizontal/vertical bars

# description:
# In the input you will see a pair of lines on the edge of the canvas that are either horizontal or vertical, and also green pixels randomly placed in between the lines
# Change the color of each green pixel to match the color of the line it is closest to

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each green pixel to match the nearest colored edge line.

    Input structure:
    - Two full edge lines of different colors sit on opposite sides of the grid.
    - Green pixels are scattered between those lines.
    - The remaining cells are black.

    Rule:
    - Remove green pixels from a line-only copy of the grid.
    - For each green pixel, find the closest non-black line cell by Manhattan distance.
    - Replace the green pixel with that closest line cell's color.

    Key insight:
    - The edge lines are color sources, and each green marker inherits from the nearest source line.
    """
    # find the two lines by removing the green pixels
    lines = np.copy(input_grid)
    lines[input_grid == Color.GREEN] = Color.BLACK

    # lines now contains only the lines, which are going to be used to assign color to the green pixels
    output_grid = np.copy(input_grid)

    for r in range(input_grid.shape[0]):
        for c in range(input_grid.shape[1]):
            if input_grid[r, c] == Color.GREEN:
                # find the closest point on the lines
                closest_r, closest_c = min([(i, j) for i in range(input_grid.shape[0]) for j in range(input_grid.shape[1]) if lines[i, j] != Color.BLACK],
                                           key=lambda p: abs(p[0] - r) + abs(p[1] - c))
                color_of_closest_line = lines[closest_r, closest_c]
                output_grid[r, c] = color_of_closest_line
    
    return output_grid

def generate_input() -> np.ndarray:
    # make a black grid first as background
    n, m = 10, 10
    grid = np.zeros((m, n), dtype=int)

    # sprinkle green pixels randomly
    for green_index in range(random.randint(5, 20)):
        r, c = random.randint(0, m - 1), random.randint(0, n - 1)
        grid[r, c] = Color.GREEN

    # decide on a pair of colors for the horizontal/vertical lines
    line_colors = random.sample([color for color in Color.ALL_COLORS if color != Color.GREEN and color != Color.BLACK], 2)

    # flip a coin to decide whether it is horizontal or vertical
    is_horizontal = random.choice([True, False])
    if is_horizontal:
        grid[:, 0] = line_colors[0]
        grid[:, -1] = line_colors[1]
    else:
        grid[0, :] = line_colors[0]
        grid[-1, :] = line_colors[1]

    return grid
