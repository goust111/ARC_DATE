import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# pixel manipulation

# description:
# In the input you will see a medium sized grid width individual colored pixels, some of which are red or blue (those ones are special)
# To make the output:
# 1. For each red pixel, add yellow pixels in its immediate diagonals (northeast, northwest, southeast, southwest)
# 2. For each blue pixel, add orange pixels in its immediate neighbors (up, down, left, right)

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Add yellow diagonal marks around red pixels and orange orthogonal marks around blue pixels.

    Input structure:
    - The input is a black grid with scattered single colored pixels.
    - Red and blue pixels are special markers.
    - Other colors may appear but do not trigger any expansion.

    Rule:
    - Preserve the original grid.
    - Around each red pixel, color the four diagonal neighbors yellow when in bounds.
    - Around each blue pixel, color the four cardinal neighbors orange when in bounds.

    Key insight:
    - Marker color selects the neighborhood geometry: red emits diagonal corners, while blue emits
      orthogonal side-neighbors.
    """

    output_grid = np.copy(input_grid)

    for r in range(input_grid.shape[0]):
        for c in range(input_grid.shape[1]):
            color = input_grid[r][c]
            if color == Color.RED:
                # put yellow pixels in the diagonals
                for dr in [-1, 1]:
                    for dc in [-1, 1]:
                        if 0 <= r+dr < input_grid.shape[0] and 0 <= c+dc < input_grid.shape[1]:
                            output_grid[r+dr, c+dc] = Color.YELLOW
            elif color == Color.BLUE:
                # put orange pixels in the neighbors
                for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    if 0 <= r+dr < input_grid.shape[0] and 0 <= c+dc < input_grid.shape[1]:
                        output_grid[r+dr, c+dc] = Color.ORANGE

    return output_grid



def generate_input() -> np.ndarray:
    n, m = np.random.randint(5, 20), np.random.randint(5, 20)
    grid = np.zeros((m, n), dtype=int)

    num_red, num_blue, num_other = np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)

    for _ in range(num_red):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        grid[r, c] = Color.RED
    
    for _ in range(num_blue):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        grid[r, c] = Color.BLUE
    
    for _ in range(num_other):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        grid[r, c] = np.random.choice([ color for color in Color.NOT_BLACK if color not in [Color.RED, Color.BLUE] ])


    return grid
# ============= remove below this point for prompting =============
