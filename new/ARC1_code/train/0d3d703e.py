import numpy as np
import random
from arc_utils import Color, draw_line


# concepts:
# color mapping

# description:
# The input is a grid where each column is of the same color. 
# To make the output, change each color according to the following mapping:
# green -> yellow, blue -> gray, red -> pink, teal -> maroon, yellow -> green, gray -> blue, pink -> red, maroon -> teal

def main(input_grid):
    """
    Apply a fixed pairwise color swap mapping to every pixel in the grid.

    Input structure:
    - The input is a small grid whose rows or columns are filled with mapped colors.
    - Colors come from paired non-black color groups.
    - Any unmapped color would remain unchanged.

    Rule:
    - Replace green with yellow and yellow with green.
    - Replace blue with gray, red with pink, and teal with maroon, with each pair swapping back.
    - Apply the mapping independently to every cell.

    Key insight:
    - There is no geometry to infer; the transformation is a global color permutation.
    """
    # Initialize output grid
    output_grid = input_grid.copy()

    # Performs color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid
    
# Constructing the color map
color_map = {Color.GREEN : Color.YELLOW, 
             Color.BLUE : Color.GRAY, 
             Color.RED : Color.PINK,
             Color.TEAL : Color.MAROON,
             Color.YELLOW : Color.GREEN, 
             Color.GRAY : Color.BLUE, 
             Color.PINK : Color.RED,
             Color.MAROON : Color.TEAL             
            }


def generate_input():
    grid = np.full((3, 3), Color.BLACK)
    for r in range(grid.shape[0]):
        grid[r, :] = random.choice(list(color_map.keys()))
    return grid
# ============= remove below this point for prompting =============
