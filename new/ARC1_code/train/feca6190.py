import numpy as np

import random
from arc_utils import Color, draw_line, paste

# concepts:
# pixel patterns, counting, expanding, diagonal lines

# description:
# In the input you will see a line with several colored pixels.
# To make the output, create a square grid and place the input line at the bottom-left. Each colored pixel shoots a diagonal line outward toward the upper right.
# The length of the output grid is the product of the number of colored input pixels and the length of the input line.

def main(input_grid):
    """
    Sweep a colored input line diagonally upward-right to form a large square pattern.

    Input structure:
    - The input is a one-cell-wide line containing black and colored pixels.
    - Non-black colors appear in the line and determine the output scale.
    - The output is a black square canvas.

    Rule:
    - Set the square size to input line length times the number of distinct non-black colors.
    - Place the original line at the bottom-left of the output.
    - Repeatedly paste the same line one row up and one column right with clipping.

    Key insight:
    - Diagonal rays are produced by translating the whole sparse line, so every colored
      input pixel leaves a parallel diagonal trace.
    """
    # Plan:
    # 1. Figure out how big the output should be and make a blank output grid
    # 2. Place the input line at the bottom left of the output grid
    # 3. Repeatedly translate it diagonally toward the upper right corner, equivalently shooting diagonal lines from each colored pixel

    # 1. The output grid size is the number of non-black pixels in the input grid times the original grid width
    # Canonical (r,c): input is a 1xW line; output size = (line_length * #distinct non-black colors).
    input_height, input_width = input_grid.shape
    num_different_colors = len(set(map(int, np.unique(input_grid))) - {Color.BLACK})
    output_size = int(input_width) * int(num_different_colors)
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # 2-3. Place the input at the bottom left and then move it upward and rightward
    bottommost_r = output_size - 1
    for iteration in range(output_size):
        r0 = bottommost_r - iteration
        c0 = iteration
        # Legacy seeds used clipping blits; match that behavior.
        output_grid = paste(output_grid, input_grid, r0, c0, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Generate a line
    width = np.random.randint(3, 7)
    height = 1
    grid = np.full((width, height), Color.BLACK)

    # Randomly choose colors for each grid
    colors = np.random.choice(Color.NOT_BLACK, size=width, replace=False)
    for r in range(width):
        # Randomly assign a color to each pixel
        if np.random.rand() < 0.6:
            grid[r, 0] = colors[r]

    return grid

# ============= remove below this point for prompting =============
