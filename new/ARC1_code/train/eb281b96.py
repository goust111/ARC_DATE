import numpy as np
from arc_utils import Color, draw_line, paste
import random


# concepts:
# pattern generation, flipping

# description:
# In the input you will see a grid with wave patterns.
# To make the output, you should flip the input grid horizontally and place it in the output grid.
# And then place the flipped input grid in the output grid again.

def main(input_grid):
    """
    Stack alternating original and vertically flipped wave patterns into a taller output.

    Input structure:
    - The input is a colored wave pattern on a black background.
    - Wave segments touch in a way that overlaps by one row when repeated.
    - The output width matches the input width.

    Rule:
    - Create an output whose height holds four overlapping copies of the input.
    - Flip the input vertically for every other copy.
    - Paste original, flipped, original, and flipped patterns with one-row overlaps.

    Key insight:
    - Alternating vertical flips continues the wave smoothly because each copy starts
      before the previous one fully ends.
    """
    # Plan:
    # 1. Create output grid
    # 2. Flip the input grid horizontally and put it in the output repeatedly

    # 1. Create output grid
    H, W = input_grid.shape
    step = H - 1
    out_H = step * 4 + 1
    out_W = W
    output_grid = np.full((out_H, out_W), Color.BLACK, dtype=int)

    flipped_input = np.flipud(input_grid)
    for k in range(4):
        patch = input_grid if (k % 2 == 0) else flipped_input
        r0 = k * step
        output_grid[r0 : r0 + H, :] = patch
    return output_grid

def generate_input():
    # randomly select pattern number and pattern hight
    pattern_num = np.random.randint(3, 5)
    pattern_height = np.random.randint(3, 5)

    # Each pattern is 5 pixels wide
    width, height = pattern_num * 4 + 1, pattern_height
    grid = np.full((width, height), Color.BLACK)

    # Draw the pattern, which looks like a wave
    color = np.random.choice(Color.NOT_BLACK)
    for r in range(0, width, 4):
        grid[r, -1] = color
    for r in range(2, width, 4):
        grid[r, 0] = color
    for r in range(1, width, 2):
        draw_line(grid, r=r, c=1, end_r=r, end_c=height-2, color=color, direction=(0, 1))
    
    return grid
# ============= remove below this point for prompting =============
