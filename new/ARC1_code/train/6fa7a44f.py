import numpy as np

import random
from arc_utils import Color, draw_line, random_sprite

# concepts:
# reflection

# description:
# In the input you will see a square pattern of random colors except black.
# To make the output, reflect the pattern vertically, and put the reflected pattern beneath the input pattern.

def main(input_grid):
    """
    Append a vertical reflection of the input pattern below the original.

    Input structure:
    - The input is a dense square pattern of non-black colors.
    - The full input pattern is preserved in the output.
    - The output is twice as tall as the input.

    Rule:
    - Treat the entire input as the source pattern.
    - Flip the pattern top-to-bottom.
    - Concatenate the original pattern above the reflected copy.

    Key insight:
    - The lower half is a mirror image across the input's bottom edge.
    """
    # take the input pattern
    pattern = input_grid

    # reflect the pattern vertically (top-bottom)
    reflected_pattern = pattern[::-1, :]

    # make the output grid
    output_grid = np.concatenate((pattern, reflected_pattern), axis=0)

    return output_grid

def generate_input():
    # make a random square pattern of random size and colors
    n = m = np.random.randint(3, 6)
    grid = random_sprite(n, m, density=1, symmetry="not_symmetric", color_palette=Color.NOT_BLACK)

    return grid
# ============= remove below this point for prompting =============
