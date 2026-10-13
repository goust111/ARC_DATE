import numpy as np

import random
from arc_utils import Color, draw_line

# concepts:
# occlusion

# description:
# In the input you will see three regions separated by red vertical bars. Each region is rectangular and the regions are arranged horizontally, so there is a left region, middle region, and a right region. 
# The regions display a yellow pattern, a maroon pattern, and a blue pattern from left to right.
# To make the output, you have to copy the blue pattern first, then overlay the maroon pattern over that, finally overlay the yellow pattern over that as well.

def main(input_grid):
    """
    Composite the three separated color layers from back to front.

    Input structure:
    - Three pattern panels are separated by red vertical bars.
    - The left panel is yellow, the middle panel is maroon, and the right panel is blue.
    - Black cells are transparent within each panel.

    Rule:
    - Split the input at the red divider columns.
    - Use the right blue panel as the base output.
    - Overlay non-black maroon cells, then non-black yellow cells.

    Key insight:
    - Panel order encodes occlusion: blue is back, maroon is middle, and yellow is front.
    """
    # find the location of the vertical red bars that separate the three sections
    red_bars = np.where(input_grid == Color.RED)

    # get the unique column indices of the red bars
    red_bars_c = np.unique(red_bars[1])
    red_bars_c = np.sort(red_bars_c)

    # get the blue pattern from the third section and copy it to make the base of the output grid
    blue_pattern = input_grid[:, red_bars_c[1] + 1 :]
    output_grid = blue_pattern
    # could also have used blit_sprite:
    # output_grid = blit_sprite(output_grid, blue_pattern, x=0, y=0)

    # get the maroon pattern from the second section and overlay it on output grid
    maroon_pattern = input_grid[:, red_bars_c[0] + 1 : red_bars_c[1]]
    output_grid = np.where(maroon_pattern != Color.BLACK, maroon_pattern, output_grid)
    # could also have used blit:
    # output_grid = blit_sprite(output_grid, maroon_pattern, x=0, y=0)

    # get the yellow pattern from the first section and overlay it on output grid
    yellow_pattern = input_grid[:, 0 : red_bars_c[0]]
    output_grid = np.where(yellow_pattern != Color.BLACK, yellow_pattern, output_grid)
    # could also have used blit:
    # output_grid = blit_sprite(output_grid, yellow_pattern, x=0, y=0)

    return output_grid

def generate_input():
    # make a red divider to be used to separate the three sections
    red_divider = np.full((4, 1), Color.RED, dtype=int)

    # make a yellow section and scatter yellow pixels in it
    yellow_section = np.zeros((4,4), dtype=int)
    for _ in range(12):
        r, c = np.random.randint(yellow_section.shape[1]), np.random.randint(yellow_section.shape[0])
        yellow_section[r, c] = Color.YELLOW

    # make a maroon section and scatter maroon pixels in it
    maroon_section = np.zeros((4,4), dtype=int)
    for _ in range(12):
        r, c = np.random.randint(yellow_section.shape[1]), np.random.randint(yellow_section.shape[0])
        maroon_section[r, c] = Color.MAROON

    # make a blue section and scatter blue pixels in it
    blue_section = np.zeros((4,4), dtype=int)
    for _ in range(12):
        r, c = np.random.randint(yellow_section.shape[1]), np.random.randint(yellow_section.shape[0])
        blue_section[r, c] = Color.BLUE
    
    # concatenate the three sections with the red dividers
    grid = np.concatenate([yellow_section, red_divider, maroon_section, red_divider, blue_section], axis=1)

    return grid
# ============= remove below this point for prompting =============
