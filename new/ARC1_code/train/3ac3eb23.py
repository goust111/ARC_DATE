
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, colors_present, draw_line, find_components, first_free_location, paste

# concepts:
# pixel pattern generation, falling downward

# description:
# In the input you will see a grid with several colored pixels at the top.
# To make the output, you should draw a pattern downward from each pixel:
# Color the diagonal corners, and then color downward with a vertical period of 2 from those corners and from the original pixel, making the pattern fall downward.

def main(input_grid):
    """
    Grow a downward period-2 pixel pattern from each colored seed pixel.

    Input structure:
    - The grid has a black background.
    - One or more isolated colored pixels act as seeds.
    - Seed colors are distinct and remain the colors of their generated patterns.

    Rule:
    - Detect all single-pixel colored components.
    - From each seed, start trails at the seed and its four diagonal neighboring positions.
    - Continue each trail downward every two rows while it remains inside the grid.

    Key insight:
    - Each seed generates a sparse falling pattern from five starting offsets, preserving its color.
    """
    # Plan:
    # 1. Find the pixels and make the output
    # 2. Grow the pixel pattern downward from each pixel

    # Extract the pixels (single-cell monochromatic components)
    pixels = find_components(
        input_grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    # Create output grid
    output_grid = np.full_like(input_grid, Color.BLACK)
    width, height = input_grid.shape

    # 2. Grow the pixel pattern downward from each pixel
    for p in pixels:
        r0, c0, _r1, _c1 = p["bbox"]
        r0 = int(r0)
        c0 = int(c0)
        pixel_color = int(p["color"])

        # We do the diagonal corners *and* also the original pixel, so one of the offsets is (0,0).
        for dr, dc in [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            r = r0 + int(dr)
            c = c0 + int(dc)

            # Fall downward (row direction) with vertical period 2.
            while 0 <= r < width and 0 <= c < height:
                output_grid[r, c] = pixel_color
                r += 2
            
    return output_grid


def generate_input():
    # Generate the background grid
    width, height = np.random.randint(10, 20, size=2)
    grid = np.zeros((width, height), dtype=int)

    # Randomly choose the number of pattern
    num_pixels = np.random.randint(1, 4)
    colors = np.random.choice(Color.NOT_BLACK, size=num_pixels, replace=False)

    # Randomly place one pixel on the top row of the grid, each two pixels has at least two pixels padding
    for i in range(num_pixels):
        pixel_sprite = np.full((1,1), colors[i])
        # Find a free spot but just in the top row
        top_c = 0
        top_row = grid[:, top_c:top_c+1]        
        try:
            _loc = first_free_location(top_row, pixel_sprite, transparent=Color.BLACK, border=0, padding=2, padding_connectivity=4)
            if _loc is None:
                raise ValueError("no placement")
            r, _ = _loc
        except:
            # No more space
            break
        paste(grid, pixel_sprite, r, top_c, transparent=Color.BLACK, clip=True)
    
    return grid
