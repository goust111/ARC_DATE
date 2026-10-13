
import numpy as np
from typing import *
from arc_utils import Color, bbox, bbox_anchor, colors_present, draw_line, find_components
import random

# concepts:
# pattern generation

# description:
# In the input you will see a grid with two colored pixels.
# To make the output, you should draw a pattern from the two pixels.
# step 1: draw a horizontal line outward from the two pixels in their respective colours.
# step 2: draw a border around the canvas whose top half is colored to match the top pixel and whose bottom half matches the bottom pixel.


def main(input_grid):
    """
    Draw horizontal lines through two colored pixels and split the border color by their vertical order.

    Input structure:
    - The input is a black grid with exactly two colored single pixels.
    - One pixel lies in the upper half and the other in the lower half.
    - The grid height is even so the final border split has a clear midpoint.

    Rule:
    - Sort the two pixels from top to bottom.
    - Draw a full horizontal line through each pixel in its own color.
    - Draw the border, then recolor all non-black border and line pixels in the top half to the top color and in the bottom half to the bottom color.

    Key insight:
    - The two pixels define both the line rows and the two-color frame: vertical order decides which
      color owns each half of the output.
    """
    # Plan:
    # 1. Parse the input and determine which pixel is top in which is bottom
    # 2. Draw horizontal lines
    # 3. Draw border, colored appropriately

    background = Color.BLACK

    # 1. Input parsing: find the two single-pixel components and sort by row (top -> bottom).
    pixels = find_components(
        input_grid,
        bg=background,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    pixels.sort(key=lambda d: int(d["bbox"][0]))  # r0
    assert len(pixels) == 2

    # 2. Draw horizontal lines across the row of each pixel, in that pixel's color.
    output_grid = np.full_like(input_grid, background)
    H, W = output_grid.shape
    for p in pixels:
        r0, _c0, _r1, _c1 = p["bbox"]
        color = int(p["color"])
        draw_line(output_grid, r=int(r0), c=0, direction=(0, 1), length=None, color=color)

    # 3. Draw border. Then enforce top-half vs bottom-half coloring.
    top_color = int(pixels[0]["color"])
    bottom_color = int(pixels[1]["color"])

    # Border (we recolor halves afterwards, so edges get the correct split).
    draw_line(output_grid, r=0, c=0, direction=(0, 1), length=None, color=top_color)        # top edge
    draw_line(output_grid, r=0, c=0, direction=(1, 0), length=None, color=top_color)        # left edge
    draw_line(output_grid, r=H - 1, c=0, direction=(0, 1), length=None, color=bottom_color) # bottom edge
    draw_line(output_grid, r=0, c=W - 1, direction=(1, 0), length=None, color=bottom_color) # right edge

    # Everything below the midline is bottom color, everything above is top color.
    top = output_grid[: H // 2, :]
    top[top != background] = top_color
    bottom = output_grid[H // 2 :, :]
    bottom[bottom != background] = bottom_color

    return output_grid

def generate_input():
    # Generate the grid, ensure the grid has an even height so that we can split it evenly between top and bottom
    width, height = np.random.randint(10, 20, size=2)
    while (height % 2 == 1):
        width, height = np.random.randint(10, 20, size=2)
    background = Color.BLACK
    grid = np.full((width, height), background)

    # Randomly choose two colors
    colors = np.random.choice(Color.NOT_BLACK, size=2, replace=False)

    # Randomly place the pixels so that one of them is the in the bottom half and the other in the top half
    x1, x2 = np.random.choice(range(width), size=2, replace=False)
    y1, y2 = np.random.randint(2, height//2-1), np.random.randint(height//2 + 1, height-2)
    grid[x1, y1] = colors[0]
    grid[x2, y2] = colors[1]

    return grid
