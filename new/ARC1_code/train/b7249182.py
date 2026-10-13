
import random

import numpy as np
from typing import *
from arc_utils import Color, colors_present, draw_line, find_components, paste

# concepts:
# growing, connecting

# description:
# In the input you will see two pixels of different colors aligned horizontally or vertically.
# To make the output, you need to connect the two pixels with two lines and a hollow rectangle of size 4x5 in the middle.
# Half of the rectangle is colored with the color of the one side's pixel, and the other half with the color of the other side's pixel.

def main(input_grid):
    """
    Connect two aligned colored pixels through a centered split-color hollow rectangle.

    Input structure:
    - Two single pixels of different colors are aligned horizontally or vertically.
    - The background is black.
    - There is enough space between the pixels for the hollow rectangle bridge.

    Rule:
    - Rotate vertical cases into a horizontal canonical form.
    - Build a hollow rectangle whose left half uses the left color and right half uses the right color.
    - Paste it midway between the pixels and connect each pixel to the nearest side.

    Key insight:
    - After canonical rotation, the task is a symmetric bridge between left and right color anchors.
    """
    # Plan (from the original seed solver):
    # 1. Parse the input
    # 2. Canonicalize the input: because it could be horizontal or vertical, rotate to make horizontal (we will rotate back at the end)
    # 3. Prepare a 4x5 rectangle sprite whose left half is left_color and right half is right_color
    # 4. Place the rectangle in the middle of the two pixels
    # 5. Draw lines to connect the original two pixels with the rectangle
    # 6. Rotate the grid back if it was not originally horizontal
    #
    # Implementation note:
    # - The legacy seed solver runs under a transpose wrapper; this migrated version runs canonical `(r,c)`.
    #   The high-level structure is preserved (canonicalize -> build sprite -> paste -> connect -> unrotate).
    comps = find_components(
        input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict"
    )
    if len(comps) != 2:
        return input_grid.copy()

    p0 = tuple(map(int, comps[0]["coords"][0]))
    p1 = tuple(map(int, comps[1]["coords"][0]))

    rotated = False
    grid = input_grid
    if p0[0] != p1[0]:  # not same row => rotate to make them horizontal
        grid = np.rot90(grid)
        rotated = True

    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != 2:
        out = grid.copy()
        return np.rot90(out, k=-1) if rotated else out

    p0 = tuple(map(int, comps[0]["coords"][0]))
    p1 = tuple(map(int, comps[1]["coords"][0]))
    c0 = int(comps[0]["color"])
    c1 = int(comps[1]["color"])

    # Should now be horizontally aligned
    if p0[0] != p1[0]:
        out = grid.copy()
        return np.rot90(out, k=-1) if rotated else out

    # Sort left->right by column
    if p0[1] <= p1[1]:
        left_pos, right_pos = p0, p1
        left_color, right_color = c0, c1
    else:
        left_pos, right_pos = p1, p0
        left_color, right_color = c1, c0

    r = left_pos[0]
    c_left = left_pos[1]
    c_right = right_pos[1]

    output_grid = grid.copy()

    # Hollow rectangle: 4x5 (width x height) in the seed coordinate system.
    # In canonical (r,c) grids, that's height=5 rows, width=4 cols.
    rect_h, rect_w = 5, 4
    rect = np.full((rect_h, rect_w), Color.BLACK, dtype=int)
    rect[:, 0] = left_color
    rect[:, -1] = right_color
    rect[0, : rect_w // 2] = left_color
    rect[0, rect_w // 2 :] = right_color
    rect[-1, : rect_w // 2] = left_color
    rect[-1, rect_w // 2 :] = right_color

    middle_c = (c_left + c_right + 1) // 2
    paste(
        output_grid,
        rect,
        r - rect_h // 2,
        middle_c - rect_w // 2,
        transparent=Color.BLACK,
        clip=True,
    )

    # Connect left/right pixels to the rectangle via horizontal lines
    draw_line(
        grid=output_grid,
        r=r,
        c=c_left + 1,
        direction=(0, 1),
        color=left_color,
        stop_at=Color.NOT_BLACK,
    )
    draw_line(
        grid=output_grid,
        r=r,
        c=c_right - 1,
        direction=(0, -1),
        color=right_color,
        stop_at=Color.NOT_BLACK,
    )

    if rotated:
        output_grid = np.rot90(output_grid, k=-1)
    return output_grid

def generate_input():
    # Generate a random background grid
    # Make sure the grid's width is greater than its height
    width, height = random.randint(15, 20), random.randint(10, 15)
    grid = np.zeros((width, height), dtype=int)

    # Get the color of two pixels
    colors = np.random.choice(Color.NOT_BLACK, 2, replace=False)
    color1, color2 = colors

    # Place the two pixels on the grid horizontally (randomly rotate at the end to get a variety of orientations)
    x1 = np.random.randint(0, width)
    x2 = np.random.randint(0, width)
    c = np.random.randint(0, height)

    # Ensure there is enough horizontal distance to place the 4x5 rectangle
    distance = abs(x2 - x1)
    rectangle_width, rectangle_height = 4, 5
    if distance < rectangle_width+1: return generate_input()

    # Ensure the remaining distance is odd, so that we can split the pixel colors in half between them
    if (distance - rectangle_width) % 2 != 1: return generate_input()

    # Ensure there is enough vertical space to place the rectangle
    if c < rectangle_height // 2 or c + rectangle_height // 2 >= height: return generate_input()    

    # Place the two pixels on the grid
    grid[x1, c] = color1
    grid[x2, c] = color2

    # Randomly rotate the whole grid
    if np.random.rand() < 0.5:
        grid = np.rot90(grid, k=-1)

    return grid
