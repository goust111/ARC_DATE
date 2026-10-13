
import random

import numpy as np
from typing import *
from arc_utils import Color, bbox, crop_bbox, detect_objects, draw_line, paste

# concepts:
# color stripe, fill the square in order

# description:
# In the input you will see several vertical or horizontal stripes of different colors and a gray square with length equal the number of the stripes.
# To make the output, fill the gray square with the colors of the stripes in the order and direction they appear.

def main(input_grid):
    """
    Fill the gray square with stripe colors in the order and orientation they appear.

    Input structure:
    - The grid contains colored stripes and one gray square.
    - The gray square side length matches the number of stripe color entries.
    - Stripes are all vertical or all horizontal.

    Rule:
    - Extract the gray square as the output canvas.
    - Sort non-gray stripes by column for vertical stripes or row for horizontal stripes.
    - Fill each output column or row with the corresponding stripe color in order.

    Key insight:
    - Stripe orientation determines whether the ordered colors become output columns or rows.
    """
    # Find the gray square in the input grid and get its position and size
    gray_rectangle = detect_objects(grid=input_grid, colors=[Color.GRAY], monochromatic=True, connectivity=4)[0]
    _b = bbox(gray_rectangle != Color.BLACK)
    assert _b is not None
    output_grid = crop_bbox(gray_rectangle, _b)

    # Get the color lines in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8)
    lines = []
    for obj in objects:
        # Get the position and size of the color line
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        r0, c0, r1, c1 = _b
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        color = obj[r0, c0]
        if color != Color.GRAY:
            lines.append({"r": r0, "c": c0, "color": int(color), "h": h, "w": w})
    
    # Check if the stripes are horizontal or vertical
    stripes_are_vertical = all(line["w"] == 1 for line in lines)

    # Sort the color lines by their position
    if stripes_are_vertical:
        lines = sorted(lines, key=lambda d: d["c"])
    else:
        lines = sorted(lines, key=lambda d: d["r"])
    
    # Get the direction of the stripes
    direction = (1, 0) if stripes_are_vertical else (0, 1)

    # Find the colors lines in the input grid in order
    colors_in_square = [
        d["color"] for i, d in enumerate(lines) if i == 0 or d["color"] != lines[i - 1]["color"]
    ]

    # Fill the gray square with the colors of the stripes in the order and direction they appear
    for i in range(len(output_grid)):
        if stripes_are_vertical:
            # Fill column i top-to-bottom
            draw_line(grid=output_grid, r=0, c=i, direction=direction, color=colors_in_square[i])
        else:
            # Fill row i left-to-right
            draw_line(grid=output_grid, r=i, c=0, direction=direction, color=colors_in_square[i])
            
    return output_grid

def generate_input():
    # Initialize the grid, make sure the length and width are not too different
    n, m = np.random.randint(9, 24), np.random.randint(9, 24)
    while n > 1.5 * m or m > 1.5 * n:
        n, m = np.random.randint(9, 24), np.random.randint(9, 24)
    grid = np.zeros((m, n), dtype=int)

    # Set stripe's interval
    STRIPE_INTERVAL = 3

    # Randomly choose the starting position of the stripe
    start_from = np.random.randint(0, STRIPE_INTERVAL)

    # Calculate the number of horizontal stripes given the start position, direction and interval
    num_lines = (m - start_from + STRIPE_INTERVAL - 1) // STRIPE_INTERVAL
    
    # Randomly choose the colors of the stripes, each stripe has a different color
    avaliable_colors = [c for c in Color.NOT_BLACK if c != Color.GRAY]
    colors = random.sample(avaliable_colors, num_lines)

    # Draw horizontal stripes with different colors (rows spaced by STRIPE_INTERVAL).
    for i in range(num_lines):
        rr = i * STRIPE_INTERVAL + start_from
        draw_line(grid=grid, r=rr, c=0, direction=(0, 1), color=colors[i])

    # Draw the gray square and its black border, the length of the square is the same as the number of the stripes
    square_r, square_c = np.random.randint(0, m - (num_lines + 1)), np.random.randint(0, n - (num_lines + 1))
    black_border = np.full((num_lines + 2, num_lines + 2), Color.BLACK)
    gray_object = np.full((num_lines, num_lines), Color.GRAY)
    grid = paste(grid, black_border, square_r, square_c, transparent=Color.GRAY, clip=True)
    grid = paste(grid, gray_object, square_r + 1, square_c + 1, transparent=Color.BLACK, clip=True)

    # Randomly rotate the grid
    grid = np.rot90(grid, -1) if np.random.choice([True, False]) else grid

    return grid
