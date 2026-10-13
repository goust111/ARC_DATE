
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, crop_bbox, draw_line, find_components, first_free_location, paste

# concepts:
# pixel patterns, expansion, color sequence

# description:
# In the input you will see a grid with a cross pattern. Each pixel in the cross has a different color.
# To make the output, you should expand the cross right/left/top/bottom following the original color sequence of the cross.

def main(input_grid):
    """
    Extend a multicolor cross across the canvas by repeating its color sequences.

    Input structure:
    - The grid contains one connected multicolor cross on a black background.
    - One full row and one full column of the cropped sprite meet at the cross center.
    - The row and column colors form the sequences to repeat.

    Rule:
    - Crop the cross and locate the row and column that pass through its center.
    - Copy the input grid as the output canvas.
    - Paste the center column and center row sequences repeatedly in both directions.

    Key insight:
    - The cross is a seed for periodic extension: its central row and column are reused
      at intervals equal to their own lengths.
    """
    # Plan:
    # 1. Parse the input and create output canvas to draw on top of
    # 2. Extract the vertical and horizontal parts of the cross, and make note of the coordinate of the middle
    # 3. Expand the horizontal part to the right and left, aligned with the middle y coordinate
    # 4. Expand the vertical part to the top and bottom, aligned with the middle x coordinate    

    # 1. Input parsing
    # Extract the cross, which has many colors and so is not monochromatic.
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="objects")
    assert len(objects) == 1, "exactly one cross expected"
    obj = objects[0]
    _b = bbox(obj != Color.BLACK)
    assert _b is not None
    cross_r, cross_c = bbox_anchor(_b, anchor="upper left")

    # Create output grid, which we are going to draw on top of, so we start with the input grid
    output_grid = input_grid.copy()
    width, height = input_grid.shape

    # 2. Cross analysis: Extract subsprites, get the middle
    # Extract the horizontal/vertical parts of the cross sprite by figuring out where its middle is (where the horizontal and vertical lines meet)
    _b = bbox(obj != Color.BLACK)
    assert _b is not None
    sprite = crop_bbox(obj, _b)
    cross_width, cross_height = sprite.shape
    # Middle is where they meet
    cross_middle_r = next( r for r in range(cross_width) if np.all(sprite[r, :] != Color.BLACK) )
    cross_middle_c = next( c for c in range(cross_height) if np.all(sprite[:, c] != Color.BLACK) )
    # Extract the horizontal and vertical parts of the cross
    vertical_sprite = sprite[cross_middle_r:cross_middle_r+1, :]
    horizontal_sprite = sprite[:, cross_middle_c:cross_middle_c+1]

    # 3. Expand the horizontal line to the right and left
    r_start, c_start, len_line = cross_r, cross_c + cross_middle_c, cross_width
    for i in range(r_start, width, len_line):
        paste(output_grid, horizontal_sprite, i, c_start, transparent=Color.BLACK, clip=True)
    for i in range(r_start, -(len_line), -len_line):
        paste(output_grid, horizontal_sprite, i, c_start, transparent=Color.BLACK, clip=True)
    
    # 4. Expand the vertical line to the top and bottom
    r_start, c_start, len_line = cross_r + cross_middle_r, cross_c, cross_height
    for i in range(c_start, height, len_line):
        paste(output_grid, vertical_sprite, r_start, i, transparent=Color.BLACK, clip=True)
    for i in range(c_start, -(len_line), -len_line):
        paste(output_grid, vertical_sprite, r_start, i, transparent=Color.BLACK, clip=True)
        
    return output_grid

def generate_input():
    # Generate the background grid
    width, height = np.random.randint(20, 30, size=2)
    grid = np.full((width, height), Color.BLACK)

    # Randomly choose the number of colors
    num_colors = np.random.randint(2, 5)
    colors = np.random.choice(Color.NOT_BLACK, size=num_colors, replace=False)

    # Generate a line with these colors in sequence
    line = np.full((num_colors, 1), Color.BLACK)
    for i in range(num_colors):
        line[i, 0] = colors[i]

    # form a cross pattern randomly
    cross_points = random.randint(0, num_colors - 1)
    sprite = np.full((num_colors, num_colors), Color.BLACK)
    line_t = np.transpose(line)
    paste(sprite, line, 0, cross_points, transparent=Color.BLACK, clip=True)
    paste(sprite, line_t, cross_points, 0, transparent=Color.BLACK, clip=True)

    # Randomly rotate the pattern
    sprite = np.rot90(sprite, k=-(np.random.randint(4)))

    # Randomly place the pattern on the grid
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
