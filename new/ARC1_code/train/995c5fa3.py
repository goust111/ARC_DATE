
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, detect_objects, draw_line, paste

# concepts:
# pattern matching, color correspondence

# description:
# In the input you will see three different 4x4 patterns of gray pixels place horizonlly and seperate by black interval. 
# To make the output grid, you should find out each pattern corresponds to a color: red, teal, yellow, or green, 
# and color the corresponding row in the output grid with the corresponding color in the order from left to right.

def main(input_grid):
    """
    Decode three fixed gray glyphs into colored rows ordered from left to right.

    Input structure:
    - The input is three 4x4 gray/black glyphs separated horizontally by black columns.
    - Each glyph is one of four known black-mask patterns.
    - The left-to-right glyph order determines the top-to-bottom output row order.

    Rule:
    - Crop and sort the three glyph components by column position.
    - Map each glyph pattern to its associated color: red, teal, yellow, or green.
    - Create a square output with one solid row per decoded glyph color.

    Key insight:
    - The position of black cells within each 4x4 gray block is a symbolic code for an output color.
    """
    # Distinguish four different pattern with different black pixels placing on the gray background.
    b, g = Color.BLACK, Color.GRAY
    red_pattern = np.array([[g, g, g, g], [g, g, g, g], [g, g, g, g], [g, g, g, g]])
    teal_patten = np.array([[g, g, g, g], [g, b, b, g], [g, b, b, g], [g, g, g, g]])
    yellow_pattern = np.array([[g, g, g, g], [g, g, g, g], [g, b, b, g], [g, b, b, g]])
    green_pattern = np.array([[g, g, g, g], [b, g, g, b], [b, g, g, b], [g, g, g, g]])

    # Detect the patterns of gray pixels with size 4x4 place horizonlly and seperate by black interval.
    detect_patterns = detect_objects(grid=input_grid, colors=[Color.GRAY], connectivity=8, monochromatic=True)

    # Get the bounding box of each pattern and crop the pattern.
    pattern_lists = []
    for pattern in detect_patterns:
        _b = bbox(pattern != Color.BLACK)
        assert _b is not None
        r, c, _r1, _c1 = _b  # (r0,c0, r1,c1) in canonical coords
        pattern_shape = crop_bbox(pattern, _b)
        pattern_lists.append({'r0': int(r), 'c0': int(c), 'pattern': pattern_shape})
    # Patterns are laid out left-to-right -> sort by column.
    pattern_lists = sorted(pattern_lists, key=lambda rec: rec['c0'])

    # Find the corresponding color of each pattern from left to right.
    color_list = []
    for pattern in pattern_lists:
        cur_pattern = pattern['pattern']
        if np.array_equal(cur_pattern, red_pattern):
            color_list.append(Color.RED)
        elif np.array_equal(cur_pattern, teal_patten):
            color_list.append(Color.TEAL)
        elif np.array_equal(cur_pattern, yellow_pattern):
            color_list.append(Color.YELLOW)
        elif np.array_equal(cur_pattern, green_pattern):
            color_list.append(Color.GREEN)
        else:
            raise ValueError("Invalid pattern")
    square_number = len(color_list)

    # Color the corresponding row in the output grid with the corresponding color in order.
    output_grid = np.zeros((square_number,square_number), dtype=int)
    for cnt, color in enumerate(color_list):
        draw_line(grid=output_grid, color=color, r=cnt, c=0, direction=(0, 1), length=None)
    return output_grid

def generate_input():
    # There are three patterns of gray pixels with size 4x4 place horizonlly and seperate by black interval.
    square_length = 4
    square_number = 3

    # Create four different pattern by placing black pixels on the gray background.
    b, g = Color.BLACK, Color.GRAY
    red_pattern = np.array([[g, g, g, g], [g, g, g, g], [g, g, g, g], [g, g, g, g]], dtype=int)
    teal_patten = np.array([[g, g, g, g], [g, b, b, g], [g, b, b, g], [g, g, g, g]], dtype=int)
    yellow_pattern = np.array([[g, g, g, g], [g, g, g, g], [g, b, b, g], [g, b, b, g]], dtype=int)
    green_pattern = np.array([[g, g, g, g], [b, g, g, b], [b, g, g, b], [g, g, g, g]], dtype=int)
    
    # Calulate the size of the input grid.
    n = square_number * square_length + square_number - 1
    m = square_length

    grid = np.zeros((m, n), dtype=int)

    # Assign the color of each pattern.
    available_colors = [Color.RED, Color.TEAL, Color.YELLOW, Color.GREEN]

    # Randomly choose the three patterns represented by the given colors on the input grid.
    color_list = np.random.choice(available_colors, size=square_number, replace=False)
    
    # Place the corresponding pattern on the input grid.
    for square_num, color in enumerate(color_list):
        r = 0
        c = square_num * (square_length + 1)
        # Each pattern corresponds to a diffferent color: red, teal, yellow, or green.
        if color == Color.RED:
            grid = paste(grid, red_pattern, r, c, transparent=Color.BLACK, clip=True)
        elif color == Color.TEAL:
            grid = paste(grid, teal_patten, r, c, transparent=Color.BLACK, clip=True)
        elif color == Color.YELLOW:
            grid = paste(grid, yellow_pattern, r, c, transparent=Color.BLACK, clip=True)
        elif color == Color.GREEN:
            grid = paste(grid, green_pattern, r, c, transparent=Color.BLACK, clip=True)

    return grid
