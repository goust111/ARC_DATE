
import numpy as np
from typing import *
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite
import random

# concepts:
# symmetry, mirror

# description:
# In the input you will see two patterns close by two frames.
# To make the output, you need to split the inner pattern into two parts by the symmetry of the framework. 
# Then, mirror the two parts according to each frame's line and place them in the output grid.

def main(input_grid):
    """
    Mirror the inner pattern pieces outward across the two red frame halves.

    Input structure:
    - The grid contains two red frame halves enclosing gray pattern pieces.
    - The frame pair may be horizontal or vertical after rotation.
    - The inner pattern lies between the red frame halves.

    Rule:
    - Detect the red frame components and paste them into a blank output.
    - Crop the full inner pattern and split it into two halves along the frame axis.
    - Mirror each half across its corresponding frame line and paste it outside the frame.

    Key insight:
    - The red frame halves are symmetry axes; the enclosed gray pattern is split and
      reflected outward rather than copied in place.
    """
    # Extract the framework
    frame_color = Color.RED

    # Create an empty grid
    n, m = input_grid.shape
    output_grid = np.zeros((n, m), dtype=int)

    # Get two frame that form the framework
    frames = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    frames = [frame for frame in frames if np.any(frame == frame_color)]
    framework = []
    for frame in frames:
        # Get the frame and its position
        _b = bbox(frame != Color.BLACK)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1
        _b = bbox(frame != Color.BLACK)
        assert _b is not None
        cropped_frame = crop_bbox(frame, _b)
        framework.append({"x": r, "y": c, "w": w, "h": h, "frame": cropped_frame})
        # Place the frame in the output grid
        output_grid = paste(output_grid, cropped_frame, r, c, transparent=Color.BLACK, clip=True)

    # Sort the framework by position
    framework = sorted(framework, key=lambda r: r["x"])
    framework = sorted(framework, key=lambda r: r["y"])

    # Get the inner pattern
    _b = bbox(input_grid != Color.BLACK)
    assert _b is not None
    r_whole, c_whole, _r1, _c1 = _b
    w_whole = _r1 - r_whole + 1
    h_whole = _c1 - c_whole + 1
    inner_pattern = input_grid[r_whole + 1 : r_whole + w_whole - 1, c_whole + 1 : c_whole + h_whole - 1]

    # Check if the framework is horizontal or vertical
    if_horizontal = framework[0]["w"] > framework[0]["h"]

    if if_horizontal:
        # Split the inner pattern into two parts by the symmetry of the framework
        pattern_len = inner_pattern.shape[1] // 2
        pattern_part1 = inner_pattern[:, : pattern_len]
        pattern_part2 = inner_pattern[:, pattern_len:]

        # Mirror the two inner patterns according to each frame's line
        pattern_part1 = np.fliplr(pattern_part1)
        pattern_part2 = np.fliplr(pattern_part2)

        # Place the two inner patterns in the output grid
        output_grid = paste(output_grid, pattern_part1, framework[0]['x'] + 1, framework[0]['y'] - pattern_len, transparent=Color.BLACK, clip=True)
        output_grid = paste(output_grid, pattern_part2, framework[1]['x'] + 1, framework[1]['y'] + 2, transparent=Color.BLACK, clip=True)
    else:
        # Split the inner pattern into two parts by the symmetry of the framework
        pattern_len = inner_pattern.shape[0] // 2
        pattern_part1 = inner_pattern[: pattern_len, :]
        pattern_part2 = inner_pattern[pattern_len:, :]

        # Mirror the two inner patterns according to each frame's line
        pattern_part1 = np.flipud(pattern_part1)
        pattern_part2 = np.flipud(pattern_part2)

        # Place the two inner patterns in the output grid
        output_grid = paste(output_grid, pattern_part1, framework[0]['x'] - pattern_len, framework[0]['y'] + 1, transparent=Color.BLACK, clip=True)
        output_grid = paste(output_grid, pattern_part2, framework[1]['x'] + 2, framework[1]['y'] + 1, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Create a 2D background
    n, m = np.random.randint(15, 30), np.random.randint(15, 30)
    grid = np.zeros((m, n), dtype=int)

    # Get a framework for pattern
    frame_length = np.random.randint(5, 10)
    frame_width = 2
    frame_color = Color.RED

    # Draw a half of the framework
    frame = np.zeros((frame_length, frame_width), dtype=int)
    draw_line(grid=frame, r=0, c=0, direction=(0, 1), color=frame_color)
    frame[0][1] = frame_color
    frame[-1][1] = frame_color

    # Select the interval for two half frameworks that form one entire framework
    frame_interval = 4
    
    # Calculate the sizes of two patterns in framwork
    pattern_color = Color.GRAY
    pattern_interval = 1
    pattern_length = frame_length - 2
    pattern_width_1 = frame_interval // 2
    pattern_width_2 = frame_interval - pattern_width_1 - pattern_interval

    # Generate two patterns in framwork
    pattern_1 = random_sprite(m=pattern_length, n=pattern_width_1, color_palette=[pattern_color])
    pattern_2 = random_sprite(m=pattern_length, n=pattern_width_2, color_palette=[pattern_color])

    # place the pattern in the frame
    whole_frame = np.zeros((frame_length, frame_width * 2 + frame_interval), dtype=int)
    # Place the upper half of the frame
    whole_frame = paste(whole_frame, frame, 0, 0, transparent=Color.BLACK, clip=True)
    # Place the upper pattern
    whole_frame = paste(whole_frame, pattern_1, 1, frame_width, transparent=Color.BLACK, clip=True)
    # Place the lower pattern
    whole_frame = paste(whole_frame, pattern_2, 1, frame_width + pattern_width_1 + pattern_interval, transparent=Color.BLACK, clip=True)
    # Place the lower half of the frame
    whole_frame = paste(whole_frame, np.flipud(frame), 0, frame_width + pattern_width_1 + pattern_interval + pattern_width_2, transparent=Color.BLACK, clip=True)

    # place the frame in the background
    _loc = first_free_location(grid, whole_frame, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    grid = paste(grid, whole_frame, r, c, transparent=Color.BLACK, clip=True)

    # Randomly rotate the grid
    grid = np.rot90(grid, k=-(np.random.randint(4)))

    return grid
