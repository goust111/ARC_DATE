
import numpy as np
from typing import *
from arc_utils import Color, bbox, crop_bbox, detect_objects, draw_line, first_free_location, paste, random_sprite
import random

# concepts:
# pattern recognition, rotation, color correspondence, pattern reconstruction

# description:
# In the input you will see one or two color pattern with a red or green pixel as an indicator
# and a set of red and green pixels. 
# To make the output, you should reconstruct the pattern with the red and green pixels 
# indicates the pattern's position. If the indicator is red, the pattern should be flipped by x-axis before reconstructing.

def main(input_grid):
    """
    Reconstruct each indicated pattern at matching isolated indicator pixels, mirroring red-indicated patterns first.

    Input structure:
    - The grid contains one or more multi-color patterns on a black background.
    - Each pattern contains a red or green indicator pixel.
    - Isolated red and green pixels elsewhere mark reconstruction targets.

    Rule:
    - Separate larger pattern sprites from isolated indicator pixels.
    - Mirror a pattern left-right when its embedded indicator is red.
    - Paste each pattern so its indicator aligns with every isolated pixel of the same color.

    Key insight:
    - The indicator color pairs each pattern with its destinations, and red additionally
      signals the mirror operation before stamping.
    """
    bg = Color.BLACK
    output_grid = input_grid.copy()

    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8)
    pixels: list[np.ndarray] = []
    patterns: list[np.ndarray] = []

    for obj in objects:
        b = bbox(obj != bg)
        if b is None:
            continue
        spr = crop_bbox(obj, b)
        if spr.shape == (1, 1):
            pixels.append(obj)
        else:
            patterns.append(spr)

    for pattern in patterns:
        if bool((pattern == Color.RED).any()):
            indicator_color = Color.RED
            # Legacy seeds ran in transpose-mode; in canonical (r,c) this ends up
            # being a left-right mirror when the indicator is RED.
            pattern2 = np.fliplr(pattern)
        else:
            indicator_color = Color.GREEN
            pattern2 = pattern

        ind = np.argwhere(pattern2 == int(indicator_color))
        if ind.size == 0:
            continue
        rel_r, rel_c = map(int, ind[0])

        for pix in pixels:
            if bool((pix == Color.RED).any()):
                pix_color = Color.RED
            elif bool((pix == Color.GREEN).any()):
                pix_color = Color.GREEN
            else:
                continue

            if int(pix_color) != int(indicator_color):
                continue

            pos = np.argwhere(pix == int(pix_color))
            if pos.size == 0:
                continue
            pr, pc = map(int, pos[0])

            r0 = pr - rel_r
            c0 = pc - rel_c
            paste(output_grid, pattern2, r0, c0, transparent=bg, clip=True)

    return output_grid

def generate_input():
    # Initialize the grid
    n, m = 13, 13
    grid = np.zeros((m, n), dtype=int)

    # The indicator color is red or green, the pattern color is blue, yellow or teal
    number_object = np.random.randint(1, 3)
    indicator_color = [Color.RED, Color.GREEN]
    available_color = [Color.BLUE, Color.YELLOW, Color.TEAL]

    # mask color: dummy color used to indicate where we aren't allowed to put something
    # (choose a color distinct from the indicator + pattern palette)
    mask_color = random.choice([Color.GREY, Color.PINK, Color.ORANGE, Color.MAROON])

    # Randomly shuffle the indicator color and available color
    random.shuffle(indicator_color)
    random.shuffle(available_color)


    # There are 1 or 2 object pattern need to be reconstructed
    for i in range(number_object):
        # Generate the object pattern need to be reconstructed, use mask_color to represent the object pattern
        object_pattern = random_sprite(m=3, n=3, color_palette=[mask_color], density=0.5, symmetry="not_symmetric")

        # There are 1 or 2 object pattern need to be reconstructed
        num_sample = np.random.randint(1, 3)

        # Place the indicator in the object pattern
        position_r, position_c = np.random.randint(0, 3), np.random.randint(0, 3)
        while object_pattern[position_r, position_c] != mask_color:
            position_r, position_c = np.random.randint(0, 3), np.random.randint(0, 3)

        # Randomly place the object pattern in the grid
        for _ in range(num_sample):
            object_pattern[position_r, position_c] = indicator_color[i]
            _loc = first_free_location(grid, object_pattern, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
            if _loc is None:
                return generate_input()
            r, c = _loc
            grid = paste(grid, object_pattern, r, c, transparent=Color.BLACK, clip=True)
        
        # Replace the gray color with the available color as the original pattern for reconstruction
        showed_pattern = object_pattern.copy()
        showed_pattern[showed_pattern == mask_color] = available_color[i]

        # If the indicator color is red, flip the pattern by x-axis
        if indicator_color[i] == Color.RED:
            showed_pattern = np.fliplr(showed_pattern)
        
        # Place the original pattern in the grid
        _loc = first_free_location(grid, showed_pattern, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            return generate_input()
        r, c = _loc
        grid = paste(grid, showed_pattern, r, c, transparent=Color.BLACK, clip=True)
    
    grid[grid == mask_color] = Color.BLACK

    return grid
