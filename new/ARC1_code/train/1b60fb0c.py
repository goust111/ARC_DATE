import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, detect_rotational_symmetry, draw_line, first_free_location, paste, random_sprite


# concepts:
# symmetry

# description:
# In the input you will see an image containing blue pixels that is almost rotationally symmetric, except that it is missing the section either north, south, east, or west that would make it rotationally symmetric
# Color red all the pixels that would need to be colored in order to make the image rotationally symmetric (when rotating clockwise)

def main(input_grid):
    """
    Mark the one-step clockwise rotational completions of a blue shape in red.

    Input structure:
    - The input contains blue pixels forming an almost rotationally symmetric object.
    - One directional section is missing and appears as black.
    - Black is the background and red is reserved for needed completion pixels.

    Rule:
    - Detect the rotational symmetry center of the blue object.
    - Rotate every blue pixel one quarter-turn clockwise.
    - If the rotated location is black, color it red.

    Key insight:
    - The output does not fully recolor the object; it highlights only the missing clockwise
      counterparts required by the rotational pattern.
    """
    # Uses the library function `detect_rotational_symmetry`.
    
    # Plan:
    # 1. Detect the (x,y) point that the object is rotated around
    # 2. Rotate each blue colored pixel around that point. If the rotated pixel is not colored, color it red.

    output_grid = np.copy(input_grid)

    # Find the symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK], background=Color.BLACK)
    
    # Rotate the blues and color red as needed
    blues = np.argwhere(input_grid == Color.BLUE)
    for r, c in blues:
        rr, cc = sym.apply(int(r), int(c), iters=1)
        if input_grid[rr, cc] == Color.BLACK:
            output_grid[rr, cc] = Color.RED
    
    return output_grid

def generate_input():
    # make a black medium large grid
    n, m = np.random.randint(10, 20), np.random.randint(10, 20)
    grid = np.zeros((m, n), dtype=int)

    # make a blue radially symmetric sprite and put it at a random free location
    sprite_size = np.random.randint(8, min(m, n))
    sprite = random_sprite(sprite_size, sprite_size, symmetry='radial', color_palette=[Color.BLUE], density=0.2)
    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc

    # remove a random section of the sprite to make it not radially symmetric
    remove_length = np.random.randint(1, sprite_size//4)
    quadrant = np.random.choice(['north', 'south', 'east', 'west'])
    if quadrant == 'north':
        sprite[sprite_size//2 - remove_length : sprite_size//2 + remove_length, : sprite_size//2] = Color.BLACK
    elif quadrant == 'south':
        sprite[sprite_size//2 - remove_length : sprite_size//2 + remove_length, sprite_size//2 :] = Color.BLACK
    elif quadrant == 'east':
        sprite[: sprite_size//2, sprite_size//2 - remove_length : sprite_size//2 + remove_length] = Color.BLACK
    elif quadrant == 'west':
        sprite[sprite_size//2 :, sprite_size//2 - remove_length : sprite_size//2 + remove_length] = Color.BLACK

    paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
