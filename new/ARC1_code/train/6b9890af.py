import numpy as np
from arc_utils import Color, bbox, colors_present, crop_bbox, detect_objects, draw_line, paste, random_free_location, random_sprite, scale_sprite
import random


# concepts:
# object detection, scaling

# description:
# In the input you will see a 3x3 object and a red square n times larger than the 3x3 object.
# To make the output, you should scale the 3x3 object to the size of the red square and place it in the red square.
# Return just the red square (with the rescaled object put into it)

def main(input_grid):
    """
    Scale a small pattern to fill the interior of the red square frame and return that frame.

    Input structure:
    - The input is a black grid with a red square frame.
    - A separate non-red 3x3 pattern object appears elsewhere.
    - The frame interior is an integer scale of the small pattern.

    Rule:
    - Detect the red frame object and the separate pattern object.
    - Compute the scale from the red frame's interior size.
    - Scale the pattern and paste it inside the frame, returning only the framed result.

    Key insight:
    - The red square supplies both the target scale and the output extent.
    """
    bg = Color.BLACK

    # Detect the red frame square and the 3x3 pattern square as separate objects.
    objects = detect_objects(input_grid, monochromatic=True, connectivity=8, background=bg)

    outer_sprite = None
    inner_sprite = None
    for obj in objects:
        b = bbox(obj != bg)
        if b is None:
            continue
        sprite = crop_bbox(obj, b)
        if Color.RED in colors_present(sprite, exclude={bg}):
            outer_sprite = sprite
        else:
            inner_sprite = sprite

    assert outer_sprite is not None and inner_sprite is not None

    # Calculate the scaling factor.
    # Subtract 2 because the red frame has a 1-pixel border on each side.
    scale = (outer_sprite.shape[0] - 2) // inner_sprite.shape[0]
    scaled_inner_sprite = scale_sprite(inner_sprite, scale)

    # Render just the red frame square (with the scaled inner sprite inside it).
    output_grid = np.full_like(outer_sprite, bg)
    output_grid = paste(output_grid, outer_sprite, 0, 0)
    output_grid = paste(output_grid, scaled_inner_sprite, 1, 1, transparent=bg)
    return output_grid

def generate_input():
    # Initialize the grid
    n, m = np.random.randint(20, 30), np.random.randint(20, 30)
    grid = np.zeros((m, n), dtype=int)    

    # Set the pattern size and scaling range
    small_object_size = 3
    availabe_scales = range(1, 5)
    scale = np.random.choice(availabe_scales)

    # Generate the red square frame scale times larger than the pattern square and randomly place it
    big_red_rectangle = np.full((scale * small_object_size + 2, scale * small_object_size + 2), Color.RED)
    big_red_rectangle[1:-1, 1:-1] = Color.BLACK
    r_square, c_square = random_free_location(grid, big_red_rectangle, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    grid = paste(grid, big_red_rectangle, r_square, c_square, transparent=Color.BLACK, clip=True)

    # Generate the random pattern square with pattern_size
    availabe_color = [c for c in Color.NOT_BLACK if c != Color.RED]
    random_color = np.random.choice(availabe_color)

    # Randomly place the pattern square in the area without the red square
    sprite = random_sprite(m=small_object_size, n=small_object_size, color_palette=[random_color], density=0.5)
    r, cc = random_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    grid = paste(grid, sprite, r, cc, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
