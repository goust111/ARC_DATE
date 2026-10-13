import numpy as np

import random
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components, get_bg, paste, random_free_location

# concepts:
# non-black background, occlusion

# description:
# In the input you will see a non-black background (teal background) and the outlines of 5x5 blue rectangles
# To make the output, draw pink horizontal/vertical bars at the center of each rectangle. The bars should be underneath the rectangles, and they should reach the edges of the canvas.

def main(input_grid):
    """
    Draw full-length pink crossbars through the center of each blue outline rectangle, behind the rectangles.

    Input structure:
    - The background color is teal.
    - One or more separated blue hollow rectangles sit on the background.
    - The rectangles are the foreground objects to preserve above the new bars.

    Rule:
    - Detect the blue rectangle outline objects against the teal background.
    - Draw a pink full row and full column through each rectangle center.
    - Redraw the original rectangles over the bars.

    Key insight:
    - The bars are canvas-spanning centerlines, and the original rectangle outlines occlude
      the bars where they overlap.
    """
    # Plan:
    # 1. Find the background color; check that it is teal
    # 2. Find the rectangles
    # 3. Draw the pink bars
    # 4. Ensure the rectangles are on top of the bars by drawing the rectangles last

    # The background is the most common color
    background = get_bg(input_grid)
    assert background == Color.TEAL

    # Extract the rectangle outline objects (full-grid objects).
    objects = find_components(input_grid, bg=background, connectivity=4, monochromatic=True, return_format="objects")

    # Each object gets pink bars at its center, but these are going to be drawn over the object, which we have to undo later by redrawing the objects
    out = input_grid.copy()
    for obj in objects:
        b = bbox(obj != background)
        if b is None:
            continue
        center_r, center_c = bbox_anchor(b, anchor="center")
        out[center_r, :] = Color.PINK
        out[:, center_c] = Color.PINK
    
    # Redraw the objects
    for obj in objects:
        out = paste(out, obj, 0, 0, transparent=background)
    
    return out

def generate_input():
    background_color, rectangle_color = Color.TEAL, Color.BLUE
    width, height = np.random.randint(10, 20, size=2)
    input_grid = np.full((width, height), fill_value=background_color)

    n_rectangles = np.random.randint(1, 3)
    for _ in range(n_rectangles):
        # Create a rectangle and then hollow it out by filling its inside with the background color
        rectangle_sprite = np.full((5, 5), rectangle_color)
        rectangle_sprite[1:-1, 1:-1] = background_color
        # Place the rectangle randomly, taking care though that it does not touch or overlap any other rectangles
        try:
            r, c = random_free_location(input_grid, rectangle_sprite, transparent=background_color, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(input_grid, rectangle_sprite, r, c, transparent=background_color, clip=True)

    return input_grid

# ============= remove below this point for prompting =============
