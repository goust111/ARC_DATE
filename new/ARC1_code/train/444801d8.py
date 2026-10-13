import numpy as np
import random
from arc_utils import Color, bbox, colors_present, detect_objects, draw_line, paste, random_free_location, random_sprite


# concepts:
# cups, filling

# description:
# In the input you will see several blue "cups", meaning an almost-enclosed shape with a small opening at the top, and empty space (black pixels) inside, as well as a single colored pixel inside.
# To make the output grid, you should fill the interior of each cup with the same color as the colored pixel inside it. 
# Also, put a single layer of colored pixels above the cup with the same color as what's inside.

def main(input_grid):
    """
    Fill each blue cup with its internal marker color and add a matching cap layer above it.

    Input structure:
    - The grid contains one or more blue cup-shaped outlines on a black background.
    - Each cup has a black interior with exactly one non-blue colored marker inside.
    - The cup bounding box defines the interior area to fill.

    Rule:
    - Detect each blue cup object.
    - Find the unique non-black, non-blue color inside its bounding box.
    - Fill the cup interior with that color and paint one matching row above the cup.

    Key insight:
    - The marker pixel supplies the fill color, while the blue outline supplies the geometry
      for both the interior and the added top layer.
    """
    # Plan:
    # 1. Detect all the blue cups
    # 2. For each cup, find the mask of what is inside of it
    # 3. Find the color of the single pixel inside the cup
    # 4. Fill the cup with the color
    # 5. Put a single layer of colored pixels above the cup with the same color as what's inside
    
    # Detect all the blue cups
    blue_cups = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4, background=Color.BLACK)

    output_grid = input_grid.copy()

    # For each cup object...
    for obj in blue_cups:
        # Extract what's inside the cup (as its own object), which is everything in the bounding box that is not the object itself
        b = bbox(obj != Color.BLACK)
        assert b is not None
        r0, c0, r1, c1 = b
        inside_cup_mask = np.zeros_like(input_grid, dtype=bool)
        inside_cup_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        inside_cup_mask = inside_cup_mask & (obj != Color.BLUE)
        object_inside_cup = np.where(inside_cup_mask, input_grid, Color.BLACK)

        # Find the color of the single pixel inside the cup
        colors = set(map(int, np.unique(object_inside_cup))) - {Color.BLACK, Color.BLUE}
        assert len(colors) == 1, "There should be exactly one color inside the cup"
        color = next(iter(colors))

        # Fill the cup with the color
        output_grid[inside_cup_mask] = color

        # Put a single layer of colored pixels above the cup with the same color as what's inside
        top_r = r0 - 1
        if top_r >= 0:
            output_grid[top_r, c0 : c1 + 1] = color

    return output_grid

def generate_input():
    # Generate the grid with random size
    width = np.random.randint(12, 30)
    height = np.random.randint(12, 30)
    grid = np.full((width, height), Color.BLACK)

    n_cups = np.random.randint(1, 2+1)

    for cup_index in range(n_cups):
        # Pick a random width/height for this cup
        cup_width = np.random.randint(4, 8)
        cup_height = np.random.randint(3, 8)

        # Make a sprite, which is just going to be a blue outline of a rectangle with a hole at the top
        sprite = np.full((cup_width, cup_height), Color.BLACK)
        sprite[0, :] = Color.BLUE
        sprite[-1, :] = Color.BLUE
        sprite[:, 0] = Color.BLUE
        sprite[:, -1] = Color.BLUE

        # Make the hole centered at the top (variable size)
        hole_left_r = np.random.randint(1, cup_width//2)
        hole_right_r = cup_width - hole_left_r - 1
        hole_c = 0
        sprite[hole_left_r:hole_right_r+1, hole_c] = Color.BLACK

        # Put another color inside the cup
        color = np.random.choice([color for color in Color.NOT_BLACK if color != Color.BLUE])
        r, c = np.random.randint(1, cup_width-1), np.random.randint(1, cup_height-1)
        sprite[r, c] = color

        # Find a random free location for it
        try:
            free_r, free_c = random_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(grid, sprite, free_r, free_c, transparent=Color.BLACK, clip=True)

    return grid    
# ============= remove below this point for prompting =============
