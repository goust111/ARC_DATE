import numpy as np
import random
from arc_utils import Color, bbox, contact, crop_bbox, detect_objects, draw_line, first_free_location, paste, random_sprite


# concepts:
# object extraction, contact, cropping

# description:
# In the input you will see several objects with same color placed in a 10x10 grid, only one of 
# them contact a gray pixel.
# To make the output grid, you should select the object contact the gray pixel, crop it, and then output it.

def main(input_grid):
    """
    Select and output the object that touches the gray indicator pixel.

    Input structure:
    - Several same-colored objects sit on a black background.
    - A single gray pixel touches exactly one of those objects.
    - The object color is the only non-black, non-gray color in the grid.

    Rule:
    - Detect all objects of the shared non-gray color.
    - Detect the gray indicator pixel.
    - Return the cropped object that has 4-connected contact with the indicator.

    Key insight:
    - The gray pixel is an external selector and is not included in the returned pattern.
    """
    # Get the color of the pattern
    pattern_color = [color for color in np.unique(input_grid) if color != Color.BLACK and color != Color.GRAY][0]

    # Detect all the patterns with pattern color in the input grid
    pattern_list = detect_objects(grid=input_grid, colors=[pattern_color], connectivity=8, monochromatic=True, background=Color.BLACK)

    # Detect the indicator gray pixel
    gray_pixel = detect_objects(grid=input_grid, colors=[Color.GRAY], connectivity=8, monochromatic=True, background=Color.BLACK)[0]

    # Find out which pattern has contact the gray pixel
    for pattern in pattern_list:
        b = bbox(pattern != Color.BLACK)
        assert b is not None
        cropped_pattern = crop_bbox(pattern, b)
        # Check if the gray pixel contact the pattern
        if contact(object1=pattern, object2=gray_pixel, connectivity=4, background=Color.BLACK):
            # Crop the pattern and output it
            output_grid = cropped_pattern
            break

    return output_grid

def generate_input():
    # Generate a 10x10 grid with several objects with same color placed in it
    n, m = 10, 10
    grid = np.zeros((m, n), dtype=int)
    
    # Select a color for the objects and the number of objects.
    available_colors = [c for c in Color.NOT_BLACK if c != Color.GRAY]
    obj_color = random.choice(available_colors)
    num_sprite = np.random.randint(2,4)

    # Place the objects in the grid.
    for i in range(num_sprite):
        # Create a 3x3 object with the selected color and random pattern.
        sprite = random_sprite(m=3, n=3, color_palette=[obj_color], density=0.4)
        # Place the object in the grid.
        try:
            _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, cc = _loc
        except:
            continue
        # Add a gray pixel to contact one of the object.
        if i == 0:
            grid[r + 1, cc - 1] = Color.GRAY
            # Make sure the grey pixel contact the object.
            grid[r + 1, cc] = obj_color
        # Place the object in the grid.
        grid = paste(grid, sprite, r, cc, transparent=Color.BLACK, clip=True)
            
    return grid
# ============= remove below this point for prompting =============
