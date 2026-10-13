
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, collision, colors_present, draw_line, find_components, paste, random_free_location, translate

# concepts:
# collision, translation

# description:
# In the input you will see a red object overlaid on a track of green dots.
# To make the output, move the red object one green dot to the right (if the track is horizontal) or one green dot down (if the track is vertical).

def main(input_grid):
    """
    Move the red object forward one slot along the green-dot track.

    Input structure:
    - A red object overlaps one position on a straight track of green dots.
    - The green dots are evenly spaced in a row or column.
    - The background is black.

    Rule:
    - Detect the red object and all green dot objects.
    - Determine the track direction from the green dot centers.
    - Redraw the green dots, then translate the red object along the track to the next non-colliding slot.

    Key insight:
    - The green dots are discrete alignment slots, and the first collision-free translation
      places the red object one step forward on that track.
    """
    # Plan:
    # 1. Detect the objects
    # 2. Determine the orientation of the track of green dots
    # 3. Move in the appropriate direction until it perfectly fits over the next green dot, meaning there are no collisions

    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")

    red_objects = [ obj for obj in objects if Color.RED in colors_present(obj, exclude={ Color.BLACK }) ]
    green_objects = [ obj for obj in objects if Color.GREEN in colors_present(obj, exclude={ Color.BLACK }) ]

    assert len(red_objects) == 1, "There should be exactly one red object"
    assert len(green_objects) >= 1, "There should be at least one green object"

    red_object = red_objects[0]

    # Determine the orientation of the track of green dots by comparing the positions of two dots
    x1, y1 = min(bbox_anchor(bbox(obj != Color.BLACK), anchor="center") for obj in green_objects)
    x2, y2 = max(bbox_anchor(bbox(obj != Color.BLACK), anchor="center") for obj in green_objects)
    if x1 == x2:
        # vertical track
        dr, dc = 0, 1
    elif y1 == y2:
        # horizontal track
        dr, dc = 1, 0
    
    # Make the output grid: Start with all the greens, then put the red in the right spot by moving it one-by-one
    output_grid = np.full_like(input_grid, Color.BLACK)
    for green_object in green_objects:
        paste(output_grid, green_object, 0, 0, transparent=Color.BLACK, clip=True)

    for distance in range(1, 100):
        translated_red_object = translate(red_object, dr*distance, dc*distance)
        if not collision(object1=translated_red_object, object2=output_grid):
            paste(output_grid, translated_red_object, 0, 0, transparent=Color.BLACK, clip=True)
            break

    return output_grid

def generate_input():
    # Make a grid with a red rectangle on the bottom and a teal diagonal line at the top pointing at it
    # Then randomly rotate to get a variety of orientations

    width, height = np.random.randint(15, 25), np.random.randint(15, 25)
    grid = np.full((width, height), Color.BLACK)

    # the red object will be a rectangle with a hole in the middle that the green dot fits into
    # for the hole to fit perfectly in the middle, it needs to be of odd dimension
    rectangle_width, rectangle_height = random.choice([3, 5]), random.choice([3, 5])
    red_rectangle = np.full((rectangle_width, rectangle_height), Color.RED)

    # randomly place the red rectangle away from the edge
    r, c = random_free_location(grid, red_rectangle, transparent=Color.BLACK, border=2, padding=0, padding_connectivity=8)
    paste(grid, red_rectangle, r, c, transparent=Color.BLACK, clip=True)

    # Create a row of green dots, starting with the center of the rectangle
    _b = bbox(grid != Color.BLACK)
    assert _b is not None
    center_r, center_c = bbox_anchor(_b, anchor="center")
    grid[center_r, center_c] = Color.GREEN

    # create a horizontal row of green dots
    for i in range(-10, 10):
        dr = rectangle_width
        if 0 <= center_r + i*dr < width:
            grid[center_r + i*dr, center_c] = Color.GREEN
    
    # randomly flip/rotate
    grid = np.rot90(grid,-(np.random.randint(0, 4)))

    return grid
