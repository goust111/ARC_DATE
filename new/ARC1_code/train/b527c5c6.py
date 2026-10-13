
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, draw_line, find_components, first_free_location, paste

# concepts:
# growing

# description:
# In the input you will see a grid with some green rectangles. Each rectangle has one red pixel on one of its borders.
# To make the output, you should extend the green region of each rectangle to the border of the grid in the direction of the red pixel, with the extent of the line increasing with the extent of the rectangle.
# Also you should draw a red line from the red pixel to the border of the grid.


def main(input_grid):
    """
    Grow each green rectangle outward from its red border indicator to the canvas edge.

    Input structure:
    - Each foreground object is a green rectangle with one red pixel on its border.
    - Objects sit on a black background.
    - The red pixel indicates the side from which growth should extend.

    Rule:
    - Find each rectangle-and-indicator component.
    - Extend a green band from the rectangle toward the canvas edge on the indicator side.
    - Draw a red line from the indicator pixel to that same edge.

    Key insight:
    - The indicator side controls direction, while the rectangle thickness controls the grown band.
    """
    # Initialize the output grid
    output_grid = np.copy(input_grid)
    
    rectangle_color = Color.GREEN
    indicator_color = Color.RED
    background = Color.BLACK

    # get all the green rectangles on the grid. because of the red pixel on the border of the rectangle, they are not monochromatic.
    objects = find_components(input_grid, bg=background, connectivity=4, monochromatic=False, return_format="objects")
    for obj in objects:
        # find the red indicator
        for red_r, red_c in np.argwhere(obj == indicator_color):
            break

        # Get the dimensions of the object, and its position
        _b = bbox(obj != background)
        assert _b is not None
        r, c, _r1, _c1 = _b
        width = _r1 - r + 1
        height = _c1 - c + 1

        # depending on which side of the rectangle that the indicator is on, we draw in different directions
        # left edge: draw to the left
        if red_r == r:
            # Extend the green rectangle to the left until it reaches the border
            output_grid[0:red_r, red_c-(width-1) : red_c+(width-1)+1] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, r=red_r, c=red_c, direction=(-1, 0), color=indicator_color)
        elif red_r == r + width - 1:
            # Extend the green rectangle to the right until it reaches the border
            output_grid[red_r:, red_c-(width-1) : red_c+(width-1)+1] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, r=red_r, c=red_c, direction=(1, 0), color=indicator_color)
        elif red_c == c:
            # Extend the green rectangle to the top until it reaches the border
            output_grid[red_r-(height-1) : red_r+(height-1)+1, 0:red_c] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, r=red_r, c=red_c, direction=(0, -1), color=indicator_color)
        elif red_c == c + height - 1:
            # Extend the green rectangle to the bottom until it reaches the border
            output_grid[red_r-(height-1) : red_r+(height-1)+1, red_c:] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, r=red_r, c=red_c, direction=(0, 1), color=indicator_color)
        else:
            assert False, "The red pixel is not on the border of the rectangle"

    return output_grid

def generate_input():
    # Create the background grid
    n, m = np.random.randint(20, 30), np.random.randint(20, 30)
    grid = np.full((m, n), Color.BLACK)

    rectangle_num = 2
    rectangle_color = Color.GREEN
    indicator_color = Color.RED

    # Draw rectangles on the grid
    for i in range(rectangle_num):
        width, height = np.random.randint(3, 20), np.random.randint(3, 20)

        # The width should be twice smaller than the height
        while (width * 2 + 2) >= height:
            width, height = np.random.randint(3, 20), np.random.randint(3, 20)

        green_rectangle = np.full((width, height), rectangle_color)

        # Place one indicator pixel on one random border of the rectangle
        # Spare enough space for the extend line with the same size of width - 1
        r_coord = np.random.choice([0, width - 1])
        c_coord = np.random.choice(range(width + 1, height - (width + 1)))
        green_rectangle[r_coord, c_coord] = indicator_color
        # Randomly rotate the object so that we get a variety of orientations
        if np.random.choice([True, False]):
            green_rectangle = np.rot90(green_rectangle, k=-1)
            r_coord, c_coord = c_coord, r_coord

        # Place the rectangle on the grid
        try:
            _loc = first_free_location(grid, green_rectangle, transparent=Color.BLACK, border=0, padding=2, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        # Check if there is no space for the rectangle
        except:
            return generate_input()
        paste(grid, green_rectangle, r, c, transparent=Color.BLACK, clip=True)

    # Check if the generated grid is valid
    transformed_grid = main(grid)
    objects = find_components(transformed_grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="objects")

    # After transformation, the two objects should not touch each other
    if len(objects) != rectangle_num:
        # Regenerate the grid
        return generate_input()

    return grid
