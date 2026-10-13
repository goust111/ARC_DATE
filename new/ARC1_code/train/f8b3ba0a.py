import numpy as np

import random
from arc_utils import Color, draw_line, find_components, paste, random_sprite

# concepts:
# rectangular cells, counting

# description:
# In the input you will see monochromatic rectangles arranged into an array.
# To make the output, find the most common color of the 2x1 rectangles, second most common color, etc.
# Then, make the output a vertical stripe colored with the second common color, then the third most common color, etc., starting at the top, and skipping the most common color.

def main(input_grid):
    """
    Output a vertical list of rectangle colors ordered by frequency after skipping the most common.

    Input structure:
    - The grid contains separated monochromatic rectangle objects on a black background.
    - Rectangles are arranged in an array and colored from several non-black colors.
    - Color frequency is measured by object count, not pixel count.

    Rule:
    - Detect each rectangle object and record its non-black color.
    - Sort colors by how many rectangles have that color.
    - Drop the most frequent color and write the remaining colors top to bottom in a one-column grid.

    Key insight:
    - The output is a rank summary of object colors, so rectangle positions and dimensions
      are discarded after counting.
    """
    # Plan:
    # 1. Extract the objects, arranged into the array
    # 2. Count the colors
    # 3. Sort the colors by count
    # 4. Create the output grid, remembering to skip the most common color

    background = Color.BLACK
    
    objects = find_components(input_grid, bg=background, monochromatic=True, connectivity=4, return_format="objects")

    # Count colors per rectangle object (each object is monochromatic).
    obj_colors = []
    for obj in objects:
        cols = set(map(int, np.unique(obj))) - {background}
        if cols:
            obj_colors.append(next(iter(cols)))

    color_counts = {c: obj_colors.count(c) for c in set(obj_colors)}
    sorted_colors = sorted(color_counts, key=lambda c: color_counts[c], reverse=True)

    # Skip the most common color.
    sorted_colors = sorted_colors[1:]

    # Output is a vertical stripe: one pixel per remaining color.
    output_grid = np.full((len(sorted_colors), 1), background, dtype=int)
    for i, color in enumerate(sorted_colors):
        output_grid[i, 0] = int(color)
    return output_grid




def generate_input():
    # Plan:
    # 1. randomly pick size and possible colors for the rectangles
    # 2. randomly pick how far apart the rectangle should be when they are arranged in an array
    # 3. create the array of rectangles, randomly choosing colors
    # 5. double check that no two colors have the exact same count

    rectangle_dimensions = np.random.randint(1, 4, size=(2))
    number_of_distinct_colors = np.random.randint(3, 6)
    rectangle_colors = np.random.choice(Color.NOT_BLACK, size=(number_of_distinct_colors), replace=False)

    rectangle_spacing = np.random.randint(1, 4, size=(2))
    grid_dimensions = np.random.randint(10, 30, size=(2))

    grid = np.full(grid_dimensions, Color.BLACK)

    possible_x_values = np.arange(rectangle_spacing[0], grid_dimensions[0] - rectangle_dimensions[0], rectangle_spacing[0] + rectangle_dimensions[0])
    possible_y_values = np.arange(rectangle_spacing[1], grid_dimensions[1] - rectangle_dimensions[1], rectangle_spacing[1] + rectangle_dimensions[1])
    for r in possible_x_values:
        for c in possible_y_values:
            color = np.random.choice(rectangle_colors)
            sprite = random_sprite(rectangle_dimensions[0], rectangle_dimensions[1], density=1, color_palette=[color])
            paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return grid

# ============= remove below this point for prompting =============
