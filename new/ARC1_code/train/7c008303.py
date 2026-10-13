import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, paste, random_sprite


# concepts:
# color correspondence, object splitting

# description:
# In the input you will see a 9x9 grid with a 6x6 green sprite and a 2x2 sprite with 4 different colors separated by two teal lines.
# To make the output grid, you should separate the 6x6 green sprite into 4 3x3 sub-sprites and color them 
# with the 4 different colors in the 2x2 sprite, with the same relative position.

def main(input_grid):
    """
    Use a 2x2 color guide to recolor the four quadrants of a 6x6 green mask.

    Input structure:
    - The input is a 9x9 grid split by one teal row and one teal column.
    - One region contains a 2x2 guide with four non-black, non-green colors.
    - The opposite region contains a 6x6 green sprite on black.

    Rule:
    - Separate the teal-divided regions and identify the color guide and green sprite.
    - Split the green sprite into four equal quadrants.
    - Recolor green pixels in each quadrant with the corresponding guide color in the same relative position.

    Key insight:
    - The teal cross only organizes the two artifacts; the 2x2 guide is a miniature color map for
      the 6x6 sprite's four 3x3 quadrants.
    """
    # Detect four parts seperated by two intersected teal lines.
    sub_grids = find_components(input_grid, bg=Color.TEAL, connectivity=4, monochromatic=False, return_format="objects")

    # Find the green pattern and square with four colors as a coloring guidance.
    for sub_grid in sub_grids:
        b = bbox(sub_grid != Color.TEAL)
        if b is None:
            continue
        cropped_sub_grid = crop_bbox(sub_grid, b)

        # If this part is the color guide, then store the color guide.
        if np.all(cropped_sub_grid != Color.BLACK) and np.all(cropped_sub_grid != Color.GREEN):
            color_guide = cropped_sub_grid

        # If this part is the green pattern, then store the green pattern.s
        elif np.any(cropped_sub_grid == Color.GREEN):
            green_pattern = cropped_sub_grid
    
    # Caculate the size of four sub-sprites on green pattern to be colored.
    height_green, width_green = green_pattern.shape
    height_green_half, width_green_half = height_green // 2, width_green // 2
    
    # Color each sub-sprite on the green pattern follow the color guide: with the color in same relative position.
    green_pattern[0:height_green_half, 0:width_green_half][green_pattern[0:height_green_half, 0:width_green_half] == Color.GREEN] = color_guide[0, 0]
    green_pattern[height_green_half:height_green, 0:width_green_half][green_pattern[height_green_half:height_green, 0:width_green_half] == Color.GREEN] = color_guide[1, 0]
    green_pattern[0:height_green_half, width_green_half:width_green][green_pattern[0:height_green_half, width_green_half:width_green] == Color.GREEN] = color_guide[0, 1]
    green_pattern[height_green_half:height_green, width_green_half:width_green][green_pattern[height_green_half:height_green, width_green_half:width_green] == Color.GREEN] = color_guide[1, 1]

    output_grid = green_pattern
    return output_grid

def generate_input():
    # Initialize the 9x9 grid with black color.
    n, m = 9, 9
    grid = np.zeros((m, n), dtype=int)

    # Get available colors for the 2x2 sprite.
    available_colors = [c for c in Color.NOT_BLACK if c != Color.GREEN and c != Color.TEAL]

    # Generate the 6x6 green sprite with random pattern.
    green_pattern = random_sprite(m=6, n=6, color_palette=[Color.GREEN], density=0.3)

    # Get four different colors for the 2x2 sprite.
    four_colors = random.sample(available_colors, 4)
    color_guide = np.array([[four_colors[0], four_colors[1]], [four_colors[2], four_colors[3]]])

    r_green, c_green = 3, 3
    r_board, c_board = 0, 0

    # Place the 6x6 green sprite and the 2x2 sprite on the input grid.
    grid = paste(grid, green_pattern, r_green, c_green, transparent=Color.BLACK, clip=True)
    grid = paste(grid, color_guide, r_board, c_board, transparent=Color.BLACK, clip=True)

    # Place two teal divider lines (a full row and a full column) so the top-left region is exactly 2x2.
    grid[2, :] = Color.TEAL
    grid[:, 2] = Color.TEAL

    # Randomly rotate the input grid.
    grid = np.rot90(grid,-(np.random.randint(0, 4)))
    return grid
# ============= remove below this point for prompting =============
