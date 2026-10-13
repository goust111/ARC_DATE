import numpy as np
from typing import *
import random
from arc_utils import Color, colors_present, draw_line, find_components, paste, randomly_scatter_points

# concepts:
# counting, dividers, filling

# description:
# In the input you will see grey horizontal and vertical bars that divide rectangular regions. Each rectangular region is black with some colored pixels added.
# To make the output, fill the rectangular region with the most colored pixels. Fill it with is color. Fill the other rectangular regions with black.   

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Keep only the divider region or regions with the most colored pixels and fill them with their color.

    Input structure:
    - Full divider rows and columns split the grid into rectangular regions.
    - Each region is mostly black with scattered pixels of one non-black, non-divider color.
    - The divider color appears as a full constant row or column.

    Rule:
    - Detect the divider color and extract all separated regions.
    - Count non-black, non-divider colored pixels in each region.
    - Fill regions tied for the maximum count with their single color and fill all other regions black.

    Key insight:
    - The scattered pixels are votes for a region; only the densest region contents survive.
    """
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)  

    # Get all the rectangular regions seperated by horizontal and vertical dividers
    # The dividers are colored grey, but more generally their color is just whatever color stretches all the way horizontally or vertically
    for r in range(input_grid.shape[0]):
        if np.all(input_grid[r, :] == input_grid[r, 0]):
            divider_color = input_grid[r, 0]
            break
    # For this problem you could also do: divider_color = Color.GRAY
    regions = find_components(output_grid, bg=divider_color, connectivity=4, monochromatic=False, return_format="objects")

    # Find the region with the most colored pixels inside of it
    num_colored_pixels = [ np.sum((region != divider_color) & (region != Color.BLACK)) for region in regions ]
    max_colored_pixels = max(num_colored_pixels)

    # Fill the region with the most colored pixels with its color
    # Fill the other regions with black
    for region_obj in regions:
        # Figure out if it is one of the max colored regions to determine what the target color is that we are going to fill with
        num_colored_pixels_in_this_region = np.sum((region_obj != divider_color) & (region_obj != Color.BLACK))
        if num_colored_pixels_in_this_region == max_colored_pixels:
            colors = [ color for color in colors_present(region_obj, exclude={ divider_color }) if color != Color.BLACK ]
            assert len(colors) == 1, "Each region should have only one color"
            target_color = colors[0]
        else:
            target_color = Color.BLACK

        # Fill the region with the target color
        output_grid[region_obj != divider_color] = target_color

    return output_grid

def generate_input() -> np.ndarray:
    # 3x3 grid of rectangular regions separated by full grey dividers.
    region_len = int(np.random.choice([4, 5, 7]))
    region_num = 3
    divider_color = Color.GREY
    interior_color = int(random.choice([c for c in Color.NOT_BLACK if c != divider_color]))

    H = region_len * region_num + (region_num - 1)
    W = region_len * region_num + (region_num - 1)
    grid = np.full((H, W), Color.BLACK, dtype=int)

    # Full divider rows/cols.
    for i in range(region_len, H, region_len + 1):
        grid[i, :] = divider_color
    for j in range(region_len, W, region_len + 1):
        grid[:, j] = divider_color

    # Add different densities of a SINGLE interior color per region.
    for r0 in range(0, H, region_len + 1):
        for c0 in range(0, W, region_len + 1):
            square = np.full((region_len, region_len), Color.BLACK, dtype=int)
            density = np.random.uniform(0.1, 0.9)
            square = randomly_scatter_points(square, color=interior_color, density=density)
            paste(grid, square, r0, c0, transparent=Color.BLACK, clip=True)

    # Ensure the first scanned row is not constant-black, so main() finds the divider color.
    grid[0, 0] = interior_color
    return grid
