
import numpy as np
from typing import *
from arc_utils import Color, colors_present, draw_line, paste, randomly_scatter_points, randomly_spaced_indices
import random

# concepts:
# downscaling, rectangular partitions

# description:
# In the input you will see a grid consisting of a chessboard pattern (rectangular partitions) of different colors.
# Each rectangular partition region is incompletely scattered with a color. Regions are separated by black lines, going all the way top-bottom/left-right. 
# To make the output, make a grid with one color pixel for each colored rectangular region of the input.

def main(input_grid):
    """
    Downscale a black-line grid of colored partitions to one color pixel per region.

    Input structure:
    - The input is divided into rectangular regions by full black rows and columns.
    - Each region contains scattered pixels of one non-black color.
    - Region layout determines the output grid layout.

    Rule:
    - Locate the horizontal and vertical black divider lines.
    - Iterate the rectangular intervals between dividers.
    - Put each region's non-black color into the corresponding output cell.

    Key insight:
    - Divider geometry maps large colored regions directly to output pixel coordinates.
    """
    # Plan:
    # 1. Partition the input into rectangular regions by finding all horizontal and vertical black lines
    # 2. For each region, find the color of the region
    # 3. Use one pixel to represent the original region and create the output grid

    # 1. Input parsing
    # Get the shape of the input grid
    width, height = input_grid.shape
    background = Color.BLACK
    # Find all horizontal and vertical lines
    vertical_lines = [ r for r in range(width) if np.all(input_grid[r, :] == background) ]
    horizontal_lines = [ c for c in range(height) if np.all(input_grid[:, c] == background) ]
    
    # Start from (0, 0)
    vertical_lines = [0] + vertical_lines
    horizontal_lines = [0] + horizontal_lines

    # Deduplicate successive lines
    vertical_lines = [r for i, r in enumerate(vertical_lines) if i == 0 or r != vertical_lines[i - 1]]
    horizontal_lines = [c for i, c in enumerate(horizontal_lines) if i == 0 or c != horizontal_lines[i - 1]]

    # use one pixel to represent the original region and create the output grid
    output_width, output_height = len(vertical_lines), len(horizontal_lines)
    output_grid = np.full((output_width, output_height), background) 

    # Initialize the output grid
    for i in range(len(vertical_lines)):
        for j in range(len(horizontal_lines)):
            # Get the region of the color
            x1 = vertical_lines[i]
            x2 = vertical_lines[i + 1] if i + 1 < len(vertical_lines) else width
            y1 = horizontal_lines[j]
            y2 = horizontal_lines[j + 1] if j + 1 < len(horizontal_lines) else height

            # Get the original region
            region = input_grid[x1:x2, y1:y2]
            # Get the color of the region
            color = colors_present(region, exclude={ Color.BLACK })[0]
            # Use one pixel to represent the original region
            output_grid[i, j] = color

    return output_grid

def generate_input():
    # Randomly generate the grid size
    width, height = np.random.randint(20, 30), np.random.randint(20, 30)
    grid = np.full((width, height), Color.BLACK)

    # Randomly select the grid separation
    n_region_horizontal, n_region_vertical = np.random.randint(2, 5), np.random.randint(2, 5)
    horizontal_region_boundaries = randomly_spaced_indices(max_len=width, n_indices=n_region_horizontal - 1, border_size=1, padding=2)
    vertical_region_boundaries = randomly_spaced_indices(max_len=height, n_indices=n_region_vertical - 1, border_size=1, padding=2)

    # Randomly select the colors to put in each region/partition
    colors = np.random.choice(Color.NOT_BLACK, (n_region_horizontal, n_region_vertical), replace=True)

    # Assign the colors to each partition
    # Big (X,Y) indexes the partition coordinates, not the canvas coordinates, which are little (x,y)
    for X in range(n_region_horizontal):
        for Y in range(n_region_vertical):
            # Get the region on the canvas
            # Note that the final region goes all the way to the width/height
            x1 = 0 if X == 0 else horizontal_region_boundaries[X-1]
            x2 = horizontal_region_boundaries[X] if X < len(horizontal_region_boundaries) else width
            y1 = 0 if Y == 0 else vertical_region_boundaries[Y-1]
            y2 = vertical_region_boundaries[Y] if Y < len(vertical_region_boundaries) else height

            # Each region is incomplete scattered color
            sprite = np.full((x2 - x1, y2 - y1), Color.BLACK)
            randomly_scatter_points(sprite, color=colors[X, Y], density=0.8)

            # Place the pattern in the grid
            paste(grid, sprite, x1, y1, transparent=Color.BLACK, clip=True)

    # Draw black lines to separate the colors
    # First draw vertical lines
    for r in horizontal_region_boundaries:
        draw_line(grid=grid, r=r, c=0, direction=(0, 1), color=Color.BLACK)
        # equivalently: grid[x, :] = Color.BLACK

    # Then draw horizontal lines
    for c in vertical_region_boundaries:
        draw_line(grid=grid, r=0, c=c, direction=(1, 0), color=Color.BLACK)
        # equivalently: grid[:, y] = Color.BLACK

    return grid
