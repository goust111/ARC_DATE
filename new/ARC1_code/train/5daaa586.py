import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, randomly_scatter_points

# concepts:
# pattern extraction, pixel expanding

# description:
# In the input you will see four lines of different colors intersecting and forming a rectangle.
# Few pixels of one specific line's color are scattered in the grid.
# To make the output, you should cropped out the rectangle and extend the scatterd pixels to 
# the specific line which has same color as the scattered pixels.

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Crop the colored rectangle frame and connect scattered pixels to the same-colored frame side.

    Input structure:
    - Four full non-black lines form a rectangle on a black background.
    - Each frame side has a distinct color.
    - Extra scattered pixels share the color of exactly one frame side.

    Rule:
    - Locate the two full horizontal and two full vertical frame lines.
    - Identify the scattered pixel color outside those lines.
    - Crop to the frame and draw straight connections from each scattered pixel to the matching-colored side.

    Key insight:
    - The scattered pixel color selects which frame side to connect to after the rectangle
      is cropped out of the larger grid.
    """
    output_grid = np.copy(input_grid)
    
    # Identify two vertical and two horizontal lines
    horizontal_lines = np.where(np.all(output_grid != Color.BLACK, axis=1))[0]
    vertical_lines = np.where(np.all(output_grid != Color.BLACK, axis=0))[0]
    
    # Find out the color of the scattered pixels
    mask = np.ones(output_grid.shape, dtype=bool)
    mask[horizontal_lines, :] = False
    mask[:, vertical_lines] = False
    color = output_grid[mask & (output_grid != Color.BLACK)][0]
    
    # Crop the grid by the rectangle formed by the four lines
    output_grid = output_grid[horizontal_lines[0] : horizontal_lines[1] + 1, vertical_lines[0] : vertical_lines[1] + 1]
    
    # Extend the scattered pixels to the line that has same color as the scattered pixels
    for r in range(len(output_grid)):
        for c in range(len(output_grid[0])):
            # did we find a scattered pixel? (of color `color`)
            if output_grid[r, c] == color:
                # draw a line to the matching color line, which is going to be either left/right/top/bottom
                # so we need to examine four cases for each location that the matching color line might be

                # Left: x=0 indicates this is the left line
                if output_grid[0, c] == color:
                    draw_line(output_grid, r, c, end_r= 0, end_c= c, color = color)
                # Right: x=len(output_grid) - 1 indicates this is the right line
                if output_grid[len(output_grid) - 1, c] == color:
                    draw_line(output_grid, r, c, end_r= len(output_grid) - 1, end_c= c, color = color)
                # Top: y=0 indicates this is the top line
                if output_grid[r, 0] == color:
                    draw_line(output_grid, r, c, end_r= r, end_c= 0, color = color)
                # Bottom: y=len(output_grid[0]) - 1 indicates this is the bottom line
                if output_grid[r, len(output_grid[0]) - 1] == color:
                    draw_line(output_grid, r, c, end_r= r, end_c= len(output_grid[0]) - 1, color = color)

    return output_grid

def generate_input() -> np.ndarray:
    # Generate the background grid with size of n x m.
    n, m = np.random.randint(15, 25), np.random.randint(15, 25)
    grid = np.zeros((m, n), dtype=int)

    # Randomly get the position of two horizontal and two vertical lines
    # The lines form a rectangle
    horizontal_lines_pos = [np.random.randint(1, m // 2), np.random.randint(m // 2 + 1, m - 1)]
    vertical_lines_pos = [np.random.randint(1, n // 2), np.random.randint(n // 2 + 1, n - 1)]
    
    # Generate four different colors for four lines
    line_colors = np.random.choice(Color.NOT_BLACK, 4, replace=False)

    # Randomly choose a color from line and create several scattered pixels of that color
    scattered_pixel_color = random.choice(line_colors)
    randomly_scatter_points(grid, color=scattered_pixel_color, density=0.2)
    
    # Randomly determine order for placing colored lines
    lines_pos = [(horizontal_lines_pos[0], 0), (horizontal_lines_pos[1], 0), (0, vertical_lines_pos[0]), (0, vertical_lines_pos[1])]
    random.shuffle(lines_pos)

    # Draw four lines on the grid
    for i in range(len(lines_pos)):
        line_r, line_c = lines_pos[i]

        # Draw horizontal line
        if line_c == 0:
            draw_line(grid = grid, r= line_r, c= 0, end_r= line_r, end_c= n, color = line_colors[i])
        
        # Draw vertical line
        if line_r == 0:
            draw_line(grid = grid, r= 0, c= line_c, end_r= m, end_c= line_c, color = line_colors[i])
    
    return grid
