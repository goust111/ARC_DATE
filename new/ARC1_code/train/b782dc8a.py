
import numpy as np
from typing import *
import random
from arc_utils import Color, colors_present, draw_line, find_components

# concepts:
# maze, path finding

# description:
# In the input you will see a maze with a path that has two indicator pixels of different colors.
# To make the output, fill all reachable parts of the maze starting with the indicator pixels and alternating colors.

def main(input_grid):
    """
    Fill reachable maze corridors from two colored seeds with alternating colors.

    Input structure:
    - Teal cells form the maze structure.
    - Two non-teal, non-black indicator colors seed black corridor cells.
    - Black cells connected to the seeds are the fill targets.

    Rule:
    - Identify the two indicator colors excluding teal and black.
    - From each seed, recursively fill adjacent black cells.
    - Alternate between the two indicator colors at every step.

    Key insight:
    - Corridor color is determined by path parity from a seed, while teal maze cells stay fixed.
    """
    # Output grid draws on top of the input grid
    output_grid = input_grid.copy()

    # Parse the input
    maze_color = Color.TEAL
    indicator_colors = [ color for color in colors_present(input_grid, exclude={ Color.BLACK }) if color != maze_color]
    assert len(indicator_colors) == 2, "expected exactly two indicator colors"
    
    # Fill the path with alternating colors in turn
    def fill_maze(cur_color, next_color, r, c, grid):
        width, height = grid.shape
        # Search outward in four directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for direction in directions:
            new_r, new_c = r + direction[0], c + direction[1]
            if 0 <= new_r < width and 0 <= new_c < height and grid[new_r, new_c] == Color.BLACK:
                grid[new_r, new_c] = next_color
                # Change the next color to the current color: swap current and next
                fill_maze(next_color, cur_color, new_r, new_c, grid)
    
    # Fill the path with two colors
    # Start to fill the path with the pixel that already has the path_color
    for r, c in np.argwhere((input_grid != Color.BLACK) & (input_grid != maze_color)):
        cur_color = input_grid[r, c]
        next_color = indicator_colors[0] if cur_color == indicator_colors[1] else indicator_colors[1]
        fill_maze(cur_color, next_color, r, c, output_grid)

    return output_grid

def generate_input():
    # Create the background grid
    width, height = np.random.randint(15, 30), np.random.randint(15, 30)
    grid = np.full((width, height), Color.BLACK)

    # function to check if the surrounding pixels are all black
    def check_available(r, c, grid, directions):
        for direction in directions:
            new_r, new_c = r + direction[0], c + direction[1]
            if 0 <= new_r and new_r < grid.shape[0] and 0 <= new_c and new_c < grid.shape[1] and grid[new_r, new_c] != Color.BLACK:
                return False
        return True
    
    # function for random walk to generate the maze
    def random_walk(color, grid, r, c):
        # Four walking directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Randomly choose one available direction
        random.shuffle(directions)
        for direction in directions:
            cur_r, cur_c = r + direction[0], c + direction[1]
            # Check if the next position is out of the grid
            if cur_r < 0 or cur_r >= width or cur_c < 0 or cur_c >= height:
                continue

            # Check if the next position touch other maze path
            rest_directions = [d for d in directions if d != (-direction[0], -direction[1])]
            if check_available(cur_r, cur_c, grid, rest_directions):
                # If not, mark the next position as the maze path
                # And continue the random walking
                grid[cur_r, cur_c] = color
                random_walk(color, grid, cur_r, cur_c)
                break

    # The color for the maze path        
    maze_color = Color.TEAL

    # Iterate over the grid to generate the maze
    for r, c in np.argwhere(grid == Color.BLACK):
        # If the current position is not empty, skip
        if grid[r, c] != Color.BLACK:
            continue

        # Check if the surrounding pixels are all black, which means we can start the random walking
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        if check_available(r, c, grid, directions):
            # Start the random walking
            grid[r, c] = maze_color

            # Avoid the exception `maximum recursion depth exceeded`
            try:
                random_walk(maze_color, grid, r, c)
            except:
                continue
    
    # The path has two colors which appear in turn
    path_color = np.random.choice([color for color in Color.NOT_BLACK if color != maze_color], 2, replace=False)

    # Mark the longest path to be colored to the path_color in turn
    objects = find_components(grid, bg=maze_color, connectivity=4, monochromatic=True, return_format="objects")
    path = max(objects, key=lambda r: np.sum(r == Color.BLACK))

    # Find one empty position to start the path
    r, c = random.choice(np.argwhere(path == Color.BLACK))

    # Color the position with one of the path_color
    grid[r, c] = path_color[0]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for direction in directions:
        if 0 <= r + direction[0] < width and 0 <= c + direction[1] < height and grid[r + direction[0], c + direction[1]] == Color.BLACK:
            # Color the surrounding pixels with the other path_color
            # Which indicates the color should be changed in the next step
            grid[r + direction[0], c + direction[1]] = path_color[1]

    return grid
