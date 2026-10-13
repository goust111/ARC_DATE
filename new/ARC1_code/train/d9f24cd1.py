import numpy as np
import random
from arc_utils import Color, draw_line, randomly_scatter_points


# concepts:
# line drawing, obstacle avoidance

# description:
# In the input you will see several red pixels on the bottom row of the grid, and some gray pixels scattered on the grid.
# To make the output grid, you should draw a red line upward from each red pixel, but avoiding the gray pixels.
# To avoid touching the gray pixels, go right to avoid them until you can go up again.

def main(input_grid):
    """
    Grow red paths upward from red starts while sidestepping gray obstacles to the right.

    Input structure:
    - The grid has red start pixels along the bottom row.
    - Gray pixels are scattered as obstacles away from the outer border.
    - All other cells are black background.

    Rule:
    - For each red start, attempt to move one cell upward at each step.
    - If the cell above is gray, draw one red step to the right instead.
    - Stop a path when it reaches the top or can no longer sidestep within the grid.

    Key insight:
    - Each path follows a simple local rule: upward progress is preferred, and gray
      blockers only cause rightward detours.
    """
    # The output grid is the same size as the input grid, and we are going to draw on top of the input, so we copy it
    output_grid = input_grid.copy()
    H, W = input_grid.shape

    # Red pixels are on the bottom row.
    starts = np.argwhere(input_grid[H - 1, :] == Color.RED).flatten()
    for c0 in starts:
        r = H - 1
        c = int(c0)
        # Draw upward; if blocked by GRAY above, sidestep right.
        while r > 0 and 0 <= c < W:
            if int(output_grid[r - 1, c]) == Color.GRAY:
                if c + 1 >= W:
                    break
                output_grid[r, c + 1] = Color.RED
                c += 1
            else:
                output_grid[r - 1, c] = Color.RED
                r -= 1

    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    n, m = 10, 10
    grid = np.zeros((m, n), dtype=int)

    # Generate the red pixels on the bottom row.
    # Get 3 random positions for the red pixels.
    available_postion = range(1, 9)
    red_location = random.sample(available_postion, 3)

    # Draw the red pixels on the bottom row.
    for pos_r in red_location:
        grid[pos_r, -1] = Color.RED
    
    # Get the region except the bottom row, left most column and right most column.
    # Randomly scatter the gray pixels on the grid.
    randomly_scatter_points(grid[1:-1, 1:-1], color=Color.GRAY, density=0.1)
    
    return grid
# ============= remove below this point for prompting =============
