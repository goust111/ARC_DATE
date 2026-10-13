import numpy as np
import random
from arc_utils import Color, draw_line, randomly_scatter_points


# concepts:
# draw lines, detect objects

# description:
# In the input you will see three colors scattered on the grid. One color only have four pixels on the boundary of the grid.
# To make the output grid, you should connect the four pixels of the color on the boundary of the grid to make two lines.

def main(input_grid):
    """
    Draw full grid-spanning lines through the special color's boundary marker pairs.

    Input structure:
    - The input contains scattered pixels of several colors.
    - One color appears only as four pixels on the grid boundary.
    - Those boundary pixels mark the rows or columns of the output lines.

    Rule:
    - Find the color whose pixels all lie on the boundary.
    - For markers on the top or bottom edge, draw a full vertical line through that column.
    - For markers on the left or right edge, draw a full horizontal line through that row.

    Key insight:
    - Boundary-only pixels are line endpoints; all other scattered colors are noise.
    """
    # Find the used color
    all_color = np.unique(input_grid)

    # Find the color all on the boundary, which is the lines' color
    def on_boundary(rr, cc):
        return (
            rr == 0
            or rr == input_grid.shape[0] - 1
            or cc == 0
            or cc == input_grid.shape[1] - 1
        )
    
    # Get the color all on the boundary, which is the lines' color
    line_color = None
    for color in all_color:
        coords = np.argwhere(input_grid == color)
        if coords.size == 0:
            continue
        if all(on_boundary(int(rr), int(cc)) for rr, cc in coords):
            line_color = int(color)
            break
    assert line_color is not None
    output_grid = np.zeros_like(input_grid)

    # Find the boundary pixels of the line_color and then draw a horizontal/vertical line to its matching pair
    coords = [tuple(map(int, xy)) for xy in np.argwhere(input_grid == line_color)]
    H, W = input_grid.shape
    for r, c in coords:
        if r == 0 or r == H - 1:
            # point on top/bottom => draw vertical line down/up through it
            output_grid[:, c] = line_color
        elif c == 0 or c == W - 1:
            # point on left/right => draw horizontal line across through it
            output_grid[r, :] = line_color
    
    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    n, m = np.random.randint(12, 20), np.random.randint(12, 20)
    grid = np.zeros((m, n), dtype=int)

    # Generate random three colors on the grid.
    three_colors = random.sample(Color.NOT_BLACK, k = 3)

    scattered_colors = three_colors[:2]
    special_color = three_colors[2]

    # Two colors of pixels are scattered on the grid as noise.
    for scattered_color in scattered_colors:
        randomly_scatter_points(grid, color=scattered_color, density=0.2)

    # One color of pixels are scattered on the boundary of the grid, and only four pixels
    # They are guaranteed to form two lines.
    for _ in range(2):
        # Choose if the line is horizontal or vertical
        if_horizonal = np.random.choice([True, False])
        if if_horizonal:
            # Choose the position of the line
            pos = np.random.choice(range(1, n - 1))

            # Draw horizontal line
            grid[0, pos] = special_color
            grid[m - 1, pos] = special_color
        else:
            # Choose the position of the line
            pos = np.random.choice(range(1, m - 1))

            # Draw vertical line
            grid[pos, 0] = special_color
            grid[pos, n - 1] = special_color

    return grid
# ============= remove below this point for prompting =============
