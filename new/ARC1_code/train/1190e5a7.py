import numpy as np
import random
from arc_utils import Color, draw_line, first_free_location, paste


# concepts:
# horizontal/vertical bars, counting

# description:
# In the input, you will see horizontal and vertical bars, dividing the input into a grid of rectangular regions, on a non-black background.
# To make the output produce a monochromatic image whose width is the number of background-colored regions going left-to-right, and whose height is the number of regions going top-to-bottom.
# The output should have the same background.


def main(input_grid):
    """
    Count divider-defined regions and output a solid grid with those region dimensions.

    Input structure:
    - The input has a non-black background crossed by full-row and full-column bars of another color.
    - Bars divide the canvas into rectangular regions.
    - The background color is the dominant non-bar color.

    Rule:
    - Identify the bar color from a full row or full column.
    - Count bar columns and rows to infer the number of regions left-to-right and top-to-bottom.
    - Return a solid grid of background color with shape `(region_rows, region_columns)`.

    Key insight:
    - The answer abstracts away bar positions and sizes; only the number of partitions in each
      direction matters.
    """
    # Plan:
    # 1. Find the color of the background and the bars
    # 2. Count the number of regions going left-to-right
    # 3. Count the number of regions going top-to-bottom

    H, W = input_grid.shape

    # Bar color = color of any full row/col stripe.
    bar_color = None
    for r in range(H):
        if np.all(input_grid[r, :] == input_grid[r, 0]):
            bar_color = int(input_grid[r, 0])
            break
    if bar_color is None:
        for c in range(W):
            if np.all(input_grid[:, c] == input_grid[0, c]):
                bar_color = int(input_grid[0, c])
                break
    assert bar_color is not None

    # Background = most common non-bar color.
    vals = input_grid.flatten().astype(int)
    non_bar = vals[vals != bar_color]
    background = int(np.bincount(non_bar).argmax())

    # Count vertical bar columns and horizontal bar rows.
    n_bar_cols = sum(1 for c in range(W) if input_grid[0, c] == bar_color)
    n_bar_rows = sum(1 for r in range(H) if input_grid[r, 0] == bar_color)

    n_regions_lr = n_bar_cols + 1
    n_regions_tb = n_bar_rows + 1

    # Output is (n_regions_tb, n_regions_lr), filled with background color.
    return np.full((n_regions_tb, n_regions_lr), background, dtype=int)


def generate_input():
    # Picking background and line colors
    background_color, bar_color = random.sample(list(Color.NOT_BLACK), k=2)

    # Creating the background grid
    n = random.randint(11, 21)
    grid = np.full((n, n), background_color)

    # Create sprites for the bars. We will place them randomly later.
    vertical_line_sprite = np.full((1, n), bar_color)
    horizontal_line_sprite = np.full((n, 1), bar_color)

    # Picking how many lines to have in each dimension
    line_n, line_m = random.randint(1, n // 3), random.randint(1, n // 3)

    # Create separate grids for horizontal and vertical lines, and then merge them later
    # This is because horizontal/vertical are allowed to intersect, but horizontals can't touch each other (and verticals can't touch each other)
    # So we need to do collision detection / free location separately for each type of line
    vertical_line_grid = grid.copy()
    horizontal_line_grid = grid.copy()

    # Verticals
    for i in range(line_n):
        _loc = first_free_location(vertical_line_grid, vertical_line_sprite, transparent=background_color, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(vertical_line_grid, vertical_line_sprite, r, c, transparent=Color.BLACK, clip=True)

    # Horizontals
    for i in range(line_m):
        _loc = first_free_location(horizontal_line_grid, horizontal_line_sprite, transparent=background_color, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(horizontal_line_grid, horizontal_line_sprite, r, c, transparent=Color.BLACK, clip=True)

    # Combine the two line grids
    paste(grid, vertical_line_grid, 0, 0, transparent=background_color, clip=True)
    paste(grid, horizontal_line_grid, 0, 0, transparent=background_color, clip=True)

    return grid
# ============= remove below this point for prompting =============
