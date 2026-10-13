import numpy as np
import random
from arc_utils import Color, draw_line, paste, random_free_location, random_sprite

# concepts:
# alignment, objects

# description:
# In the input you will see a red, blue, and yellow shape. Each are the same shape (but different color). They occur left to right in the input grid on a black background, but at different vertical heights.
# The output is the same as the input, but with the vertical heights of the red and yellow shapes adjusted to match the height of the blue shape.

def main(input_grid):
    """
    Align the red and yellow copies vertically to the blue copy while preserving their horizontal positions.

    Input structure:
    - A black background contains blue, red, and yellow versions of the same shape.
    - The three colored objects appear left to right in separate horizontal spans.
    - The copies may start at different row offsets.

    Rule:
    - Locate the blue, red, and yellow pixel coordinates.
    - Keep the blue object fixed.
    - Replace the red and yellow row coordinates with the blue row coordinates, keeping their columns.

    Key insight:
    - Because all three objects have the same mask, the blue object's row coordinates provide
      the vertical alignment template for the other colors.
    """
    # find the blue shape, red shape, and yellow shape
    blue_coords = np.where(input_grid == Color.BLUE)
    red_coords = np.where(input_grid == Color.RED)
    yellow_coords = np.where(input_grid == Color.YELLOW)

    # Set the vertical height (row coordinate) of the red and yellow shapes to match the blue shape.
    red_coords = (blue_coords[0], red_coords[1])
    yellow_coords = (blue_coords[0], yellow_coords[1])

    # make output grid with the colored shapes at their new locations
    output_grid = np.full_like(input_grid, Color.BLACK)
    output_grid[blue_coords] = Color.BLUE
    output_grid[red_coords] = Color.RED
    output_grid[yellow_coords] = Color.YELLOW

    return output_grid


def generate_input():

    # All three shapes are the same shape, but different colors, so we generate one sprite and color it three ways
    # We put each sprite in a different grid, and concatenate the grids to make the input grid

    # make a random sprite of size (rows, cols) in [1..4]
    sprite_h = np.random.randint(1, 5)
    sprite_w = np.random.randint(1, 5)
    sprite = random_sprite(sprite_h, sprite_w)

    # Figure out the height of the output grid
    # This has to be the same across all three colors, because we concatenate them along the x axis
    grid_h = np.random.randint(sprite_h + 1, 16)

    # for each color,
    # put a colored form of the shape in a random spot in a new grid
    subgrids = []
    for color in [Color.BLUE, Color.RED, Color.YELLOW]:
        # make a grid to put the shape in
        # the grid should be wide enough to fit the shape
        grid_w = np.random.randint(sprite_w, 30 // 3)
        subgrid = np.full((grid_h, grid_w), Color.BLACK, dtype=int)

        # make the shape that color
        colored_sprite = np.copy(sprite)
        colored_sprite[sprite != Color.BLACK] = color

        # put the shape in a random spot in its grid
        r, c = random_free_location(subgrid, colored_sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        paste(subgrid, colored_sprite, r, c, transparent=Color.BLACK, clip=True)
        subgrids.append(subgrid)

    # now concatenate the subgrids along the x axis to make the input grid
    grid = np.concatenate(subgrids, axis=1)
    return grid
# ============= remove below this point for prompting =============
