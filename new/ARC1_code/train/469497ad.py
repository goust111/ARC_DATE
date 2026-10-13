import numpy as np

import random
from arc_utils import Color, bbox, draw_line, find_components, first_free_location, paste, random_sprite

# concepts:
# counting, resizing

# description:
# In the input, you will see a grid with a row of colored blocks on the bottom and the right. 
# There is also a square in the top left that is not touching the other colors.
# To make the output:
# 1. count the number of colors that aren't black
# 2. enlarge every pixel in the input by a factor of the number of colors
# 3. add diagonal red lines coming out of the corners of the square in the top left portion of the grid

def main(input_grid):
    """
    Scale the whole grid by its number of non-black colors and draw red diagonals from the enlarged square's corners.

    Input structure:
    - The grid has non-black color runs on the bottom row and right column.
    - A separate non-black square sits away from those border runs.
    - The number of distinct non-black colors determines the scale factor.

    Rule:
    - Count distinct non-black colors and magnify every input pixel by that factor.
    - Find the enlarged square component in the scaled output.
    - Draw red diagonal rays outward from the four corners until they meet non-black cells or the edge.

    Key insight:
    - Border colors encode the numeric scale, while the isolated square anchors the added
      diagonal construction after scaling.
    """
    # count the number of colors that aren't black
    num_colors = len(set(map(int, np.unique(input_grid))) - {Color.BLACK})
    if num_colors <= 0:
        return np.copy(input_grid)

    # magnify the pixels in input grid onto the output grid
    output_grid = np.repeat(np.repeat(input_grid, num_colors, axis=0), num_colors, axis=1)

    # find the square in the output grid
    objects = find_components(output_grid, bg=Color.BLACK, connectivity=8, monochromatic=False, return_format="objects")
    square = next(obj for obj in objects if int(obj[-1, -1]) == Color.BLACK)
    
    # find the bounding box of the square
    b = bbox(square != Color.BLACK)
    assert b is not None
    r0, c0, r1, c1 = b

    # draw the diagonal red lines
    stop_at = set(Color.NOT_BLACK)
    L = max(output_grid.shape) * 3
    draw_line(output_grid, r0 - 1, c0 - 1, length=L, color=Color.RED, direction=(-1, -1), stop_at=stop_at, clip=True)
    draw_line(output_grid, r1 + 1, c1 + 1, length=L, color=Color.RED, direction=(1, 1), stop_at=stop_at, clip=True)
    draw_line(output_grid, r0 - 1, c1 + 1, length=L, color=Color.RED, direction=(-1, 1), stop_at=stop_at, clip=True)
    draw_line(output_grid, r1 + 1, c0 - 1, length=L, color=Color.RED, direction=(1, -1), stop_at=stop_at, clip=True)

    return output_grid

def generate_input():
    # make a 5x5 black grid for the background
    n = m = 5
    grid = np.zeros((n,m), dtype=int)

    # pick the colors for the bottom and right of the grid
    colors = list(Color.NOT_BLACK)

    # construct a random sequence of colors for the bottom and right of the grid
    # don't repeat a color in the sequence
    sequence = []
    while len(sequence) < 5:
        # select a color
        color = np.random.choice(colors)
        # remove the color from the array of remaining colors
        colors = np.delete(colors, np.where(colors == color))
        length = np.random.randint(1, 6 - len(sequence))
        sequence.extend([color] * length)

    # put the same sequence on the bottom and right of the grid
    grid[-1, :] = sequence
    grid[:, -1] = sequence

    # pick the color of the square
    square_color = np.random.choice(list(Color.NOT_BLACK))
    
    # make the square
    square_sprite = random_sprite(2, 2, density=1, color_palette=[square_color])

    # put the square on the grid so it doesn't touch the bottom or right
    _loc = first_free_location(grid, square_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, square_sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return grid


# ============= remove below this point for prompting =============
