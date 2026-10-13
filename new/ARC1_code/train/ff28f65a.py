import numpy as np
import random

from arc_utils import Color, detect_objects, first_free_location, paste, random_sprite

# concepts:
# counting, object detection, alternating pattern

# description:
# In the input you will see several 2 x 2 red squares on the grid.
# To make the output grid, you should count the number of red squares
# Then place the same number of 1 x 1 blue squares on the output grid following this pattern in the output:
# First fill the top row, then the next row, but skip every other column. Begin the first/third/fifth/etc row in the first column, but begin the second/forth/etc row in the second column.

def main(input_grid):
    """
    Count red 2x2 squares and mark that count in a fixed blue 3x3 pattern.

    Input structure:
    - The input is a square black grid containing separated 2x2 red squares.
    - Up to five red squares may be present.
    - The output is always a 3x3 black grid.

    Rule:
    - Detect all connected red square objects.
    - Count how many red squares are present.
    - Fill that many positions in the fixed blue order: corners by alternating columns with the center between them.

    Key insight:
    - The output ignores the locations of the red squares and encodes only their count
      along a predetermined sparse 3x3 sequence.
    """
    # Detect all the 2 x 2 red squares on the grid.
    red_square = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4, background=Color.BLACK)

    # Count the number of 2 x 2 red squares.
    num_red_square = len(red_square)

    # Output grid is always 3 x 3.
    output_grid = np.zeros((3, 3), dtype=int)

    # Fill the output grid with red square number follow specific pattern sequence:
    # 1. Fill the top row, then the next row, but skip every other column.
    # 2. Begin the first/third/fifth/etc row in the first column, but begin the second/forth/etc row in the second column.
    # Fill order inferred from training pairs:
    # (0,0), (0,2), (1,1), (2,0), (2,2)
    pos_list = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]

    # Place the same number of 1 x 1 blue squares on the output grid follow the specific pattern sequence.
    for i in range(num_red_square):
        r, c = pos_list[i]
        output_grid[r, c] = Color.BLUE
        
    return  output_grid

def generate_input():
    # Generate the background grid with size of n x n.
    square_len = np.random.randint(3, 10)
    n, m = square_len, square_len
    grid = np.zeros((m, n), dtype=int)

    # Generate the 2 x 2 red squares on the grid.
    square = random_sprite(m=2, n=2, color_palette=[Color.RED], density=1.0)

    # Randomly choose the number of 2 x 2 red squares.
    square_num = np.random.randint(1, 6)

    # Place the 2 x 2 red squares on the grid.
    for _ in range(square_num):
        _loc = first_free_location(grid, square, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=4)
        if _loc is None:
            break
        r, c = _loc
        paste(grid, square, r, c, transparent=Color.BLACK, clip=False)
    return grid

# ============= remove below this point for prompting =============
