import numpy as np

import random
from arc_utils import Color, bbox, bbox_anchor, bbox_h, bbox_w, colors_present, draw_line, find_components, first_free_location, paste, random_sprite

# concepts:
# counting

# description:
# In the input, you will see a 10x10 black grid containing a few 2x2 squares. Each square contains 1-4 colors.
# To create the output, draw a green 2xL rectangle just below each 2x2 square, where L is the number of colors in the square.

def main(input_grid):
    """
    Draw a green marker rectangle below each 2x2 square sized by its color count.

    Input structure:
    - The input is a 10x10 black grid with separated 2x2 colored squares.
    - Each square may contain one to four non-black colors.
    - The generated placement leaves room around squares for added markers.

    Rule:
    - Detect each multicolor square object.
    - Count the distinct non-black colors inside it.
    - Paste a green rectangle directly below the square with height equal to that count and matching square width.

    Key insight:
    - The added green shape encodes diversity inside each square: more distinct colors
      produce a taller marker.
    """
    # copy the input grid to the output grid
    output_grid = input_grid.copy()
    # find all connected components in the input grid
    bg = Color.BLACK
    objects = find_components(
        input_grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="objects",
    )

    # for each object, draw a green rectangle below it of shape 2xL, where L is the number of colors in the square
    for obj in objects:
        b = bbox(obj != bg)
        if b is None:
            continue
        r0, c0 = bbox_anchor(b, anchor="upper left")
        h = bbox_h(b)
        w = bbox_w(b)
        num_colors = len(colors_present(obj, exclude={bg}))
        # Place a green rectangle just below the square.
        # In canonical (r,c), the rectangle extends downward with height = (#colors in the square)
        # and width = (square width).
        green_rectangle = np.full((num_colors, w), Color.GREEN)
        output_grid = paste(output_grid, green_rectangle, r0 + h, c0, clip=True)

    return output_grid


def generate_input():
    input_grid = np.full((10, 10), Color.BLACK)
    num_squares = np.random.randint(1, 5)
    for _ in range(num_squares):
        square = random_sprite(2, 2, density=1.0, color_palette=Color.NOT_BLACK)
        _loc = first_free_location(input_grid, square, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        input_grid = paste(input_grid, square, r, c, transparent=Color.BLACK, clip=True)

    return input_grid
# ============= remove below this point for prompting =============
