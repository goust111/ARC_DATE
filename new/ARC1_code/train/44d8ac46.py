import numpy as np
import random
from arc_utils import Color, bbox, bbox_anchor, crop_bbox, draw_line, find_components, first_free_location, object_interior, paste, random_sprite


# concepts:
# filling, topology

# description:
# The input is a black 12x12 grid containing a few grey squares. Each square has a "hole" in it, a contiguous black region of pixels.
# To create the output, fill in the hole of each grey object with red if the hole is a square. Otherwise, leave the hole as is.

def main(input_grid):
    """
    Fill only square-shaped holes inside gray square objects with red.

    Input structure:
    - The grid is black with multiple gray square objects.
    - Each gray object may contain a contiguous black hole inside it.
    - Some holes are solid square blocks and others are irregular.

    Rule:
    - Detect each gray object and crop it to its bounding box.
    - Compute the enclosed black interior mask.
    - Recolor the hole red only when its bounding box is square and completely filled.

    Key insight:
    - The decision is based on the hole's own geometry, not on the outer gray square.
    """
    # get the grey squares
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    output_grid = input_grid.copy()

    for obj in objects:
        b = bbox(obj != Color.BLACK)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        sprite = crop_bbox(obj, b)

        # Identify the black hole(s) inside the grey object.
        hole_mask = (sprite == Color.BLACK) & object_interior(sprite, background=Color.BLACK)

        hb = bbox(hole_mask)
        if hb is not None:
            hr0, hc0, hr1, hc1 = hb
            hh = hr1 - hr0 + 1
            hw = hc1 - hc0 + 1
            hole_crop = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
            if hh == hw and int(hole_crop.sum()) == hh * hw:
                sprite[hole_mask] = Color.RED

        output_grid = paste(output_grid, sprite, r0, c0, transparent=Color.BLACK)

    return output_grid


def generate_input():
    # create a 12x12 black grid
    grid = np.full((12, 12), Color.BLACK)

    # add 2-3 grey squares.
    # For each grey square, add a hole in the middle.
    # 50% chance the hole is a square, otherwise it's a random contiguous object.
    # The hole should not overlap with the border of the square.

    num_grey_squares = np.random.randint(2, 4)

    for _ in range(num_grey_squares):
        # create grey square, and try to find a location for it
        length = np.random.randint(4, 6)
        grey_square = np.full((length, length), Color.GREY)
        try:
            _loc = first_free_location(grid, grey_square, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except ValueError:
            # we were unable to find a space for a square; try over from scratch
            return generate_input()

        # # add a hole in the middle.
        # # 50% chance the hole is a square, otherwise it's a random contiguous object.
        # # The hole should not overlap with the border of the square.
        has_square_hole = np.random.choice([True, False])
        if has_square_hole:
            hole_size = np.random.randint(1, length - 1)
            hole_r, hole_c = np.random.randint(1, length - hole_size), np.random.randint(1, length - hole_size)
            grey_square[hole_r:hole_r + hole_size, hole_c:hole_c + hole_size] = Color.BLACK
        else:
            hole_obj = random_sprite(length-2, length-2, color_palette=[Color.BLACK], background=Color.GREY)
            grey_square[1:-1, 1:-1] = hole_obj

        grid = paste(grid, grey_square, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============

