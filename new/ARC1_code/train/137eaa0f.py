import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste


# concepts:
# objects, alignment by color

# description:
# In the input you will see some objects scattered around on a black grid. Each object has a single grey pixel, but everything else is a single other color.
# To make the output, place each object into the output grid such that the grey pixel is in the center of the output.
# Equivalently, move the objects to line up all their grey pixels so they overlap.
# The output grid should be the smallest possible size that contains all the objects (after they have been placed correctly), which for all the inputs here is 3x3.

def main(input_grid):
    """
    Overlay separated objects by aligning their gray anchor pixels and crop the result.

    Input structure:
    - The input contains several multicolor objects on black.
    - Each object has exactly one gray anchor pixel plus pixels of one other color.
    - Objects are initially separated from each other.

    Rule:
    - Crop each object to its sprite.
    - Place every sprite so its gray pixel lands on the same output center.
    - Crop the final overlay to the smallest bounding box containing all non-black pixels.

    Key insight:
    - Gray is a registration mark, not part of a color rule; aligning all gray anchors reveals the
      composite object.
    """
    # Plan:
    # 1. Extract the objects from the input, convert them into sprites by cropping them
    # 2. Make a big output grid
    # 3. Place each sprite into the output grid such that the grey pixel is in the center of the output
    # 4. Make the output as small as you can to contain all the objects

    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=8, monochromatic=False, return_format="objects")

    # Start with a generous canvas; we'll crop at the end.
    output_grid = np.full(input_grid.shape, bg, dtype=int)
    center_r, center_c = output_grid.shape[0] // 2, output_grid.shape[1] // 2

    for obj in objects:
        b = bbox(obj != bg)
        if b is None:
            continue
        sprite = crop_bbox(obj, b)
        (grey_r, grey_c) = map(int, np.argwhere(sprite == Color.GREY)[0])
        r0 = int(center_r - grey_r)
        c0 = int(center_c - grey_c)
        output_grid = paste(output_grid, sprite, r0, c0, transparent=bg, clip=True)

    b_out = bbox(output_grid != bg)
    assert b_out is not None
    return crop_bbox(output_grid, b_out)


def generate_input():
    # Create a series of 3x3 objects, each of which has a great pixel at the center, and none of which overlap except for the grey pixels
    sprites = []

    # To make sure there is no overlap, we keep track of what pixels are already occupied
    occupied = np.full((3, 3), False)

    n_objects = np.random.randint(2, 4)
    for _ in range(n_objects):
        sprite = np.full((3, 3), Color.BLACK)
        sprite[1, 1] = Color.GREY
        other_color = random.choice([c for c in Color.NOT_BLACK if c != Color.GREY])

        # Randomly pick a subset of valid pixels to color
        # To be valid it has to be unoccupied and not grey (it will be black)
        valid_pixels = np.argwhere(~occupied & (sprite == Color.BLACK))
        if len(valid_pixels) == 0:
            break

        n_pixels = np.random.randint(1, len(valid_pixels) + 1)
        pixels = valid_pixels[np.random.choice(len(valid_pixels), n_pixels, replace=False)]
        for r, cc in pixels:
            sprite[r, cc] = other_color
            occupied[r, cc] = True
        
        sprites.append(sprite)

    # Place the sprites randomly on a medium sized canvas but make sure they don't touch each other
    n, m = np.random.randint(10, 20), np.random.randint(10, 20)
    grid = np.full((m, n), Color.BLACK)
    for sprite in sprites:
        _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=2, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, cc = _loc
        paste(grid, sprite, r, cc, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
