
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, collision, colors_present, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite

# concepts:
# puzzle pieces,

# description:
# In the input you will see two monochromatic objects -- "puzzle pieces"
# To make the output, fit the pieces together so that they form a single tightly-packed rectangular object. The pieces can be translated, but not rotated.

def main(input_grid):
    """
    Fit two monochromatic puzzle pieces together into the tightest rectangular assembly.

    Input structure:
    - The input is a black grid containing exactly two non-black monochromatic objects.
    - Each object is a puzzle piece that may be translated but not rotated.
    - The pieces come from a compact two-color object.

    Rule:
    - Crop the two pieces from the input.
    - Try all non-overlapping translations on a canvas large enough to hold both pieces.
    - Return the placement whose cropped bounding box leaves the fewest black cells.

    Key insight:
    - The reconstruction is the densest translation-only packing of the two pieces.
    """
    # Plan:
    # 1. Detect the pieces
    # 2. Consider different ways of placing them together
    # 3. Pick the way which makes them the most tightly packed, meaning that there is as little empty pixels remaining as possible

    # 1. Extract puzzle pieces
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")
    assert len(objects) == 2, "There should be exactly 2 objects"
    sprites = []
    for obj in objects:
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        sprites.append(crop_bbox(obj, _b))
    
    width = max(sprite.shape[0] for sprite in sprites) + min(sprite.shape[0] for sprite in sprites)
    height = max(sprite.shape[1] for sprite in sprites) + min(sprite.shape[1] for sprite in sprites)
    output_grid = np.full((width, height), Color.BLACK)

    # 2. Try to fit the pieces together
    possible_placements = [ (x1, x2, y1, y2)
                           for x1 in range(width - sprites[0].shape[0] + 1)
                           for x2 in range(width - sprites[1].shape[0] + 1)
                           for y1 in range(height - sprites[0].shape[1] + 1)
                           for y2 in range(height - sprites[1].shape[1] + 1) 
                           if not collision(object1=sprites[0], object2=sprites[1], r1=x1, r2=x2, c1=y1, c2=y2) ]
    
    def score_placement(x1, x2, y1, y2):
        # We are trying to make the puzzle pieces fit together perfectly
        # Therefore, there shouldn't be very many unfilled (black) pixels remaining after we place the pieces
        # So we are minimizing the number of black pixels
        # Equivalently maximizing the negative number of black pixels
        test_canvas = np.full_like(output_grid, Color.BLACK)
        paste(test_canvas, sprites[0], x1, y1, transparent=Color.BLACK, clip=True)
        paste(test_canvas, sprites[1], x2, y2, transparent=Color.BLACK, clip=True)
        _b = bbox(test_canvas != Color.BLACK)
        assert _b is not None
        test_canvas = crop_bbox(test_canvas, _b)
        return -np.sum(test_canvas == Color.BLACK)
    
    # pick the best one
    x1, x2, y1, y2 = max(possible_placements, key=lambda placement: score_placement(*placement))
    paste(output_grid, sprites[0], x1, y1, transparent=Color.BLACK, clip=True)
    paste(output_grid, sprites[1], x2, y2, transparent=Color.BLACK, clip=True)

    _b = bbox(output_grid != Color.BLACK)
    assert _b is not None
    return crop_bbox(output_grid, _b)

def generate_input():
    # Create a 2-color object and then break it up into 2 pieces
    two_color = random_sprite([2,3,4], [2,3,4], density=1, color_palette=Color.NOT_BLACK, background=Color.BLACK)
    if len(colors_present(two_color, exclude={ Color.BLACK })) != 2:
        return generate_input()
    
    # break it up into 2 pieces
    pieces = find_components(two_color, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")
    if len(pieces) != 2:
        return generate_input()
    
    # convert them to sprites and then randomly place them on a big canvas
    sprites = []
    for piece in pieces:
        _b = bbox(piece != Color.BLACK)
        assert _b is not None
        sprites.append(crop_bbox(piece, _b))
    width, height = np.random.randint(10, 20), np.random.randint(10, 20)
    grid = np.full((width, height), Color.BLACK)
    for sprite in sprites:
        _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return grid
