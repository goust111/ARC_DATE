
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, colors_present, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite, scale_sprite

# concepts:
# puzzle pieces, rescaling

# description:
# In the input you will see a small multicolored object, and a big region bordered by 4 yellow pixels in the corners. The big region contains some colored things.
# To make the output, rescale and translate the small multicolored object so that it is in the big region and matches as much as possible the colors of the things in the big region.
# The output should be just the big region with its 4 yellow pixels (plus the rescaled and translated thing).

def main(input_grid):
    """
    Scale and place the external multicolor object into the yellow-cornered region where it matches existing colors.

    Input structure:
    - The input contains one small multicolor object outside a larger region.
    - Four yellow singleton pixels mark the corners of the larger output region.
    - The marked region contains partial colored evidence from a scaled version of the small object.

    Rule:
    - Use the four yellow pixels to crop the target region.
    - Generate scaled versions of the small object and try all placements inside that crop.
    - Choose a placement whose colored cells cover every existing non-black, non-yellow cell in the target region, then paste it.

    Key insight:
    - The partial colored cells inside the frame are alignment constraints for the scaled miniature;
      the yellow corners define the output crop.
    """
    # Plan:
    # 1. Detect the little object, yellow pixels, and big region
    # 2. Rescale and translate the little object to cover as much of the big region as possible, matching colors whenever they overlap

    # 1. Detect little object, yellow pixels, and big region
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="objects")
    yellow_pixel_objects = []
    for obj in objects:
        _b = bbox(obj != Color.BLACK)
        if _b is None:
            continue
        r0, c0, r1, c1 = _b
        if r0 == r1 and c0 == c1 and int(obj[r0, c0]) == int(Color.YELLOW):
            yellow_pixel_objects.append(obj)
    assert len(yellow_pixel_objects) == 4, "There should be exactly 4 yellow pixels"

    # Find the region indicated by the 4 yellow pixels
    yellow_coords = []
    for obj in yellow_pixel_objects:
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        yellow_coords.append(bbox_anchor(_b, anchor="center"))
    yellow_pixel_min_r = min(r for r, _c in yellow_coords)
    yellow_pixel_max_r = max(r for r, _c in yellow_coords)
    yellow_pixel_min_c = min(c for _r, c in yellow_coords)
    yellow_pixel_max_c = max(c for _r, c in yellow_coords)
    
    # The little object is what is outside that region
    little_objects = []
    for obj in objects:
        _b = bbox(obj != Color.BLACK)
        if _b is None:
            continue
        r, c = bbox_anchor(_b, anchor="center")
        if r < yellow_pixel_min_r or r > yellow_pixel_max_r or c < yellow_pixel_min_c or c > yellow_pixel_max_c:
            # Exclude the yellow corner markers
            if int(obj[r, c]) == int(Color.YELLOW):
                continue
            little_objects.append(obj)
    assert len(little_objects) == 1, "There should be exactly one little object"
    little_object = little_objects[0]

    # The output grid is going to be the region delimited by the yellow pixels
    output_grid = input_grid[yellow_pixel_min_r:yellow_pixel_max_r+1, yellow_pixel_min_c:yellow_pixel_max_c+1].copy()

    # The big region has some colored pixels that we are going to try and match with
    # Extract the content of big region as tuples (x, y, color)
    big_region_pixels = [ (r, c, output_grid[r, c])
                         for r in range(output_grid.shape[0]) for c in range(output_grid.shape[1])
                         if output_grid[r, c] != Color.BLACK and output_grid[r, c] != Color.YELLOW ]

    # 2. Rescale and translate the little object to cover as much of the big region as possible, matching colors whenever they overlap
    _b = bbox(little_object != Color.BLACK)
    assert _b is not None
    little_sprite = crop_bbox(little_object, _b)
    scaled_sprites = [ scale_sprite(little_sprite, factor) for factor in [1, 2, 3, 4, 5, 6] ]
    # A placement solution is a tuple of (x, y, scaled_sprite) where x, y is the top-left corner of the scaled sprite
    possible_solutions = [ (r, c, scaled_sprite) for scaled_sprite in scaled_sprites
                          for r in range(output_grid.shape[0] - scaled_sprite.shape[0] + 1)
                          for c in range(output_grid.shape[1] - scaled_sprite.shape[1] + 1) ]
    
    # Filter placement solutions to only those where the colors of the big region match the colors of the scaled+translated sprite
    def valid_solution(r, c, scaled_sprite):
        # Make a canvas to try putting down the scaled sprite
        test_canvas = np.full_like(output_grid, Color.BLACK)
        paste(test_canvas, scaled_sprite, r, c, transparent=Color.BLACK, clip=True)
        # Check if every big region color is also in the test canvas
        test_colors = [ (r, c, test_canvas[r, c]) for r in range(output_grid.shape[0]) for c in range(output_grid.shape[1]) if test_canvas[r, c] != Color.BLACK ]
        return all( (r, c, color) in test_colors for r, c, color in big_region_pixels )        
    
    possible_solutions = [ solution for solution in possible_solutions if valid_solution(*solution) ]
    if len(possible_solutions) == 0:
        assert False, "No solution found for the little object"
    
    # Pick the first solution and blit the sprite into the output grid
    r, c, scaled_sprite = list(possible_solutions)[0]
    paste(output_grid, scaled_sprite, r, c, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Create a little object and then rescale it to make a big version.
    # Cover up some parts of the big version with black pixels, and put yellow markers in the corners.
    # Finally figure out how to put everything on a big canvas

    little_sprite = random_sprite([2,3,4], [2,3,4], connectivity=4, color_palette=[color for color in Color.NOT_BLACK if color != Color.YELLOW], background=Color.BLACK)
    scaling_factor = np.random.randint(2, 4+1)
    big_sprite = scale_sprite(little_sprite, scaling_factor)

    # Cover up some parts of the big sprite with black pixels
    for _ in range(np.random.randint(1, 3)):
        x0, y0 = np.random.randint(0, big_sprite.shape[0]), np.random.randint(0, big_sprite.shape[1])
        x1, y1 = np.random.randint(x0, big_sprite.shape[0]), np.random.randint(y0, big_sprite.shape[1])
        big_sprite[x0:x1+1, y0:y1+1] = Color.BLACK

    # Create a fixed big canvas so we can safely place corner markers and keep the little object
    # entirely outside the marked big region.
    width = height = 30
    grid = np.full((width, height), Color.BLACK, dtype=int)
    
    # Put the big sprite in the canvas
    _loc = first_free_location(grid, big_sprite, transparent=Color.BLACK, border=6, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, big_sprite, r, c, transparent=Color.BLACK, clip=True)

    # Put yellow markers in the corners
    grid[r-1, c-1] = Color.YELLOW
    grid[r-1, c+big_sprite.shape[1]] = Color.YELLOW
    grid[r+big_sprite.shape[0], c-1] = Color.YELLOW
    grid[r+big_sprite.shape[0], c+big_sprite.shape[1]] = Color.YELLOW
    
    # Place the little sprite outside the marked region (top-left corner).
    paste(grid, little_sprite, 0, 0, transparent=Color.BLACK, clip=True)

    return grid
