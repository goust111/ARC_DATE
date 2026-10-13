
import numpy as np
from typing import *
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite
import random

# concepts:
# symmetries, objects

# description:
# In the input you will see one or two medium-sized multicolor objects, and some multicolor pixels sprinkled around in one or two clusters
# To make the output, take each of the medium-sized object and move it so that it perfectly covers some of the pixels sprinkled around, matching colors wherever they overlap.
# You can rotate and flip the objects in order to make them fit the pixels, but they have to match colors wherever they overlap.

def main(input_grid):
    """
    Place each large multicolor object over the sparse matching pixels after trying rotations and flips.

    Input structure:
    - The input contains one or two larger multicolor objects on a black background.
    - It also contains small clusters of colored pixels that are partial views of those objects.
    - Objects may need rotation or reflection to align with their pixel guides.

    Rule:
    - Separate large objects from small pixel clusters.
    - Generate rotated and flipped variants of each large object.
    - Choose the placement and variant that covers the most guide pixels without any color mismatch.

    Key insight:
    - The sparse pixels are anchors for transformed copies: color-consistent overlap identifies both
      the orientation and the position of each object.
    """
    # Plan:
    # 1. Break the input into 1-2 big object(s) and a mask containing the remaining pixels
    # 2. For each object, create rotations and flips of it
    # 3. For each object, find the best way to place one of its rotations/flips so that it covers the most number of the pixels (remember the pixel has to match object color)

    # Extract the objects from the input and categorize them into objects and pixels
    # Objects can be multicolored
    connected_components = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="objects")
    objects = [ cc for cc in connected_components if np.count_nonzero(cc != Color.BLACK) > 4 ]
    pixels = [ cc for cc in connected_components if np.count_nonzero(cc != Color.BLACK) <= 4 ]

    # Make the pixel mask, which shows where the pixels are. These guide the placement of objects.
    pixel_mask = np.full(input_grid.shape, Color.BLACK)
    for pixel_object in pixels:
        paste(pixel_mask, pixel_object, 0, 0, transparent=Color.BLACK, clip=True)
    
    output_grid = np.full(input_grid.shape, Color.BLACK)
    
    # For each object, find the best way to place it so that it covers the most number of the pixels
    for obj in objects:
        # The object can be rotated and flipped to match the pixels
        # First, convert it to a sprite before transforming, because these operations are independent of position
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        sprite = crop_bbox(obj, _b)
        sprite_variations = [sprite, np.rot90(sprite), np.rot90(sprite, 2), np.rot90(sprite, 3), np.flipud(sprite), np.fliplr(sprite), np.flipud(np.rot90(sprite)), np.fliplr(np.rot90(sprite))]

        # We are going to optimize the position and variation, so we need to keep track of the best placement so far
        best_output, best_pixels_covered = None, 0
        for r, c, sprite_variation in [(r, c, variant) for r in range(input_grid.shape[0]) for c in range(input_grid.shape[1]) for variant in sprite_variations]:
            test_grid = np.copy(output_grid)
            paste(test_grid, sprite_variation, r, c, transparent=Color.BLACK, clip=True)
            # Check if there was any color mismatch: A colored pixel in the mask which is different from what we just made
            # If there is a mismatch, we can't place the object here
            if np.any((pixel_mask != Color.BLACK) & (test_grid != Color.BLACK) & (pixel_mask != test_grid)):
                continue
            num_covered_pixels = np.count_nonzero((pixel_mask != Color.BLACK) & (test_grid != Color.BLACK))
            if num_covered_pixels > best_pixels_covered:
                best_output, best_pixels_covered = test_grid, num_covered_pixels
        output_grid = best_output

    return output_grid


def generate_input():
    # to make the input, we make random objects
    # we need to make sure that the objects can match with the "sprinkled" pixels, so we make the pixels by placing some objects and then removing some parts of them

    n, m = np.random.randint(10, 20), np.random.randint(10, 20)
    grid = np.full((m, n), Color.BLACK)

    num_objects = np.random.randint(1, 3)
    sprites = [ random_sprite(np.random.randint(3, 6), np.random.randint(3, 6), color_palette=Color.NOT_BLACK, density=0.5)
                for _ in range(num_objects) ]
    sprite_variations = [ [sprite, np.rot90(sprite, k=-1), np.rot90(sprite,-(2)), np.rot90(sprite,-(3)), np.fliplr(sprite), np.flipud(sprite), np.fliplr(np.rot90(sprite, k=-1)), np.flipud(np.rot90(sprite, k=-1))]
                          for sprite in sprites ]
    # Pick a variation for each sprite
    sprite_variations = [ random.choice(variations) for variations in sprite_variations ]
    
    # Remove parts of the sprite to create the "sprinkled" pixels
    # We remove enough of the sprite so that there are only 3 pixels remaining
    # We do this to the variation, which is what is going to be put on the canvas to be matched with
    occluded_sprites = []
    for sprite in sprite_variations:
        occluded_sprite = np.copy(sprite)
        could_be_removed = np.argwhere(occluded_sprite != Color.BLACK)
        np.random.shuffle(could_be_removed)
        for i in range(len(could_be_removed) - 3):
            r, c = could_be_removed[i]
            occluded_sprite[r, c] = Color.BLACK

        occluded_sprites.append(occluded_sprite)

    # place everything on the canvas but make sure that nothing overlaps
    for sprite in sprites + occluded_sprites:
        _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=2, padding=2, padding_connectivity=8)
        if _loc is None:
            # Retry generation (placement can fail due to padding constraints).
            return generate_input()
        r, c = _loc
        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return grid     
