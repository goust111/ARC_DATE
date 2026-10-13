
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, collision, colors_present, crop_bbox, detect_mirror_symmetry, detect_objects, draw_line, first_free_location, paste, random_sprite

# concepts:
# symmetry, occlusion

# description:
# In the input you will see a left-right symmetric monochromatic object occluded by a colored rectangle
# To make the output, remove the colored rectangle and fill in the missing parts of the object to make it left-right symmetric

def main(input_grid):
    """
    Remove a filled rectangular occluder and mirror the remaining object to restore symmetry.

    Input structure:
    - A black grid contains a monochromatic object with left-right mirror symmetry.
    - A different-colored filled rectangle overlaps and hides part of that object.
    - The rectangle is itself a solid connected component.

    Rule:
    - Detect the solid rectangle component and erase it to black.
    - Detect mirror symmetry while ignoring the rectangle color.
    - Reflect every remaining object pixel across the detected mirror axes to fill the missing cells.

    Key insight:
    - The occluder color should be ignored for symmetry detection, while black remains meaningful absence.
    """
    # Plan:
    # 1. Extract and separate the rectangle from the symmetric object
    # 2. Find the left-right symmetry
    # 3. Fill in the missing parts of the object

    background_color = Color.BLACK

    # Each object has a different color, so we can look at connected components by color
    objects = detect_objects(input_grid, monochromatic=True, connectivity=8, background=background_color)
    sprites = []
    for obj in objects:
        _b = bbox(obj != background_color)
        if _b is None:
            continue
        sprites.append(crop_bbox(obj, _b))
    # Find the rectangle
    for obj, sprite in zip(objects, sprites):
        # the rectangle will be completely filled, so we can check if its total area is the whole rectangular region
        if sprite.shape[0] * sprite.shape[1] == np.sum(sprite != Color.BLACK):
            rectangle = obj
            break

    # Find the color of the rectangle, because it is the occluder
    rectangle_color = colors_present(rectangle, exclude={ background_color })[0]
    
    # Delete the rectangle
    rectangle_mask = rectangle != Color.BLACK
    output_grid = input_grid.copy()
    output_grid[rectangle_mask] = Color.BLACK

    # Find the symmetry
    # The occluder is rectangle_color, so we ignore it. In contrast, Color.BLACK is places where the object *actually* isn't located, so we can't ignore that.
    mirrors = detect_mirror_symmetry(input_grid, ignore_colors=[rectangle_color], background=Color.BLACK)

    # Mirror each colored pixel
    for r, c in np.argwhere(output_grid != Color.BLACK):
        for mirror in mirrors:
            source_color = output_grid[r,c]
            destination = mirror.apply(r, c)
            output_grid[destination] = source_color

    return output_grid


def generate_input():
    # Create a medium sized grid
    n, m = np.random.randint(10, 15), np.random.randint(10, 15)
    grid = np.zeros((m, n), dtype=int)

    object_color, rectangle_color = random.sample(list(Color.NOT_BLACK), 2)

    sprite = random_sprite(np.random.randint(5, m-2), np.random.randint(5, n-2), density=0.5, symmetry="horizontal", color_palette=[object_color])
    rectangle = np.full((np.random.randint(2, 5), np.random.randint(2, 5)), rectangle_color)

    _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    sprite_r, sprite_c = _loc
    paste(grid, sprite, sprite_r, sprite_c, transparent=Color.BLACK, clip=True)

    # Find a random place for the rectangle that is NOT free (because the rectangle occludes the sprite)
    while True:
        rectangle_r, rectangle_c = np.random.randint(0, m-rectangle.shape[0]), np.random.randint(0, n-rectangle.shape[1])
        if collision(object1=grid, object2=rectangle, r2=rectangle_r, c2=rectangle_c, background=Color.BLACK):
            # Equivalently, could have done:
            # collision(object1=sprite, object2=rectangle, r1=sprite_x, c1=sprite_y, r2=rectangle_x, c2=rectangle_y)
            break
    paste(grid, rectangle, rectangle_r, rectangle_c, transparent=Color.BLACK, clip=True)
    return grid
    
