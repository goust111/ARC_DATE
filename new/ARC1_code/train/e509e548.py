import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite


# concepts:
# topology, counting

# description:
# In the input you will see a grid with several green objects with different number of bends.
# To make the output, color the objects with one bend with blue, two bends with pink, three bends with red.

def main(input_grid):
    """
    Recolor each green object according to how many bends its shape contains.

    Input structure:
    - The grid has several green connected objects on a black background.
    - Each object is a small bent shape with one, two, or three corner bends.
    - The generated inputs include at least one object for each bend count.

    Rule:
    - Detect each green object and crop its bounding box.
    - Count L-shaped 2x2 bend patterns across all rotations.
    - Recolor objects with one, two, and three bends to blue, pink, and red respectively.

    Key insight:
    - Object color is determined by topology of the mask, not by position or size.
    """
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)

    # Find all the green objects in the grid
    object_color = Color.GREEN
    background = Color.BLACK
    objects = find_components(input_grid, bg=background, monochromatic=True, connectivity=4, return_format="objects")
    for obj in objects:
        # Get the bounding box of the sprite and crop the sprite
        b = bbox(obj != background)
        assert b is not None
        r0, c0, r1, c1 = b
        sprite = crop_bbox(obj, b)
        mask = sprite != background

        # Determine how many bends the mask has
        # Do this by building an L-shaped mask, and seeing how many times it appears in the sprite
        bend_mask = np.array([[1, 1], [1, 0]])
        rotated_bend_masks = [bend_mask, np.rot90(bend_mask), np.rot90(bend_mask, 2), np.rot90(bend_mask, 3)]

        from scipy.ndimage import correlate
        
        n_bends = sum( np.sum( correlate(mask*1, filter*1, mode='constant', cval=0) == np.sum(filter*1) ) for filter in rotated_bend_masks )

        # find the new color based on bends
        new_color = {1: Color.BLUE, 2: Color.PINK, 3: Color.RED}[int(n_bends)]
        # color the sprite with the new color
        sprite[sprite == object_color] = new_color
        output_grid = paste(output_grid, sprite, r0, c0, transparent=background)

    return output_grid

def generate_input():
    # Generate grid of size n x m
    n, m = np.random.randint(15, 25), np.random.randint(15, 25)
    grid = np.zeros((m, n), dtype=int)

    # The objects color is green
    object_color = Color.GREEN

    # keep track of what bends we've already created, so that we end up with at least one of each bend
    bends_we_have = set()

    for _ in range(random.randint(3, 6)):
        # Randomly generate sprite with one to three bends
        w, h = np.random.randint(3, 6), np.random.randint(3, 6)

        # Generate sprite with the object color
        sprite = random_sprite(w, h, color_palette=[object_color], symmetry="not_symmetric", density=0.3)

        # calculate how many bends it has and make sure that the number of bends are 1-3
        mask = sprite != Color.BLACK
        bend_mask = np.array([[1, 1], [1, 0]])
        rotated_bend_masks = [bend_mask, np.rot90(bend_mask, k=-1), np.rot90(bend_mask,-(2)), np.rot90(bend_mask,-(3))]
        from scipy.ndimage import correlate
        n_bends = sum( np.sum( correlate(mask*1, filter*1, mode='constant', cval=0) == np.sum(filter*1) ) for filter in rotated_bend_masks )
        if n_bends not in [1, 2, 3]: continue

        # keep track of what bends we've already created
        bends_we_have.add(n_bends)

        # Randomly place the sprite on the grid
        _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc

        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    # The input grid should have at least one object one type of bend as example.
    if len(bends_we_have) < 3:
        return generate_input()
    
    return grid
# ============= remove below this point for prompting =============
