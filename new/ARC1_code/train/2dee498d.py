import numpy as np
import random
from arc_utils import Color, draw_line, paste, random_sprite


# concepts:
# translational symmetry, symmetry detection, reflection

# description:
# In the input, you will see a sprite repeated horizontally, and some of those repetitions might be reflected top/down/right/left.
# To make the output, just extract the repeated sprite.

def main(input_grid):
    """
    Return the leftmost repeated slice that can tile the grid horizontally, allowing reflected copies.

    Input structure:
    - The grid is composed of equal-width column blocks.
    - Each block is expected to match the first block directly, upside down, or left-right flipped.
    - The block width divides the overall grid width.

    Rule:
    - Try candidate periods from smallest to largest.
    - Compare each block to the first block and to its vertical and horizontal reflections.
    - Return the first leftmost block whose variants explain all blocks.

    Key insight:
    - The original sprite is recovered as the smallest horizontal period consistent with all reflected repetitions.
    """
    # Find the period along the horizontal repetition axis (columns),
    # allowing each repetition to be a reflection of the base sprite.
    H, W = input_grid.shape
    for period in range(1, W + 1):
        if W % period != 0:
            continue
        sprite = input_grid[:, :period]
        reps = [input_grid[:, i * period : (i + 1) * period] for i in range(W // period)]

        valid = True
        for rep in reps:
            reflections = [rep, np.flipud(rep), np.fliplr(rep)]
            if not any(np.array_equal(sprite, r) for r in reflections):
                valid = False
                break
        if valid:
            return sprite
        
    assert False, "No valid period found"

def generate_input():
    # Create the sprite to be duplicated; pick a trio of random colors
    n = random.randint(2,7)
    sprite = random_sprite(n, n, color_palette=random.sample(list(Color.NOT_BLACK),3))
    
    # Duplicate the sprite 3 times horizontally
    grid = np.zeros((3*n, n),dtype=int)
    for i in range(3):
        paste(grid, sprite, i*n, 0, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
