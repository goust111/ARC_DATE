
import numpy as np
from typing import *
import random
from arc_utils import Color, TranslationalSymmetry, detect_translational_symmetry, draw_line, orbit, paste, random_sprite

# concepts:
# occlusion, translational symmetry

# description:
# In the input you will see a translationally symmetric pattern randomly occluded by black pixels.
# To make the output, remove the occluding black pixels to reveal the translationally symmetric pattern.

def main(input_grid):
    """
    Restore black occlusions in a translationally periodic pattern using matching orbit colors.

    Input structure:
    - The input is a colored periodic tiling with some cells replaced by black.
    - Black represents missing data rather than the pattern background.
    - Non-black cells reveal at least one translational symmetry.

    Rule:
    - Detect translations that preserve the visible non-black colors.
    - For each black cell, collect the non-black colors in its translational orbit.
    - Fill the cell with the unique color found in that orbit.

    Key insight:
    - Symmetry turns each missing pixel into a constraint-satisfaction problem over its orbit; the
      matching visible copies reveal the hidden color.
    """
    # Plan:
    # 1. Find the translational symmetries
    # 2. Reconstruct the sprite by ignoring the black pixels and exploiting the symmetry

    w, h = input_grid.shape

    # Identify the translational symmetries. Note that there is no background color for this problem.
    translations = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK], background=None)
    assert len(translations) > 0, "No translational symmetry found"

    # Reconstruct the occluded black pixels by replacing them with colors found in the orbit of the symmetries
    output_grid = np.copy(input_grid)
    for r in range(w):
        for c in range(h):
            if output_grid[r, c] == Color.BLACK:
                # Use the translational symmetry to fill in the occluded pixels
                # to do this we compute the ORBIT of the current pixel under the translations
                # and take the most common non-black color in the orbit

                # Compute the orbit into the output
                orbit_pixels = orbit(output_grid, r, c, translations)
                orbit_colors = {input_grid[transformed_r, transformed_c]
                                for transformed_r, transformed_c in orbit_pixels}
                
                # occluded by black, so whatever color it is, black doesn't count
                orbit_colors = orbit_colors - {Color.BLACK}

                # Copy the color
                assert len(orbit_colors) == 1, "Ambiguity: multiple colors in the orbit"
                output_grid[r, c] = orbit_colors.pop()
    
    return output_grid

def generate_input():
    # Build a clearly-periodic (but non-trivial) tiling, then occlude with sparse black pixels.
    w, h = random.randint(3, 6), random.randint(3, 6)
    H = w * random.randint(3, 5)
    W = h * random.randint(3, 5)
    grid = np.full((H, W), Color.BLACK, dtype=int)

    palette = random.sample(Color.NOT_BLACK, k=3)
    sprite = np.array([[random.choice(palette) for _ in range(h)] for _ in range(w)], dtype=int)

    for r0 in range(0, H, w):
        for c0 in range(0, W, h):
            paste(grid, sprite, r0, c0, transparent=Color.BLACK, clip=False)

    # Sparse occluders so symmetry detection doesn't hallucinate extra translations.
    n_occlude = random.randint(H * W // 40, H * W // 20)  # ~2.5%..5%
    for _ in range(n_occlude):
        rr = random.randrange(H)
        cc = random.randrange(W)
        grid[rr, cc] = Color.BLACK

    return grid
