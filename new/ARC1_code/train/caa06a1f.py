
import random
import numpy as np
from typing import *
from arc_utils import Color, detect_translational_symmetry, draw_line, orbit, paste

# concepts:
# translational symmetry, symmetry detection, non-black background

# description:
# In the input you will see a a translationally symmetric pattern that does not extend to cover the entire canvas. The background is not black.
# To make the output, continue the symmetric pattern until it covers the entire canvas, but shift everything right by one pixel.
 
def main(input_grid):
    """
    Continue the translational pattern across the canvas and shift its phase one column right.

    Input structure:
    - The background is a non-black color inferred from the border.
    - Foreground pixels form a partially tiled translational pattern.
    - The pattern does not cover the full canvas in the input.

    Rule:
    - Infer the background color from the most common border color.
    - Detect translational symmetries of the foreground pattern.
    - Fill a one-column-wider canvas by symmetry orbits, then drop the first column.

    Key insight:
    - Extending before cropping creates the rightward phase shift without wraparound artifacts.
    """
    # Plan:
    # 1. Find the background color
    # 2. Find the repeated translation, which is a symmetry
    # 3. Extend the pattern by computing the orbit of each pixel in the pattern

    # Find the background color which is the most common color along the border of the canvas
    pixels_on_border = np.concatenate([input_grid[0, :], input_grid[-1, :], input_grid[:, 0], input_grid[:, -1]])
    background = max(set(pixels_on_border), key=list(pixels_on_border).count)
    
    # Find the repeated translation, which is a symmetry
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[], background=background)
    assert len(symmetries) > 0, "No translational symmetry found"

    # We want an *infinite-tiling* style shift (no wrap artifacts), so we:
    # - create a canvas that is 1 column wider
    # - extend the symmetric pattern into that wider canvas
    # - take a 1-column-shifted window (crop) to get the desired phase shift
    H, W = input_grid.shape
    wide = np.full((H, W + 1), background)

    for r, c in np.argwhere(input_grid != background):
        r = int(r)
        c = int(c)
        col = int(input_grid[r, c])
        for rr, cc in orbit(wide, r, c, symmetries):
            wide[rr, cc] = col

    # Drop the first column to restore the original shape; this yields the shifted phase.
    return wide[:, 1:]


def generate_input():

    background_color = random.choice(Color.NOT_BLACK)
    # Make a random large canvas
    grid = np.full((np.random.randint(15, 30), np.random.randint(15, 30)), background_color)

    # Make the basic sprite
    w, h = random.randint(2, 4), random.randint(2, 4)
    sprite = np.random.choice([color for color in Color.ALL_COLORS if color != background_color], (w, h))

    # Tile it a few times, starting from the upper left hand corner
    max_r = random.randint(w+1, grid.shape[0])
    max_c = random.randint(h+1, grid.shape[1])
    for r in range(0, max_r, w):
        for c in range(0, max_c, h):
            paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
