import numpy as np

import random
from arc_utils import Color, bbox, crop_bbox, detect_translational_symmetry, draw_line, orbit, paste, random_sprite

# concepts:
# symmetry detection, occlusion

# description:
# In the input you will see a translationally symmetric pattern that has been partially occluded by a black rectangle
# The output should be what the black rectangle should be in order to make it perfectly symmetric.
# In other words, the output should be the missing part of the pattern, and it should be the same dimensions as the black rectangle.

def main(input_grid):
    """
    Recover the black occluded rectangle from a translationally symmetric pattern.

    Input structure:
    - The input is a repeated color pattern with no true background color.
    - A black rectangle masks part of the pattern.
    - Non-black cells elsewhere reveal the translational symmetry.

    Rule:
    - Treat black cells as the occlusion mask.
    - Detect translational symmetries while ignoring black.
    - Fill each black cell from a non-black cell in its symmetry orbit, then crop the filled rectangle.

    Key insight:
    - Symmetry orbits provide equivalent unoccluded samples for each missing cell, and
      the answer is only the reconstructed occluded region.
    """
    # Plan:
    # 1. Find the black rectangle
    # 2. Find the symmetry
    # 3. Fill in the missing part
    # 4. Extract the part that you filled in, which is the final output

    # Find the black rectangle and save where it is
    # Do this first because we will need to know where it was after we fill it in
    occlusion_color = Color.BLACK
    black_rectangle_mask = (input_grid == occlusion_color)

    # Find the symmetry. Notice that black is not the background, but instead is the occlusion color. In fact there is no well-defined background color.
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[occlusion_color], background=None)

    # Fill in the missing part
    for occluded_r, occluded_c in np.argwhere(black_rectangle_mask):
        for sym_r, sym_c in orbit(input_grid, int(occluded_r), int(occluded_c), symmetries):
            if int(input_grid[sym_r, sym_c]) != occlusion_color:
                input_grid[int(occluded_r), int(occluded_c)] = input_grid[sym_r, sym_c]
                break
    
    # Extract the region that we filled in, ultimately as a 2D sprite
    # first, get just the part of the final image which corresponds to what used to be included
    filled_in_region = np.full_like(input_grid, occlusion_color)
    filled_in_region[black_rectangle_mask] = input_grid[black_rectangle_mask]
    # then, crop it to obtain the sprite
    b = bbox(filled_in_region != occlusion_color)
    assert b is not None
    filled_in_region = crop_bbox(filled_in_region, b)

    return filled_in_region
    


def generate_input():
    # Plan:
    # 1. Make a random sprite
    # 2. Tile it to make a symmetric pattern
    # 3. Occlude it with a black rectangle

    # Make a random sprite with NO black pixels (so the occlusion is unambiguous).
    sh, sw = np.random.randint(2, 5, size=(2))
    sprite = np.random.choice(Color.NOT_BLACK, size=(int(sh), int(sw)), replace=True).astype(int)

    # Tile it to make a perfectly translationally symmetric pattern.
    rep_r, rep_c = np.random.randint(3, 6, size=(2))
    pattern = np.tile(sprite, (int(rep_r), int(rep_c)))

    # Occlude within a single tile (so every occluded cell has a non-occluded counterpart in its orbit).
    oh = int(np.random.randint(1, int(sh) + 1))
    ow = int(np.random.randint(1, int(sw) + 1))
    tile_r = int(np.random.randint(0, int(rep_r)))
    tile_c = int(np.random.randint(0, int(rep_c)))
    r0 = tile_r * int(sh) + int(np.random.randint(0, int(sh) - oh + 1))
    c0 = tile_c * int(sw) + int(np.random.randint(0, int(sw) - ow + 1))
    pattern[r0 : r0 + oh, c0 : c0 + ow] = Color.BLACK

    return pattern
    

# ============= remove below this point for prompting =============
