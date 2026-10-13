import numpy as np
import random
from arc_utils import Color, detect_translational_symmetry, draw_line, orbit, random_sprite


# concepts:
# translational symmetry, symmetry detection

# description:
# In the input you will see a grid consisting of a blue sprite that is repeatedly translated vertically, forming a stack of the same sprite.
# To make the output, expand the input to have height 9, and continue to repeatedly translate the sprite vertically. Change color to red.
 
def main(input_grid):
    """
    Extend a repeated blue sprite pattern to a 9-cell span and recolor the repeated foreground red.

    Input structure:
    - The input is a black-background grid containing a blue sprite repeated by translation.
    - The repetition may run horizontally or vertically.
    - The visible grid is shorter than the required completed span.

    Rule:
    - Infer the dominant translation period from matching blue pixels.
    - Tile the corresponding base slice until the repeated axis reaches length 9.
    - Recolor all blue foreground pixels red in the completed pattern.

    Key insight:
    - The repeated sprite supplies its own period; once the period is inferred, completion is just
      tiling along that axis followed by a color change.
    """
    # Plan:
    # 1. Find the repeated translation, which is a symmetry
    # 2. Extend the pattern by copying the sprite and its symmetric copies
    # 3. Change the color from blue to red
    
    bg = Color.BLACK
    H, W = input_grid.shape

    # Detect repetition period along columns (horizontal translation) and rows (vertical translation),
    # ignoring fully-filled rows/cols (which can create trivial period=1 symmetries).
    informative_rows = [r for r in range(H) if np.any(input_grid[r, :] == bg) and np.any(input_grid[r, :] != bg)]
    informative_cols = [c for c in range(W) if np.any(input_grid[:, c] == bg) and np.any(input_grid[:, c] != bg)]

    best_dc, best_dc_score = None, 0
    for dc in range(1, W):
        score = 0
        for r in informative_rows:
            row = input_grid[r, :]
            for c in range(W - dc):
                if int(row[c]) != bg and int(row[c + dc]) == int(row[c]):
                    score += 1
        if score > best_dc_score:
            best_dc_score = score
            best_dc = int(dc)

    best_dr, best_dr_score = None, 0
    for dr in range(1, H):
        score = 0
        for c in informative_cols:
            col = input_grid[:, c]
            for r in range(H - dr):
                if int(col[r]) != bg and int(col[r + dr]) == int(col[r]):
                    score += 1
        if score > best_dr_score:
            best_dr_score = score
            best_dr = int(dr)

    weight_dc = (best_dc_score * best_dc) if (best_dc is not None) else 0
    weight_dr = (best_dr_score * best_dr) if (best_dr is not None) else 0

    if best_dc is not None and weight_dc >= weight_dr:
        p = int(best_dc)
        base = input_grid[:, :p]
        reps = (9 + p - 1) // p
        tiled = np.tile(base, (1, reps))
        output_grid = tiled[:, :9].copy()
    else:
        p = int(best_dr) if best_dr is not None else H
        base = input_grid[:p, :]
        reps = (9 + p - 1) // p
        tiled = np.tile(base, (reps, 1))
        output_grid = tiled[:9, :].copy()

    output_grid[output_grid == Color.BLUE] = Color.RED
    return output_grid


def generate_input():
    # grid is always 3x6
    grid = np.zeros((3, 6),dtype = int)

    # The input is always blue
    color = Color.BLUE

    # Creates a random smaller sprite, where the height (period) is chosen randomly
    height = random.randint(2, 7)
    sprite = random_sprite(3, height, symmetry="not_symmetric", color_palette=[color], density=0.4, connectivity=8)

    # place the smaller pattern, tiling it so that it is repeated vertically
    # tile "infinitely" (x100)
    vertically_repeated = np.tile(sprite, (1, 100))
    # crop to the size of the grid
    vertically_repeated = vertically_repeated[:, :grid.shape[1]]
    # copy to the grid
    grid[:,:] = vertically_repeated

    return grid
# ============= remove below this point for prompting =============
