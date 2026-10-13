import numpy as np
import random
from arc_utils import Color, draw_line, paste, random_sprite


# concepts:
# repeating patterns, colors as indicators, scaling

# description:
# In the input you will see a nxm sprite with black background. 
# Construct an output grid with n^2 x m^2 black pixels. Divide the output grid into subgrids, 
# and look at the corresponding pixel in the nxm input grid. If the corresponding pixel is not black, 
# then copy the nxm input grid into the subgrid. Else, the subgrid does not change. 

def main(input_grid):
    """
    Expand a small sprite into a block grid by stamping the whole sprite wherever the original has a non-black pixel.

    Input structure:
    - The input is a small rectangular sprite on a black background.
    - Non-black pixels mark which output blocks should receive a copy of the sprite.
    - The output dimensions are the input height squared by the input width squared.

    Rule:
    - Partition the output into input-sized blocks, one block per input cell.
    - For each non-black input cell, copy the full input sprite into the corresponding output block.
    - Leave blocks corresponding to black input cells entirely black.

    Key insight:
    - The input acts as both the stencil and the stamp: colored cells select block positions,
      and the same sprite is pasted into each selected block.
    """
    # creates an empty 9x9 output grid 
    output_grid = np.zeros((input_grid.shape[0]**2,input_grid.shape[1]**2),dtype=int)

    input_sprite = input_grid

    # Go through the input grid. If an input grid pixel is not black, 
    # then copy the input grid to the corresponding location on the output grid
    H, W = input_grid.shape
    for r in range(H):
        for c in range(W):
            if int(input_grid[r, c]) != Color.BLACK:
                output_grid = paste(output_grid, input_sprite, r * H, c * W, transparent=Color.BLACK)
    
    return output_grid

def generate_input():
  n,m = random.randint(3, 6), random.randint(3, 6)
  random_color = random.choice(list(Color.NOT_BLACK))
  return random_sprite(n, m, color_palette=[random_color])
# ============= remove below this point for prompting =============
