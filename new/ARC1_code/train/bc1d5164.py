import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, paste

# concepts:
# patterns, positioning, copying

# description:
# In the input you will see a pattern of pixels in the top left corner of the grid, the top right corner of the grid, the bottom left corner of the grid, and the bottom right corner of the grid. All the pixels are the same color, and the patterns are in square regions.
# To make the output, copy the pattern in each corner of the input to the corresponding corner of the output. The output grid is one pixel larger in each dimension than the maximum pattern side length.

def main(input_grid):
    """
    Copy the four corner patterns into a minimal square output frame.

    Input structure:
    - Same-color patterns occupy square regions in the four input corners.
    - Black cells separate the corner regions.
    - At least one foreground pixel is present.

    Rule:
    - Assign foreground pixels to their nearest input corner.
    - Infer the largest needed corner square side length.
    - Copy each corner square into the corresponding corner of an output one cell larger.

    Key insight:
    - The task preserves each pattern's corner-relative offsets while shrinking the canvas.
    """
    bg = Color.BLACK
    H, W = input_grid.shape

    # Infer the square "corner region" side length k from the farthest non-black pixel
    # in each corner, after assigning pixels to their nearest corner.
    coords = np.argwhere(input_grid != bg)
    assert coords.size > 0

    # Corner assignments by Manhattan distance to each corner.
    # corner order: TL, TR, BL, BR
    d_tl = coords[:, 0] + coords[:, 1]
    d_tr = coords[:, 0] + (W - 1 - coords[:, 1])
    d_bl = (H - 1 - coords[:, 0]) + coords[:, 1]
    d_br = (H - 1 - coords[:, 0]) + (W - 1 - coords[:, 1])
    ds = np.stack([d_tl, d_tr, d_bl, d_br], axis=1)
    corner_idx = ds.argmin(axis=1)

    k_candidates: list[int] = []
    for i in range(4):
        pts = coords[corner_idx == i]
        if pts.size == 0:
            continue
        rs = pts[:, 0]
        cs = pts[:, 1]
        if i == 0:  # TL
            k_i = max(int(rs.max() + 1), int(cs.max() + 1))
        elif i == 1:  # TR
            k_i = max(int(rs.max() + 1), int(W - cs.min()))
        elif i == 2:  # BL
            k_i = max(int(H - rs.min()), int(cs.max() + 1))
        else:  # BR
            k_i = max(int(H - rs.min()), int(W - cs.min()))
        k_candidates.append(int(k_i))

    k = max(k_candidates)
    n = m = int(k) + 1
    output_grid = np.full((n, m), bg, dtype=int)

    # Copy each kxk corner region into the corresponding corner of the output,
    # preserving the pattern's internal offset within that corner square.
    tl = input_grid[0:k, 0:k]
    tr = input_grid[0:k, W - k : W]
    bl = input_grid[H - k : H, 0:k]
    br = input_grid[H - k : H, W - k : W]

    output_grid = paste(output_grid, tl, 0, 0, transparent=bg)
    output_grid = paste(output_grid, tr, 0, m - k, transparent=bg)
    output_grid = paste(output_grid, bl, n - k, 0, transparent=bg)
    output_grid = paste(output_grid, br, n - k, m - k, transparent=bg)
    return output_grid
    

def generate_input():
    # make a random sized grid with black background
    n = np.random.randint(5, 8)
    m = np.random.randint(5, 8)
    grid = np.zeros((n, m), dtype=int)

    # select a color for the patterns
    color = np.random.choice(list(Color.NOT_BLACK))

    # select a size for the patterns so that there will be space between the patterns after they are in their corners
    size = np.random.randint(2, (min(n, m) + 1) // 2)


    # make a random pattern in the top left corner of the specified size
    grid[:size, :size] = [[np.random.choice([color, Color.BLACK]) for _ in range(size)] for _ in range(size)]

    # make a random pattern in the top right corner of the specified size
    grid[:size, -size:] = [[np.random.choice([color, Color.BLACK]) for _ in range(size)] for _ in range(size)]

    # make a random pattern in the bottom left corner of the specified size
    grid[-size:, :size] = [[np.random.choice([color, Color.BLACK]) for _ in range(size)] for _ in range(size)]

    # make a random pattern in the bottom right corner of the specified size
    grid[-size:, -size:] = [[np.random.choice([color, Color.BLACK]) for _ in range(size)] for _ in range(size)]

    # Check that at least one of the patterns is not all black
    # If they are all black, try again
    if np.all(grid == Color.BLACK):
        return generate_input()

    return grid


# ============= remove below this point for prompting =============
