import numpy as np
import random
from arc_utils import Color, bbox, draw_line, find_components, paste, randomly_spaced_indices

# concepts:
# objects separated by lines, color correspond to position

# description:
# In the input you will see several gray lines that separate the grid into several parts.
# To make the output grid, you should color the upper left part with blue, the lower right part with green,
# and the middle part with red.

def main(input_grid):
    """
    Color selected divider-bounded regions by their relative positions.

    Input structure:
    - Gray lines partition a black canvas into multiple regions.
    - At least three non-gray regions are present.
    - Regions are selected by upper-left, middle, and lower-right position.

    Rule:
    - Treat gray as the divider color and find connected regions between dividers.
    - Fill the upper-left region blue and the lower-right region green.
    - Fill the middle positional candidate red while leaving dividers unchanged.

    Key insight:
    - The regions are not identified by shape; their sorted positions determine their colors.
    """
    divider = Color.GREY
    regions = find_components(input_grid, bg=divider, connectivity=4, monochromatic=False, return_format="objects")

    bbs = []
    for obj in regions:
        b = bbox(obj != divider)
        if b is not None:
            bbs.append(b)
    assert len(bbs) >= 3

    # Choose TL, BR, and "middle" by position.
    bbs_sorted = sorted(bbs, key=lambda b: (b[0], b[1]))
    tl = bbs_sorted[0]
    br = max(bbs_sorted, key=lambda b: (b[0] + b[1]))
    mid_candidates = [b for b in bbs_sorted if b not in (tl, br)]
    mid = mid_candidates[len(mid_candidates) // 2]

    out = input_grid.copy()
    for b, color in [(tl, Color.BLUE), (mid, Color.RED), (br, Color.GREEN)]:
        r0, c0, r1, c1 = b
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out

def generate_input():
    # Generate a square grid as the black background
    # Need enough room for up to 4 lines with the requested padding.
    grid_len = np.random.randint(19, 30)
    n, m = grid_len, grid_len
    grid = np.zeros((m, n), dtype=int)

    # Get the number of horizontal and vertical lines
    # Keep line counts modest so `randomly_spaced_indices` is always feasible.
    line_nums = [2, 3]
    line_num = np.random.choice(line_nums)

    # Generate the gray lines' positions with interval between each line to form several rectangles
    # If padding is True, the lines will be padded with 1 grid on both sides
    vertical_lines = randomly_spaced_indices(max_len=n, n_indices=line_num, border_size=1, padding=3)
    horizontal_lines = randomly_spaced_indices(max_len=m, n_indices=line_num, border_size=1, padding=2)

    # Draw the gray vertical and horizontal lines on the grid
    for c in vertical_lines:
        draw_line(grid=grid, r=0, c=int(c), direction=(1, 0), color=Color.GRAY)
    
    for r in horizontal_lines:
        draw_line(grid=grid, r=int(r), c=0, direction=(0, 1), color=Color.GRAY)

    return grid
# ============= remove below this point for prompting =============
