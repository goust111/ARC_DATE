import numpy as np
from arc_utils import Color, bbox, detect_objects, draw_line, randomly_scatter_points

# concepts:
# boundary detection, object extraction

# description:
# In the input you will see several teal pixels and two vertical parallel gray lines with four teal pixels indicates the boundary of the output grid.
# To make the output grid, you should extract the part of grid that is bounded by the two vertical parallel gray lines and four teal pixels in each corner.

def main(input_grid):
    """
    Crop out the rectangle bounded by the two gray vertical rails and teal corner markers.

    Input structure:
    - The grid has a black background with scattered teal pixels.
    - Two vertical gray line objects form the left and right sides of the target crop.
    - Teal corner pixels sit one row beyond the gray line endpoints.

    Rule:
    - Detect the gray vertical line objects and order them from left to right.
    - Crop between their columns and extend one row above and below their line extents.
    - Clamp the crop to the input canvas.

    Key insight:
    - The gray rails provide the side boundaries, and the row expansion includes the teal
      corner markers that complete the rectangle.
    """
    bg = Color.BLACK
    # Detect the two vertical gray lines.
    vertical_lines = detect_objects(grid=input_grid, colors=[Color.GREY], monochromatic=True, connectivity=4, background=bg)
    bbs = []
    for line in vertical_lines:
        b = bbox(line != bg)
        if b is not None:
            bbs.append(b)
    assert len(bbs) >= 2

    # Sort left-to-right by column.
    bbs.sort(key=lambda b: b[1])
    left, right = bbs[0], bbs[1]

    r0 = min(left[0], right[0]) - 1
    r1 = max(left[2], right[2]) + 1
    c0 = min(left[1], right[1])
    c1 = max(left[3], right[3])

    H, W = input_grid.shape
    r0 = max(0, int(r0))
    c0 = max(0, int(c0))
    r1 = min(H - 1, int(r1))
    c1 = min(W - 1, int(c1))
    return input_grid[r0 : r1 + 1, c0 : c1 + 1]

def generate_input():
    # Generate the background grid with size of m x n.
    n, m = np.random.randint(9, 15), np.random.randint(9, 15)
    grid = np.zeros((m, n), dtype=int)

    # Generate random teal pixels on the grid.    
    randomly_scatter_points(grid, color=Color.TEAL, density=0.2)

    # Randomly pick the bounded rectangle size and location.
    # We want two VERTICAL gray lines at the left/right boundary columns, and four teal corner pixels.
    # The gray lines should NOT include the corners (so the solver's +/-1 row expansion captures them).
    box_h = np.random.randint(5, m)  # includes corners; >= 5 => gray line length >= 3
    box_w = np.random.randint(4, n)  # includes boundary columns; >= 4 => visible interior
    r0 = np.random.randint(0, m - box_h + 1)
    c0 = np.random.randint(0, n - box_w + 1)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    # Draw two vertical gray boundary lines (excluding the teal corners).
    draw_line(grid=grid, r=r0 + 1, c=c0, color=Color.GREY, direction=(1, 0), length=box_h - 2)
    draw_line(grid=grid, r=r0 + 1, c=c1, color=Color.GREY, direction=(1, 0), length=box_h - 2)

    # Teal corners.
    grid[r0, c0] = Color.TEAL
    grid[r0, c1] = Color.TEAL
    grid[r1, c0] = Color.TEAL
    grid[r1, c1] = Color.TEAL

    return grid

# ============= remove below this point for prompting =============
