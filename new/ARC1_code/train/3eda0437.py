import numpy as np
import random
from arc_utils import Color, draw_line, paste, randomly_scatter_points


# concepts:
# rectangle detection

# description:
# In the input you will see a grid with random pixels on it (mostly blue pixels).
# To make the output, you should find the largest rectangular area (of height/width >= 2, i.e. not a line) of black cells and turn it into pink.

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Find the largest all-black rectangle of area at least 2 by 2 and fill it pink.

    Input structure:
    - The grid contains non-black cells surrounding or interrupting black empty space.
    - Candidate empty regions are axis-aligned rectangles with both dimensions at least 2.
    - At least one all-black candidate rectangle is present.

    Rule:
    - Enumerate every axis-aligned rectangle with height and width at least 2.
    - Keep only rectangles whose cells are all black.
    - Fill the maximum-area remaining rectangle with pink.

    Key insight:
    - The target is the maximal empty rectangle by area, not necessarily a connected
      component of black cells.
    """
    # Plan:
    # 1. Enumerate all rectangular regions
    # 2. For each region, filter it out if it isn't all black
    # 3. Find the biggest region remaining by area
    # 4. Turn the biggest region into pink

    # 1. Enumerate all rectangular regions
    regions = { (r, c, w, h) for r in range(len(input_grid)) for c in range(len(input_grid[0])) for w in range(2, len(input_grid) - r + 1) for h in range(2, len(input_grid[0]) - c + 1) }

    # 2. For each region, filter it out if it isn't all black
    regions = { (r, c, w, h) for r, c, w, h in regions if np.all(input_grid[r:r+w, c:c+h] == Color.BLACK) }

    # 3. Find the biggest region remaining by area
    largest_region = max(regions, key=lambda region: region[2] * region[3])
    r, c, w, h = largest_region

    # 4. Turn the biggest region into pink
    output_grid = np.copy(input_grid)
    output_grid[r:r+w, c:c+h] = Color.PINK
    
    return output_grid

def generate_input() -> np.ndarray:
    # Mostly blue background, with a single large black rectangle.
    H, W = np.random.randint(8, 14), np.random.randint(8, 14)
    grid = np.full((H, W), Color.BLUE, dtype=int)

    rect_h = int(np.random.randint(2, max(3, H // 2)))
    rect_w = int(np.random.randint(2, max(3, W // 2)))
    rectangle = np.full((rect_h, rect_w), Color.BLACK, dtype=int)

    r0 = int(np.random.randint(0, H - rect_h + 1))
    c0 = int(np.random.randint(0, W - rect_w + 1))
    paste(grid, rectangle, r0, c0, clip=False)

    return grid
# ============= remove below this point for prompting =============
