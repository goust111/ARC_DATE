
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_anchor, colors_present, draw_line, find_components

# concepts:
# downscaling

# description:
# In the input you will see horizontal and vertical bars/dividers of a particular color that define rectangular regions, with some of the single-pixel vertices colored differently.
# Some rectangular regions are have same color on the four vertices, and some are not.
# To make the output, find the regions colored differently on all vertices and produce a single output pixel of that color in the corresponding part of the output.
# Ignore regions which just have the color of the horizontal and vertical bars at their vertices.

def main(input_grid):
    """
    Convert marked divider-grid cells into pixels colored by matching corner markers.

    Input structure:
    - The input has divider bars of one color forming rectangular regions.
    - Some region vertices are single pixels in colors different from the divider.
    - A valid marked region has all four diagonal corner markers in the same color.

    Rule:
    - Identify the divider color and crop to the area spanned by special vertex pixels.
    - Find black regions bounded by the divider color.
    - Emit the shared corner color for regions with four matching non-divider markers, otherwise black.

    Key insight:
    - Colored vertices label regions, while the divider color only defines the grid structure.
    """
    # Plan:
    # 1. Parse the input into  dividers, regions, and vertices
    # 2. Extract the regions colored differently from the divider on all vertices
    # 3. Produce the output grid by representing each region with a single pixel of the color of its vertices, as long as its color is not the divider

    # 1. Parse the input
    # Detect the objects
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    
    # The divider color is the most frequent non-background color, and the background is black
    divider_color = max(Color.NOT_BLACK, key=lambda color: np.sum(input_grid == color))

    # Detect the single pixels that form the vertices of the regions (non-divider, non-background).
    pixels: list[tuple[int, int]] = []
    for obj in objects:
        _b = bbox(obj != Color.BLACK)
        if _b is None:
            continue
        r0, c0, r1, c1 = _b
        if r0 == r1 and c0 == c1:
            col = int(input_grid[r0, c0])
            if col != int(Color.BLACK) and col != int(divider_color):
                pixels.append((r0, c0))
    r_positions = [rc[0] for rc in pixels]
    c_positions = [rc[1] for rc in pixels]

    # Ignore regions that are not part of those special pixels
    r_min, r_max = min(r_positions), max(r_positions)
    c_min, c_max = min(c_positions), max(c_positions)
    input_grid = input_grid[r_min:r_max+1, c_min:c_max+1]

    # Extract just the black regions delimited by the divider color
    regions = find_components(input_grid, bg=divider_color, connectivity=4, monochromatic=True, return_format="objects")
    regions = [region for region in regions if colors_present(region, exclude={ divider_color }) == [Color.BLACK]]

    # 2. Analyze vertices, which live on the diagonal corners of the regions, to find the color of the regions
    # Determine their colors by the colors of their vertices, so we are going to have to look at the corners
    def diagonal_corners(obj, background):
        _b = bbox(obj != background)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1
        return [(r-1, c-1), (r + w, c-1), (r-1, c + h), (r + w, c + h)]
    
    region_colors = []
    for region in regions:
        corners = [
            (rr, cc)
            for rr, cc in diagonal_corners(region, background=divider_color)
            if 0 <= rr < input_grid.shape[0] and 0 <= cc < input_grid.shape[1]
        ]
        if len(corners) != 4:
            region_colors.append(Color.BLACK)
            continue
        vertex_colors = {int(input_grid[rr, cc]) for rr, cc in corners}
        if len(vertex_colors) == 1 and vertex_colors != {divider_color}:
            region_colors.append(vertex_colors.pop())
        else:
            region_colors.append(Color.BLACK)

    # 3. Produce the output grid, representing each big region as a single pixel
    
    # Find the number distinct X/Y positions of the regions, which tells us the size of the output
    r_positions = sorted({bbox_anchor(bbox(region != divider_color), anchor="upper left")[0] for region in regions})
    c_positions = sorted({bbox_anchor(bbox(region != divider_color), anchor="upper left")[1] for region in regions})

    # Make the output
    output_grid = np.full((len(r_positions), len(c_positions)), Color.BLACK)

    for region, color in zip(regions, region_colors):
        _b = bbox(region != divider_color)
        assert _b is not None
        r, c = bbox_anchor(_b, anchor="upper left")
        output_grid[r_positions.index(r), c_positions.index(c)] = color
    
    return output_grid
            
def generate_input():
    # Grid of square regions separated by divider lines; some regions have same-colored vertices.
    square_size = int(np.random.randint(2, 4))
    square_num = int(np.random.randint(6, 30 // (square_size + 1)))

    H = square_size * square_num + square_num - 1
    W = H
    grid = np.full((H, W), Color.BLACK, dtype=int)

    divider_color = int(np.random.choice(Color.NOT_BLACK))
    other_colors = [int(c) for c in Color.NOT_BLACK if int(c) != divider_color]

    # Full divider rows/cols.
    for r in range(square_size, H, square_size + 1):
        grid[r, :] = divider_color
    for c in range(square_size, W, square_size + 1):
        grid[:, c] = divider_color

    # Choose interior regions (not on the border) and color all 4 vertices the same.
    starts = list(range(0, H, square_size + 1))
    candidates = [
        (r0, c0)
        for r0 in starts
        for c0 in starts
        if r0 > 0 and c0 > 0 and (r0 + square_size) < H and (c0 + square_size) < W
    ]
    random.shuffle(candidates)
    for (r0, c0) in candidates[:6]:
        r1 = r0 + square_size - 1
        c1 = c0 + square_size - 1
        corners = [(r0 - 1, c0 - 1), (r1 + 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c1 + 1)]
        col = random.choice(other_colors)
        for rr, cc in corners:
            grid[rr, cc] = col

    return grid
