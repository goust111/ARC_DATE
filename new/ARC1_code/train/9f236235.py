
import numpy as np
from typing import *
from arc_utils import Color, bbox, colors_present, draw_line, find_components, scale_sprite
import random


# concepts:
# downscaling, mirror, horizontal/vertical bars

# description:
# In the input you will see horizontal and vertical bars separating different regions/cells/partitions with each cell containing different colors, like a chessboard.
# Each separated region has a single color.
# To make the output, make a grid with one colored pixel for each region of the chessboard.
# Finally mirror along the x-axis.

def main(input_grid):
    """
    Collapse a separator grid into one color per region, then mirror the compact grid left-to-right.

    Input structure:
    - The input is divided into uniformly colored cells by horizontal and vertical separator bars.
    - Each region between separators has a single color.
    - The separator color forms a connected partition spanning the full canvas.

    Rule:
    - Identify the separator color and extract all separator-delimited regions.
    - Replace each region with one pixel of its region color, preserving row and column order.
    - Mirror the compact region-color grid horizontally.

    Key insight:
    - The large chessboard-like image is an enlarged region map; the final step is a left-right
      reflection of that compressed map.
    """
    # Plan:
    # 1. Determine the color of the separator between the regions
    # 2. Extract all the regions separated by the separator color
    # 3. Find the region positions and their possible X/Y positions
    # 4. Create the output grid so that one pixel represents the original region, preserving X/Y ordering
    # 5. Mirror the output grid by x-axis

    H, W = input_grid.shape

    # 1. Determine the separator color.
    # Mirror the legacy seed approach: find the monochromatic component (non-black)
    # whose bbox spans the whole canvas.
    bg = Color.BLACK
    sep_objs = find_components(input_grid, bg=bg, connectivity=4, monochromatic=True, return_format="objects")
    sep_candidates = [obj for obj in sep_objs if bbox(obj != bg) == (0, 0, H - 1, W - 1)]
    assert len(sep_candidates) == 1, "There should be exactly 1 separator partitioning the input"
    separator_color = colors_present(sep_candidates[0], exclude={bg})[0]

    # 2. Extract all regions separated by the separator color.
    regions = find_components(
        input_grid,
        bg=int(separator_color),
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    # 3/4. Build the downscaled grid (one pixel per region), preserving row/col ordering.
    r_positions = sorted({int(reg["bbox"][0]) for reg in regions})
    c_positions = sorted({int(reg["bbox"][1]) for reg in regions})
    r_to_i = {r: i for i, r in enumerate(r_positions)}
    c_to_j = {c: j for j, c in enumerate(c_positions)}

    output_grid = np.full((len(r_positions), len(c_positions)), Color.BLACK, dtype=int)
    for reg in regions:
        r0, c0, _r1, _c1 = reg["bbox"]
        output_grid[r_to_i[int(r0)], c_to_j[int(c0)]] = int(reg["color"])

    # 5. Mirror the downscaled grid horizontally (left-right).
    output_grid = np.fliplr(output_grid)

    return output_grid

def generate_input():
    # Randomly choose the number of regions that are going to form the output canvas
    w, h = np.random.randint(3, 6), np.random.randint(3, 6)
    # Keep track of the color of each region
    region_colors = np.full((w, h), Color.BLACK)

    # Randomly choose the colors. the separator gets a special color and colored regions get a color randomly selected from other_colors
    num_other_colors = np.random.randint(1, 4)
    separator_color, *other_colors = np.random.choice(Color.NOT_BLACK, num_other_colors + 1, replace=False)

    # Randomly color the regions
    for r, c in np.ndindex(w, h):
        # Randomly determine if the cell should be colored
        if np.random.choice([True, False]):
            # Randomly choose the color
            region_colors[r, c] = np.random.choice(other_colors)
    
    # Randomly choose the scale factor and scale the regions so that they are bigger than just a single pixel
    scale_factor = np.random.randint(3, 6)
    grid = scale_sprite(region_colors, scale_factor)
    
    # Draw horizontal/vertical lines to separate the colors
    interval = scale_factor
    # First draw vertical lines
    for r in range(interval, w * scale_factor, interval):
        draw_line(grid, r=r, c=0, direction=(0, 1), color=separator_color)
    # Then draw horizontal lines
    for c in range(interval, h * scale_factor, interval):
        draw_line(grid, r=0, c=c, direction=(1, 0), color=separator_color)
    
    # drop the extra pixels
    grid = grid[1:,1:]

    return grid
