import numpy as np
import random
from arc_utils import Color, bbox, bbox_anchor, crop_bbox, draw_line, find_components, get_bg, paste, random_sprite


# concepts:
# rotate, position

# description:
# In the input you will see a grid with 3 regions separated by grey horizontal lines. The leftmost region contains a multicolored sprite and the others are empty (black)
# To make the output, rotate the leftmost region 90 degree clockwise and place it in the first empty region, then rotate it a further 90 degrees and put it in the second empty region, etc.

def main(input_grid):
    """
    Rotate the filled template region into each following empty region.

    Input structure:
    - Gray divider lines split the grid into same-sized regions.
    - Exactly one region contains a non-black, non-gray template.
    - Later regions are black slots for rotated copies.

    Rule:
    - Locate full gray row or column dividers and derive the region slices.
    - Find the first region containing template pixels.
    - Fill each subsequent region with the next 90-degree clockwise rotation.

    Key insight:
    - The dividers define ordered slots; the template supplies the repeated rotation sequence.
    """
    divider = Color.GREY
    bg = Color.BLACK
    H, W = input_grid.shape

    # Identify divider lines (either full columns or full rows of divider color).
    col_divs = [c for c in range(W) if np.all(input_grid[:, c] == divider)]
    row_divs = [r for r in range(H) if np.all(input_grid[r, :] == divider)]

    output_grid = input_grid.copy()

    if col_divs and not row_divs:
        cuts = [-1] + sorted(col_divs) + [W]
        regions = []
        for i in range(len(cuts) - 1):
            c0 = cuts[i] + 1
            c1 = cuts[i + 1] - 1
            if c0 <= c1:
                regions.append((0, c0, H, c1 + 1))
    else:
        cuts = [-1] + sorted(row_divs) + [H]
        regions = []
        for i in range(len(cuts) - 1):
            r0 = cuts[i] + 1
            r1 = cuts[i + 1] - 1
            if r0 <= r1:
                regions.append((r0, 0, r1 + 1, W))

    # Find the template region: the one containing non-black, non-divider pixels.
    template_idx = None
    template = None
    for idx, (r0, c0, r1, c1) in enumerate(regions):
        sub = input_grid[r0:r1, c0:c1]
        if np.any((sub != bg) & (sub != divider)):
            template_idx = idx
            template = sub.copy()
            break
    assert template is not None and template_idx is not None

    # Fill each subsequent region with successive 90° clockwise rotations.
    rot = template
    for (r0, c0, r1, c1) in regions[template_idx + 1 :]:
        rot = np.rot90(rot, k=-1)
        output_grid[r0:r1, c0:c1] = rot
    return output_grid

def generate_input():
    # Create a grid with some regions separated by vertical grey lines
    # This is generalized so that there can be various numbers of regions
    n_regions = random.choice([2, 3, 4, 5])
    region_size = np.random.randint(3, 6)
    n_dividers = n_regions - 1
    width, height = n_regions * region_size + n_dividers, region_size
    grid = np.full((width, height), Color.BLACK)

    # Draw lines for the dividers
    line_color = Color.GRAY
    for r in range(region_size, width, region_size + 1):
        draw_line(grid, r=r, c=0, direction=(0, 1), color=line_color)

    # Draw one template
    possible_colors = [color for color in Color.NOT_BLACK if color != line_color]
    template_sprite = random_sprite(region_size, region_size, color_palette=possible_colors, density=1.0)

    # Place the template pattern in the first black region
    r, c = 0, 0
    paste(grid, template_sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
