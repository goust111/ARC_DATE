
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components

# concepts:
# rectangular cells, color guide

# description:
# In the input you will see grey horizontal and vertical bars that divide the grid into nine 3x3 rectangular regions, each of which contains 4-5 colored pixels
# To make the output, find the region that has exactly 4 colored pixels, and use its colors as a guide to fill in all the other cells

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely sparse 3x3 cell as a color key for all cells in the divided grid.

    Input structure:
    - The input is a 3-by-3 arrangement of cells separated by one divider color.
    - Each cell contains black plus several colored pixels.
    - Exactly one cell has four non-black, non-divider pixels; the others have five.

    Rule:
    - Split the grid into its divider-separated cells.
    - Find the cell with exactly four colored pixels and crop it as the guide sprite.
    - Fill every cell uniformly with the guide sprite color at the corresponding cell position.

    Key insight:
    - The special sparse cell is a miniature 3-by-3 lookup table: its position-indexed colors
      determine the color assigned to every region.
    """

    # First identify 

    # Trick for decomposing inputs divided into rectangular regions by horizontal/vertical bars:
    # Treat the bar color as the background, and break the input up into connected components with that background color

    # The divider color is the color of the horizontal and vertical bars
    divider_colors = [ input_grid[r,c] for r in range(input_grid.shape[0]) for c in range(input_grid.shape[1])
                     if np.all(input_grid[r,:] == input_grid[r,0]) or np.all(input_grid[:,c] == input_grid[0,c]) ]
    assert len(set(divider_colors)) == 1, "There should be exactly one divider color"
    divider_color = divider_colors[0] # background=divider_color

    # Find multicolored regions, which are divided by divider_color, so we treat that as background, because it separates objects
    # Within each region there can be multiple colors
    regions = find_components(input_grid, bg=divider_color, connectivity=4, monochromatic=False, return_format="objects")
    # Tag the regions with their location within the 2D grid of (divided) regions
    # First get the bounding-box locations...
    locations = []
    for region in regions:
        _b = bbox(region != divider_color)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1
        locations.append((r, c, region))
    # ...then re-index them so that (x, y) is the coordinate within the grid of rectangular regions
    grid_of_regions = []
    for r, c, region in locations:
        num_left_of_region = len({other_r for other_r, other_c, other_region in locations if other_r < r})
        num_above_region = len({other_c for other_r, other_c, other_region in locations if other_c < c})
        grid_of_regions.append((num_left_of_region, num_above_region, region))

    # Find the region with exactly 4 colors
    special_region = None
    for region in regions:
        not_divider_and_not_black = (region != divider_color) & (region != Color.BLACK)
        if np.sum(not_divider_and_not_black) == 4:
            assert special_region is None, "More than one special region found"
            special_region = region
    
    # Convert to a sprite
    _b = bbox(special_region != divider_color)
    assert _b is not None
    special_sprite = crop_bbox(special_region, _b)
    
    # Create the output grid
    output_grid = np.zeros_like(input_grid)

    # Put the dividers back in
    output_grid[input_grid == divider_color] = divider_color

    # Fill in the cells with the special colors
    for r, c, region in grid_of_regions:
        output_grid[region != divider_color] = special_sprite[r, c]

    return output_grid



def generate_input() -> np.ndarray:
    
    divider_color = Color.GRAY

    # make the dividers, which comprise horizontal/vertical bars creating 3x3 cells, with 3 cells in each direction
    cell_size = 3
    n_cells = 3
    divider_size = 1 # the divider is a single pixel
    n_dividers = n_cells - 1
    distance_between_cells = cell_size + divider_size
    m = cell_size*n_cells + divider_size*n_dividers
    grid = np.full((m, m), Color.BLACK)
    for i in range(n_dividers):
        # horizontal dividers
        grid[cell_size + i*(cell_size + divider_size), :] = divider_color
        # vertical dividers
        grid[:, cell_size + i*(cell_size + divider_size)] = divider_color
    
    # pick one of the cells to have exactly 4 colors (the others will have 5)
    special_cell_r, special_cell_c = np.random.randint(3), np.random.randint(3)

    for r in range(3):
        for c in range(3):
            if r == special_cell_r and c == special_cell_c:
                n_colors = 4
            else:
                n_colors = 5

            # extract view of the cell
            # each of the cells is 3x3, but there is a divider in between them, so they are actually 4x4 apart
            cell = grid[r*distance_between_cells : r*distance_between_cells + cell_size,
                        c*distance_between_cells : c*distance_between_cells + cell_size]

            # color the cell by picking random positions and random colors until we have enough colored pixels
            while np.sum(cell!=Color.BLACK) < n_colors:
                # pick a random spot to color
                cell_r, cell_c = np.random.randint(cell_size), np.random.randint(cell_size)
                cell[cell_r, cell_c] = random.choice([color for color in Color.ALL_COLORS if color != Color.BLACK and color != divider_color])

    return grid
