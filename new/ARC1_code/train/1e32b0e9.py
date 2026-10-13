import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, paste, random_sprite

# concepts:
# pattern reconstruction

# description:
# In the input you will see 9 squares seperated by 4 lines. The top-left square contains the original pattern.
# Each square contains either a small portion of pattern or remains empty.
# To make the output, you should detect the pattern on the top-left square and fill each square 

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Complete each divided square by comparing it with the top-left template pattern.

    Input structure:
    - A 3 by 3 layout of equal squares is separated by full divider lines of one color.
    - The top-left square contains the complete pattern.
    - Other squares contain black omissions from that same pattern.

    Rule:
    - Detect the divider color and split the grid into square regions.
    - Use the top-left region as the template.
    - In every other region, fill template cells that are missing or different with the divider color.

    Key insight:
    - The first region supplies the reference mask, and missing pattern cells are marked with
      the separator color instead of recreating the original pattern color.
    """
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)  

    # Detect the color of the lines
    for r, row in enumerate(output_grid):
        # Find the line
        all_equal = np.unique(row).size == 1
        if all_equal:
            line_color = row[0]
            break
    
    # Get all the squares seperated by lines in the grid
    squares = find_components(output_grid, bg=line_color, connectivity=4, monochromatic=False, return_format="objects")

    # Get all squares' bounding box and cropped pattern
    cropped_squares  = []
    for obj in squares:
        _b = bbox(obj != line_color)
        assert _b is not None
        r, c, _r1, _c1 = _b
        width = _r1 - r + 1
        height = _c1 - c + 1
        _b = bbox(obj != line_color)
        assert _b is not None
        square = crop_bbox(obj, _b)
        cropped_squares.append({'x': r, 'y': c, 'len': width, 'pattern': square})

    # Sort the squares by their position
    cropped_squares = sorted(cropped_squares, key=lambda r: (r['x'], r['y']))

    # The top-left square contains the original pattern
    template_pattern = cropped_squares[0]['pattern']
    other_patterns = cropped_squares[1:]

    # Fill the missing pattern compared to template square with line color
    for square in other_patterns:
        r, c = square['x'], square['y']
        square_pattern = square['pattern']

        # Fill the missing pattern compared to template square with line color
        for i, j in np.argwhere(template_pattern != Color.BLACK):
            if template_pattern[i, j] != square_pattern[i, j]:
                square_pattern[i, j] = line_color

        # Place the reconstructed pattern on the output grid
        output_grid = paste(output_grid, square_pattern, r, c, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input() -> np.ndarray:
    # Define the base cofiguration of the grid seperated by chessboard lines
    # Randomly select the size of the squares, create a 3x3 grid of squares
    square_len = np.random.choice([5, 7, 9])
    pattern_len = square_len - 2
    square_num = 3

    # Size of the grid is grid length plus line length
    n, m = square_len * square_num + square_num - 1, square_len * square_num + square_num - 1
    grid = np.zeros((m, n), dtype=int)

    # Select two colors for the lines and the pattern
    colors = Color.NOT_BLACK.copy()
    line_color = random.choice(colors)  
    colors.remove(line_color)
    pattern_color = random.choice(colors) 

    # Full grid dividers.
    for r in range(square_len, m, square_len + 1):
        grid[r, :] = int(line_color)
    for c in range(square_len, n, square_len + 1):
        grid[:, c] = int(line_color)

    # Create the pattern in the top-left square with the pattern color
    template_sprite = random_sprite(m=pattern_len, n=pattern_len, color_palette=[pattern_color], connectivity=8, density=0.5)

    # Fill the top-left square with the original pattern
    # Fill the other regions with corrupted versions that have pixels randomly set to black
    for i in range(0, m, square_len + 1):
        for j in range(0, n, square_len + 1):
            # are we the top-left region?
            if i == 0 and j == 0:
                paste(grid, template_sprite, i + 1, j + 1, transparent=Color.BLACK, clip=True)
                continue

            # otherwise: Create a corrupted version of the pattern; randomly set some pixels to black
            corrupted_sprite = template_sprite.copy()
            for r, c in np.argwhere(corrupted_sprite != Color.BLACK):
                if random.choice([0, 1]) == 0:
                    corrupted_sprite[r, c] = Color.BLACK
            # Place the sprite on the grid, remain a black border around the sprite
            paste(grid, corrupted_sprite, i + 1, j + 1, transparent=Color.BLACK, clip=True)
    
    return grid
