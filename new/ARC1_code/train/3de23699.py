import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, paste, random_sprite


# concepts:
# pattern extraction, color matching

# description:
# In the input you will see a grid with a central pattern with four differently-colored pixels at the corners.
# To make the output, you should extract the central pattern (removing the differently-colored corners), 
# and change the color of the central pattern to match the corner pixels.

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Extract the framed central pattern, recolor it to the corner-marker color, and remove the frame.

    Input structure:
    - A single non-black cluster sits on a black background.
    - Four matching corner pixels mark a one-cell frame around the central pattern.
    - The central pattern uses a different non-black color from the corner markers.

    Rule:
    - Crop to the bounding box of all non-black pixels.
    - Use the corner color to recolor every non-black pixel in the crop.
    - Remove the outer one-cell frame and return only the recolored interior pattern.

    Key insight:
    - The corner pixels encode both the extraction boundary and the final color for the
      interior shape.
    """
    # Extract the central pattern by the four corner squares
    output_grid = np.copy(input_grid)

    # Crop the pattern out
    b = bbox(output_grid != Color.BLACK)
    assert b is not None
    output_grid = crop_bbox(output_grid, b)

    # Get the color of the corner squares
    corner_color = output_grid[0, 0]

    # Change the color of the central pattern to match the corner squares
    output_grid[output_grid != Color.BLACK] = corner_color

    # Remove the one pixel border around the central pattern
    output_grid = output_grid[1:-1, 1:-1]
    return output_grid

def generate_input() -> np.ndarray:
    # Random generate the size of the pattern 
    n, m = random.randint(3, 10), random.randint(3, 10)

    # Get the color of the pattern and the corner squares
    available_colors = random.sample(Color.NOT_BLACK, 2)
    pattern_color, corner_color = available_colors[0], available_colors[1]
    
    # Generate a random pattern
    pattern = random_sprite(m, n, color_palette=[pattern_color], density=0.4)

    # Ensure there is space for the four corner squares
    enlarged_pattern = np.zeros((m + 2, n + 2), dtype=int)

    # Set four pixels in each corner to the corner color
    enlarged_pattern[0, 0] = corner_color
    enlarged_pattern[0, n + 1] = corner_color
    enlarged_pattern[m + 1, 0] = corner_color
    enlarged_pattern[m + 1, n + 1] = corner_color

    # Place the pattern in the center of the corner pixels
    enlarged_pattern = paste(enlarged_pattern, pattern, 1, 1, transparent=Color.BLACK, clip=True)
    
    # Create the grid and place the pattern in it
    grid_width, grid_height = random.randint(m + 2, 2 * m + 2), random.randint(n + 2, 2 * n + 2)
    grid = np.zeros((grid_width, grid_height), dtype=int)

    # Randomly place the pattern in the grid
    pos_r, pos_c = random.randint(0, grid_width - (m + 2)), random.randint(0, grid_height - (n + 2))
    grid = paste(grid, enlarged_pattern, pos_r, pos_c, transparent=Color.BLACK, clip=True)
    
    return grid
# ============= remove below this point for prompting =============
