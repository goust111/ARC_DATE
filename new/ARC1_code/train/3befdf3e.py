import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste, random_sprite

# concepts:
# growing, color change

# description:
# In the input you will see square(s) with a single pixel border around them in a different color.
# To make the output, swap the color between the square and the border, 
# and then put rectangles on the edges of the border whose width is the same as the length of the inner square, 
# and whose color is the same as the border in the input image.

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Swap each bordered square's colors and grow original-border-colored rectangles from its four sides.

    Input structure:
    - The grid contains one or two separated square patterns on a black background.
    - Each pattern has a solid inner square and a one-pixel outer border in a different color.
    - There is enough surrounding black space for the pattern to grow outward.

    Rule:
    - Split each pattern into its inner and outer square components.
    - Swap the colors of the inner square and outer border.
    - Add four rectangles outside the outer square, using the original border color.

    Key insight:
    - The original border color becomes the growth material, while the original inner and
      outer square colors trade places in the central pattern.
    """
    # Get the output grid with the same size as the input grid.
    output_grid = np.copy(input_grid)

    # Detect the square pattern in the input grid.
    entire_square = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="objects")
    for each_square in entire_square:
        # Detect the inner and outer squares in the pattern.
        object_square = find_components(each_square, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

        # Split the square pattern into inner and outer squares.
        split_objects = []
        for obj in object_square:
            _b = bbox(obj != Color.BLACK)
            assert _b is not None
            r, c, _r1, _c1 = _b
            width = _r1 - r + 1
            height = _c1 - c + 1
            _b = bbox(obj != Color.BLACK)
            assert _b is not None
            shape = crop_bbox(obj, _b)
            color = obj[r, c]
            split_objects.append({'x': r, 'y': c, 'len': width, 'color': color, 'shape': shape})

        # Get the outer and inner square by comparing their size.
        if split_objects[0]['len'] > split_objects[1]['len']:
            outer_square = split_objects[0]
            inner_square = split_objects[1]
        else:
            outer_square = split_objects[1]
            inner_square = split_objects[0]
        
        # Swap the color between the inner and outer squares.
        outer_square_pattern, inner_square_pattern = outer_square['shape'], inner_square['shape']
        outer_square_pattern[outer_square_pattern == outer_square['color']] = inner_square['color']
        inner_square_pattern[inner_square_pattern == inner_square['color']] = outer_square['color']

        # Draw the inner and outer squares after swaping on the output grid.
        output_grid = paste(output_grid, outer_square_pattern, outer_square['x'], outer_square['y'], transparent=Color.BLACK, clip=True)
        output_grid = paste(output_grid, inner_square_pattern, inner_square['x'], inner_square['y'], transparent=Color.BLACK, clip=True)

        # Draw the rectangle as growing from outer square, the width is the same as the length of the inner square.
        # The rectangle is the same color as the original outer square.
        rectangle_width, rectangle_height = outer_square['len'], inner_square['len']  
        
        # Create the rectangle pattern for the edges of the outer square.
        rectangle_up_down = np.full((rectangle_width, rectangle_height), outer_square['color'])
        rectangle_left_right = np.full((rectangle_height, rectangle_width), outer_square['color'])

        # Draw the rectangle on the four edges of the outer square.
        output_grid = paste(output_grid, rectangle_up_down, outer_square['x'], outer_square['y'] - inner_square['len'], transparent=Color.BLACK, clip=True)
        output_grid = paste(output_grid, rectangle_up_down, outer_square['x'], outer_square['y'] + outer_square['len'], transparent=Color.BLACK, clip=True)
        output_grid = paste(output_grid, rectangle_left_right, outer_square['x'] - inner_square['len'], outer_square['y'], transparent=Color.BLACK, clip=True)
        output_grid = paste(output_grid, rectangle_left_right, outer_square['x'] + outer_square['len'], outer_square['y'], transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Generate the background grid with size of n x n.
    grid_len = random.randint(24, 30)
    n, m = grid_len, grid_len
    grid = np.zeros((m, n), dtype=int)

    # Randomly select the number of square pattern.
    num_square = random.randint(1, 2)

    # Ensure there is enough space for the square pattern to grow.
    max_square_len = grid_len // (num_square * 3)

    # Choose two colors for inner and outer square, and one color represents the boundary.
    # The boundary is used for leave enough space for the square pattern in output to grow.
    available_color = Color.NOT_BLACK.copy()
    two_colors = random.sample(available_color, k=2)
    available_color.remove(two_colors[0])
    available_color.remove(two_colors[1])
    boundary_color = available_color[0]

    for _ in range(num_square):
        # Randomly select the size of the square pattern.
        # The outer square is the inner square with a single pixel border around it.
        outer_square_len = random.randint(3, max_square_len)
        inner_square_len = outer_square_len - 2

        # The square will grow inner_square_len in four directions
        border_len = outer_square_len + inner_square_len * 2
        
        # Create the inner square, outer square, and border.
        outer_square = random_sprite(m=outer_square_len, n=outer_square_len, color_palette=[two_colors[1]], density=1.0)
        inner_square = random_sprite(m=inner_square_len, n=inner_square_len, color_palette=[two_colors[0]], density=1.0)
        border = random_sprite(m=border_len, n=border_len, color_palette=[boundary_color], density=1.0)
        
        # Get a random free location for the pattern
        _loc = first_free_location(grid, border, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        posx, posy = _loc

        # Draw the square pattern on the grid
        grid = paste(grid, border, posx, posy, transparent=Color.BLACK, clip=True)
        grid = paste(grid, outer_square, posx + inner_square_len, posy + inner_square_len, transparent=Color.BLACK, clip=True)
        grid = paste(grid, inner_square, posx + inner_square_len + 1, posy + inner_square_len + 1, transparent=Color.BLACK, clip=True)
    
    # Remove the boundary color from the grid
    grid = np.where(grid == boundary_color, Color.BLACK, grid)
    return grid
