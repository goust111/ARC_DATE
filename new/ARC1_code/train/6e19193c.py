
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, paste

# concepts:
# direction, lines, pointing

# description:
# In the input, you will see several objects of the same color that are in an arrowhead shape and facing different directions.
# The goal is to find the directions of the arrowheads and draw lines that would represent the path they had been moving in to go in that direction.

def main(input_grid):
    """
    Extend each small arrowhead with a line in the direction it points.

    Input structure:
    - The input is a black grid with several same-colored arrowhead objects.
    - Each arrowhead is a small shape whose missing corner indicates its direction.
    - Lines should use the arrowhead color.

    Rule:
    - Crop each arrowhead and compare the mean positions of colored and black cells.
    - Use that offset to infer the pointing direction.
    - Draw a line from the matching outer corner of the arrowhead's bounding box to the grid edge.

    Key insight:
    - The arrowhead asymmetry identifies both the direction of travel and the corner where the path begins.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the objects in the input grid
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")

    # for each object, find the direction the arrowhead is pointing in by finding the relative mean position of colored and black pixels in the bounding box of the object
    for obj in objects:
        # find the bounding box of the object
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        r, c, _r1, _c1 = _b
        w = _r1 - r + 1
        h = _c1 - c + 1

        # crop the object to extract the sprite
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        sprite = crop_bbox(obj, _b)

        # find the color of the object
        color = np.unique(obj[obj != Color.BLACK])[0]

        # find the mean position of the colored pixels
        mean_pos = np.mean(np.argwhere(sprite != Color.BLACK), axis=0)

        # find the mean position of all the black pixels
        mean_black_pos = np.mean(np.argwhere(sprite == Color.BLACK), axis=0)

        # find the direction the arrowhead is pointing in, it is from the mean position of the colored pixels to the mean position of the black pixels
        direction = np.sign(mean_black_pos - mean_pos).astype(int)

        # draw a line in the direction the arrowhead is pointing in from the corresponding corner of the bounding box
        # list the corners of the bounding box
        corners = [(r - 1, c - 1), (r + w, c - 1), (r - 1, c + h), (r + w, c + h)]
        # compute the center of the object
        center = (r + w / 2, c + h / 2)
        # if the direction of the corner from the center of the object matches the direction we want to draw a line in, then draw a line
        for corner in corners:
            # check if the corner is in the direction that the arrowhead is pointing
            vector_to_corner = np.array(corner) - np.array(center)
            if np.all(np.sign(vector_to_corner) == direction):
                draw_line(output_grid, corner[0], corner[1], length=None, color=color, direction=direction)

    return output_grid

def generate_input():
    # make a 10x10 grid as background
    grid = np.zeros((10, 10), dtype=int)

    # choose the color of the arrowheads
    color = np.random.choice(Color.NOT_BLACK)

    # make the arrowheads by making a 2x2 square then removing one of the corners randomly and placing it randomly on the grid
    for i in range(4):
        # make at arrowhead
        arrowhead = np.full((2, 2), color)
        corner = np.random.randint(4)
        arrowhead[corner // 2, corner % 2] = Color.BLACK
        
        # place it in a random free location with 1 cell padding
        _loc = first_free_location(grid, arrowhead, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        paste(grid, arrowhead, r, c, transparent=Color.BLACK, clip=True)
 
    return grid



    
