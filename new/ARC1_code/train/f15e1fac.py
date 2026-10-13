
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line

# concepts:
# magnetism, direction, lines

# description:
# In the input, you will see a black grid with teal pixels scattered along one edge and red pixels scattered along an edge perpendicular to the teal one.
# To make the output, make the teal pixels flow from the edge they are on to the opposite edge. Whenever there is a red pixel in the same column or row as the flow of teal pixels, push the teal pixel's flow one pixel away from the red pixel.

def main(input_grid):
    """
    Flow teal pixels across the grid while red edge pixels push the paths aside.

    Input structure:
    - The black grid has teal pixels on one edge and red pixels on a perpendicular edge.
    - Teal starts flow toward the opposite edge.
    - Red edge pixels create row or column push effects.

    Rule:
    - Determine the teal flow direction and red push direction from the occupied edges.
    - Start a path from each teal pixel.
    - Move along the flow direction, but shift one cell away whenever aligned with any red pixel.

    Key insight:
    - Red pixels act like directional repellers: they do not move, but their rows or
      columns offset the teal streams.
    """
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # figure out which edges the red and teal pixels are on and decide the direction of flow and push based on that
    top = output_grid[:,0]
    bottom = output_grid[:, -1]
    left = output_grid[0, :]
    if Color.RED in top:
        # push the teal pixels down
        push = (0, 1)

        # teal can only be on the left or right edge if red is on the top edge
        if Color.TEAL in left:
            flow = (1, 0) # flow right
        else:
            flow = (-1, 0) # flow left    
    elif Color.RED in bottom:
        # push the teal pixels up
        push = (0, -1)

        # teal can only be on the left or right edge if red is on the bottom edge
        if Color.TEAL in left:
            flow = (1, 0) # flow right
        else:
            flow = (-1, 0) # flow left
    elif Color.RED in left:
        # push the teal pixels to the right
        push = (1, 0)

        # teal can only be on the top or bottom edge if red is on the left edge
        if Color.TEAL in top:
            flow = (0, 1) # flow down
        else:
            flow = (0, -1) # flow up
    else: # red is on the right edge
        # push the teal pixels to the left
        push = (-1, 0)
        
        # teal can only be on the top or bottom edge if red is on the right edge
        if Color.TEAL in top:
            flow = (0, 1) # flow down
        else:
            flow = (0, -1) # flow up

    # find the coordinates of the teal and red pixels
    teal = np.where(input_grid == Color.TEAL)
    red = np.where(input_grid == Color.RED)

    # draw the flow of teal pixels
    for i in range(len(teal[0])):
        # start at a teal pixel
        r, c = teal[0][i], teal[1][i]

        # draw across the grid one pixel at a time adjusting for red pixel effects
        while r >= 0 and r < output_grid.shape[0] and c >= 0 and c < output_grid.shape[1]:
            # push the teal pixel away from the red pixel if it is in the same row or column
            if r in red[0] or c in red[1]:
                r += push[0]
                c += push[1]

                # stop this flow if it goes off the grid
                if r < 0 or r >= output_grid.shape[0] or c < 0 or c >= output_grid.shape[1]:
                    break
                
            # draw a teal pixel in the flow
            output_grid[r, c] = Color.TEAL

            # move the flow one pixel in the direction of flow
            r += flow[0]
            c += flow[1]

    return output_grid


def generate_input():
    # make a black grid as the background
    n = np.random.randint(10, 20)
    m = np.random.randint(10, 20)
    grid = np.zeros((n, m), dtype=int)

    # select which edges will be teal and which will be red
    top_or_bottom_color = np.random.choice([Color.TEAL, Color.RED])
    left_or_right_color = Color.TEAL if top_or_bottom_color == Color.RED else Color.RED

    # make a random vector of length m for the top or bottom edge of the grid
    top_or_bottom = np.zeros(m, dtype=int)

    # scatter the selected color anywhere along the vector except the ends
    top_or_bottom[np.random.choice(range(1, m-1), np.random.randint(2, 5), replace=False)] = top_or_bottom_color
    
    # make a random vector of length n for the left or right edge of the grid
    left_or_right = np.zeros(n, dtype=int)

    # scatter the selected color anywhere along the vector except the ends
    left_or_right[np.random.choice(range(1, n-1), np.random.randint(2, 5), replace=False)] = left_or_right_color

    # randomly put the top_or_bottom vector on the top or bottom of the grid
    if np.random.rand() < 0.5:
        grid[0] = top_or_bottom
    else:
        grid[-1] = top_or_bottom
    
    # randomly put the left_or_right vector on the left or right of the grid
    if np.random.rand() < 0.5:
        grid[:,0] = left_or_right
    else:
        grid[:,-1] = left_or_right
    
    return grid
