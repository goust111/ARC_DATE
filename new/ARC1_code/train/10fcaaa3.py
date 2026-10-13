
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, first_free_location, paste

# concepts:
# Coloring diagonal pixels, repetition

# description:
# Given an input grid of arbitrary size, with some small number of colored pixels on it.
# To produce the output, replicate the input grid 4 times, 2 on the top and 2 on the bottom. 
# Color all the diagonal pixels adjacent to a colored pixel teal if the diagonal pixels are black. 

def main(input_grid):
   """
   Tile the input into a 2x2 canvas and mark black diagonal neighbors of colored pixels teal.

   Input structure:
   - The input is a small black grid containing one to four non-teal colored pixels.
   - Pixel colors are preserved when the input is tiled.
   - Teal is reserved for newly added diagonal marks.

   Rule:
   - Copy the input into all four quadrants of a doubled output grid.
   - For every non-black, non-teal pixel in the doubled grid, inspect its four diagonal neighbors.
   - Color any black diagonal neighbor teal.

   Key insight:
   - The output combines replication with local decoration: after tiling, every colored seed grows
     only into currently black diagonal cells.
   """
   # Replicate input grid 4 times to initialize output grid
   output_grid = np.zeros((2*input_grid.shape[0], 2* input_grid.shape[1]),dtype=int)
   for i in range(2):
      for j in range(2):
         paste(output_grid, input_grid, i*input_grid.shape[0], j*input_grid.shape[1], transparent=Color.BLACK, clip=True)
  
   # Create diagonal directions
   diagonal_dx_dy = [(1,1),(-1,1),(1,-1),(-1,-1)]

   # Color diagonal pixels 
   for c in range(output_grid.shape[1]):
      for r in range(output_grid.shape[0]):
         if output_grid[r,c] != Color.BLACK and output_grid[r,c] != Color.TEAL:
            for dr,dc in diagonal_dx_dy:
               # Color diagonal pixel teal if it is black
               if r+dr >= 0 and r+dr < output_grid.shape[0] and c+dc >= 0 and c+dc < output_grid.shape[1] and output_grid[r+dr,c+dc] == Color.BLACK:
                  output_grid[r+dr,c+dc] = Color.TEAL
  
   return output_grid

def generate_input():
    # Have 1 to 4 number of colored pixels in the initial square
    n_colored_pixels = random.randint(1,4)
    
    # Random pixel color that is not black or teal. 
    pixel_color = random.choice(list(Color.NOT_BLACK))
    while pixel_color == Color.TEAL:
        pixel_color = random.choice(list(Color.NOT_BLACK))

    # Random size of input grid
    n,m= random.randint(2,10), random.randint(2,10)

    # Initialize grid
    grid = np.zeros((m,n),dtype=int)
  
    # Create a dummy sprite with one pixel. 
    sprite = np.array([pixel_color]).reshape(1,1)
    
    # Randomly place n_colored_pixels pixels on the grid
    for i in range(n_colored_pixels):
      _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
      if _loc is None:
          raise ValueError("no placement")
      r, c = _loc
      paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
