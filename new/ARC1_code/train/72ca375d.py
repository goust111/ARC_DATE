
import numpy as np
from typing import *
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, paste, random_free_location, random_sprite
import random

# concepts:
# symmetry, sprites

# description:
# In the input you will see several objects. One object represents a symmetic sprite. All the other objects represent non-symmetric sprites.
# The goal is to find the symmetric sprite and return it.

def main(input_grid):
    """
    Return the one sprite among several objects that has a symmetry.

    Input structure:
    - The input is a black grid containing several monochromatic sprites.
    - One sprite is symmetric under a rotation, reflection, or diagonal symmetry.
    - The other sprites are non-symmetric distractors.

    Rule:
    - Detect and crop each object into its sprite.
    - Test each sprite for radial, vertical, horizontal, and diagonal symmetry.
    - Return the first sprite that satisfies one of those symmetry tests.

    Key insight:
    - Shape symmetry, not color or position, identifies the target sprite.
    """
    # find the objects in the input grid
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")

    # crop out the sprites from the objects
    sprites = []
    for obj in objects:
        b = bbox(obj != Color.BLACK)
        if b is None:
            continue
        sprites.append(crop_bbox(obj, b))

    # find the symmetric sprite
    symmetric_sprite = None
    for sprite in sprites:
      # check if the sprite is radially symmetric
      if np.array_equal(sprite, np.rot90(sprite, 1)):
        symmetric_sprite = sprite
        break
      # check if the sprite is vertically symmetric
      elif np.array_equal(sprite, np.fliplr(sprite)):
        symmetric_sprite = sprite
        break
      # check if the sprite is horizontally symmetric
      elif np.array_equal(sprite, np.flipud(sprite)):
        symmetric_sprite = sprite
        break
      # check if the sprite is diagonally symmetric
      elif np.array_equal(sprite, sprite.T) or (
          sprite.shape[0] == sprite.shape[1] and np.array_equal(sprite, np.rot90(sprite, 2).T)
      ):
        symmetric_sprite = sprite
        break

    return symmetric_sprite

  

def generate_input():
    # make a black 10x10 grid as background
    grid = np.zeros((10, 10), dtype=int)

    # Add a symmetric sprite.
    color = np.random.choice(Color.NOT_BLACK)
    side_length = np.random.randint(2, 8)
    width = np.random.choice([2, side_length])
    height = side_length if width == 2 else 2
    symmetry = np.random.choice(["vertical", "horizontal", "diagonal", "radial"])
    if symmetry in ["radial", "diagonal"]:
        width = height = np.random.randint(2, 5)

    symmetric_sprite = random_sprite(width, height, symmetry=symmetry, color_palette=[color], connectivity=8)
    r, c = random_free_location(grid, symmetric_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
    paste(grid, symmetric_sprite, r, c, transparent=Color.BLACK, clip=True)

    # Add some non-symmetric sprites (skip if there is no room).
    for _ in range(np.random.randint(3, 6)):
        color = np.random.choice(Color.NOT_BLACK)
        side_length = np.random.randint(3, 8)
        width = np.random.choice([2, side_length])
        height = side_length if width == 2 else 2
        non_symmetric_sprite = random_sprite(width, height, symmetry="not_symmetric", color_palette=[color], connectivity=8)
        try:
            r, c = random_free_location(grid, non_symmetric_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(grid, non_symmetric_sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
