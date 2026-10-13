
import numpy as np
from typing import *
import random
from arc_utils import Color, draw_line, find_components, paste, random_sprite, translate

# concepts:
# sprites, color change, collision detection, repetition, overlap

# description:
# In the input you will see a 3x3 object with a few other objects around it.
# For each of the other sprites around the central 3x3 object:
# 1. Slide the central sprite so it completely overlaps the other sprite (slide it as much as you can to do so)
# 2. Change the color of the central sprite to match the color of the other sprite
# 3. Repeat the slide (by the same displacement vector) indefinitely until it falls off the canvas

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Project the central sprite through each surrounding partial copy, recoloring each ray to match its guide.

    Input structure:
    - The input contains one largest central sprite and several smaller partial sprites around it.
    - Each surrounding sprite is a same-shape subset of the central sprite in its own color.
    - All objects sit on a black background.

    Rule:
    - Find the largest displacement that makes the central sprite cover each surrounding sprite.
    - Recolor the central sprite to the surrounding sprite's color.
    - Repeat that displacement from the center until the copies leave the canvas.

    Key insight:
    - Each partial outer object is a direction-and-color guide: aligning the central sprite to it
      reveals the vector that should be extended repeatedly.
    """
    # find the objects, which are monochromatic connected components
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=True, return_format="objects")

    # find the central object, which is the biggest
    central_object = max(objects, key=lambda obj: np.sum(obj != Color.BLACK))

    # find the other objects
    other_objects = [obj for obj in objects if not np.array_equal(obj, central_object)]

    output_grid = np.copy(input_grid)

    for other_object in other_objects:
        # find the biggest displacement vector that will make the central object completely overlap the other object
        biggest_displacement_vector = (0,0)
        displacement_vectors = [ (i, j) for i in range(-10, 10) for j in range(-10, 10) ]
        for displacement_vector in displacement_vectors:
            # translate the central object by the displacement vector
            translated_central_object = translate(central_object, displacement_vector[0], displacement_vector[1], background=Color.BLACK)

            # check if the translated object completely overlaps the other object
            translated_mask, other_mask = translated_central_object != Color.BLACK, other_object != Color.BLACK
            overlaps = np.all(translated_mask & other_mask == other_mask)

            if overlaps:
                # but is it the biggest?
                if biggest_displacement_vector[0] ** 2 + biggest_displacement_vector[1] ** 2 < displacement_vector[0] ** 2 + displacement_vector[1] ** 2:
                    biggest_displacement_vector = displacement_vector

        displacement_vector = biggest_displacement_vector

        # color change
        color_of_other_object = np.unique(other_object[other_object != Color.BLACK])[0]
        central_object[central_object != Color.BLACK] = color_of_other_object

        # repeat the displacement indefinitely until it falls off the canvas
        for i in range(1, 10):
            displaced_central_object = translate(central_object, displacement_vector[0] * i, displacement_vector[1] * i, background=Color.BLACK)
            paste(output_grid, displaced_central_object, 0, 0, transparent=Color.BLACK, clip=True)
            
    return output_grid



def generate_input() -> np.ndarray:
    # make a black grid first as background
    n, m = 21, 21
    grid = np.zeros((m, n), dtype=int)

    # make a 3x3 object
    central_color = np.random.choice(Color.NOT_BLACK)
    central_sprite = random_sprite(3, 3, color_palette=[central_color])

    # place the central object near the center
    r, c = np.random.randint(int(0.3*m), int(0.7*m)), np.random.randint(int(0.3*n), int(0.7*n))
    paste(grid, central_sprite, r, c, transparent=Color.BLACK, clip=True)

    # possible displacement vectors can range in any of the eight different directions (cardinal directions and in between them)
    # they should be close to just a little more than the length of the central object, however
    eight_cardinal_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    # pick a random subset of them, between 1-4
    displacement_vectors = random.sample(eight_cardinal_directions, k=np.random.randint(1, 5))

    for vector in displacement_vectors:
        vector_length = np.random.randint(4, 5)
        vector = (vector[0] * vector_length, vector[1] * vector_length)

        # make a random object by recoloring the central object, translating by the vector,
        # and then randomly removing parts of it by flipping random pixels to black
        other_sprite = np.copy(central_sprite)
        other_sprite[other_sprite != Color.BLACK] = np.random.choice(Color.NOT_BLACK)

        # flip some random pixels to black:
        # first find the foreground (nonblack) pixels,
        # then randomly sample a subset of them to color black
        nonblack_pixels = np.argwhere(other_sprite != Color.BLACK)
        num_nonblack = len(nonblack_pixels)
        num_to_flip = np.random.randint(1, num_nonblack-3)
        random_subset_of_nonblack_pixels = random.sample(list(nonblack_pixels), k=num_to_flip)

        # color black
        for pixel in random_subset_of_nonblack_pixels:
            other_sprite[pixel[0], pixel[1]] = Color.BLACK        

        # place the new object near the center, but offset by the vector
        paste(grid, other_sprite, r + vector[0], c + vector[1], transparent=Color.BLACK, clip=True)

    return grid
