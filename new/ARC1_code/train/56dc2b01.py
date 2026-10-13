
import numpy as np
from typing import *
import random
from arc_utils import Color, collision, colors_present, contact, draw_line, find_components, first_free_location, paste, random_sprite, translate

# concepts:
# attraction, magnetism, translation5

# description:
# In the input you will see a green object and a red bar
# To make the output, move the green object to touch the red bar. Finally put a teal bar on the other side of the green object.

def main(input_grid):
    """
    Move the green object until it touches the red bar, then place a teal copy of the bar beyond it.

    Input structure:
    - The grid contains exactly one red bar and exactly one green object.
    - Both objects sit on a black background.
    - The scene may be rotated.

    Rule:
    - Find the shortest cardinal displacement that makes the green object touch the red bar.
    - Paste the green object at that touching position.
    - Recolor a copy of the red bar teal and translate it opposite the movement direction until clear.

    Key insight:
    - The red bar is both the contact target and the template for the teal bar placed on
      the far side of the moved green object.
    """
    # Plan:
    # 1. Detect the objects; separate the green thing from the red bar
    # 2. Move the green object to touch the red bar
    # 3. Add a teal bar on the other side of the green object

    # 1. Object detection and setup
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    red_objects = [ obj for obj in objects if Color.RED in colors_present(obj, exclude={ Color.BLACK }) ]
    green_objects = [ obj for obj in objects if Color.GREEN in colors_present(obj, exclude={ Color.BLACK }) ]

    assert len(red_objects) == 1, "There should be exactly one red object"
    assert len(green_objects) == 1, "There should be exactly one green object"
    
    red_object = red_objects[0]
    green_object = green_objects[0]

    # Make the output grid: Start with the red object, then add the green object and the teal bar
    output_grid = np.full_like(input_grid, Color.BLACK)
    paste(output_grid, red_object, 0, 0, transparent=Color.BLACK, clip=True)

    # 2. Move the green object to touch the red bar
    # First calculate what direction we have to move in order to contact the grey object
    # Consider all displacements, starting with the smallest translations first
    possible_displacements = [ (i*dr, i*dc) for i in range(0, 30) for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)] ]

    # Only keep the displacements that cause a contact between the colored object and the grey object
    valid_displacements = [ displacement for displacement in possible_displacements
                            if contact(object1=translate(green_object, *displacement), object2=red_object) ]
    assert valid_displacements, "There should be at least one valid displacement"

    # Pick the smallest valid displacement
    displacement = min(valid_displacements, key=lambda displacement: sum(abs(r) for r in displacement))

    # Extract the direction from the displacement
    direction = np.sign(displacement, dtype=int)

    # Translate and draw on the canvas
    green_object = translate(green_object, *displacement)
    paste(output_grid, green_object, 0, 0, transparent=Color.BLACK, clip=True)

    # 3. Add a teal bar on the other side of the green object
    # It should be the same shape as the red bar, but teal
    # To place it correctly, it needs to be on the other side so we go in the opposite direction that the green object moved
    teal_object = red_object.copy()
    teal_object[teal_object != Color.BLACK] = Color.TEAL
    opposite_direction = -direction
    # Move the teal object until it doesn't collide with anything
    while collision(object1=teal_object, object2=output_grid):
        teal_object = translate(teal_object, *opposite_direction)
    # Draw the teal object on the canvas
    paste(output_grid, teal_object, 0, 0, transparent=Color.BLACK, clip=True)
    
    return output_grid

def generate_input():
    # Make a long skinny grid with a green thing and a red vertical bar
    # Then randomly rotate to get a variety of orientations

    width, height = np.random.randint(10, 25), np.random.randint(3, 8)
    grid = np.full((width, height), Color.BLACK)

    bar_r = np.random.randint(0, width)
    grid[bar_r, :] = Color.RED

    green_sprite = random_sprite(3, height, color_palette=[Color.GREEN])
    _loc = first_free_location(grid, green_sprite, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    paste(grid, green_sprite, r, c, transparent=Color.BLACK, clip=True)

    # random rotation
    grid = np.rot90(grid,-(np.random.randint(0, 4)))

    return grid
