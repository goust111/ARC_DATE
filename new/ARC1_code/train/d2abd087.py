import numpy as np
from arc_utils import Color, draw_line, find_components, first_free_location, paste, random_sprite


# concepts:
# counting

# description:
# The input consists of several grey objects in a 10x10 grid.
# To create the output, change the color of all objects of area 6 to red, and all other objects to blue.

def main(input_grid):
    """
    Recolor gray objects red when their area is six and blue otherwise.

    Input structure:
    - The input is a 10x10 black grid containing gray objects.
    - Objects are 4-connected gray components.
    - Some components have area exactly six.

    Rule:
    - Detect all gray components on the black background.
    - Recolor each component with area six red.
    - Recolor every other component blue on a black output canvas.

    Key insight:
    - Component area alone controls the output color; each object's shape is preserved.
    """
    # extract objects
    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=4, monochromatic=True, return_format="objects")

    # convert each object to the desired color
    output_grid = np.full_like(input_grid, bg)
    for obj in objects:
        new_color = Color.RED if int(np.sum(obj != bg)) == 6 else Color.BLUE
        obj2 = obj.copy()
        obj2[obj2 != bg] = int(new_color)
        output_grid = paste(output_grid, obj2, 0, 0, transparent=bg)

    return output_grid


def generate_input():
    # create a 10x10 grid
    grid = np.full((10, 10), Color.BLACK)

    # generate objects and place into the grid until it is filled
    # we want some area six objects, and some non-area six objects.
    # to do so, first place a couple area six objects, then fill up the remaining space with random objects

    # place two area six objects
    for _ in range(2):
        while True:
            obj = random_sprite(list(range(1, 5)), list(range(1, 5)), color_palette=[Color.GREY])
            if np.sum(obj != Color.BLACK) == 6:
                break
        _loc = first_free_location(grid, obj, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        if _loc is None:
            raise ValueError("no placement")
        r, c = _loc
        grid = paste(grid, obj, r, c, transparent=Color.BLACK, clip=True)

    # now fill up the remaining space with random objects.
    while True:
        obj = random_sprite(list(range(1, 5)), list(range(1, 5)), color_palette=[Color.GREY])

        # try to place the object. if we can't, we're done
        try:
            _loc = first_free_location(grid, obj, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
            grid = paste(grid, obj, r, c, transparent=Color.BLACK, clip=True)
        except ValueError:
            break

    return grid
# ============= remove below this point for prompting =============
