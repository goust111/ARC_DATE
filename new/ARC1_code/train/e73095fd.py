
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, crop_bbox, draw_line, find_components, first_free_location, object_neighbors, paste

# concepts:
# filling

# description:
# The input consists of a black grid containing a few hollow grey rectangles. Each rectangle has 1-3 grey horizontal or vertical lines emanating off of it (at most one per side), either travelling to the border or stopping at another rectangle.
# To create the output, fill in the hollow grey rectangles with yellow.

def main(input_grid):
    """
    Fill hollow gray rectangles with yellow while preserving gray outlines and lines.

    Input structure:
    - The input has black regions bounded by gray rectangle outlines.
    - Gray lines may emanate from rectangle sides.
    - The desired fill affects only true hollow rectangle interiors.

    Rule:
    - Treat gray as the background for separating black interior regions.
    - Identify regions whose cropped area is entirely black and whose border is not part of a line.
    - Recolor accepted black interiors yellow and paste all regions back over a gray canvas.

    Key insight:
    - Rectangle interiors are found as black holes inside gray structure, while extra
      gray line contacts are filtered out before filling.
    """
    # extract objects using grey as background
    objects = find_components(input_grid, bg=Color.GREY, connectivity=4, monochromatic=True, return_format="objects")

    # create an output grid to store the result
    output_grid = np.full(input_grid.shape, Color.GREY)

    # for each object, fill it in if it is a rectangle
    for obj in objects:
        # to check if the object is a rectangle,
        # we can check if the cropped object is entirely black
        _b = bbox(obj != Color.GREY)
        assert _b is not None
        sprite = crop_bbox(obj, _b)
        is_rectangle = np.all(sprite == Color.BLACK)

        if is_rectangle:
            # we also need to make sure the rectangle isn't caused from an emanating line.
            # to do so, check for grey pixels around the border of the grey
            # border adjacent to a corner (aka the x/y value is one less than the max)
            border = object_neighbors(obj, background=Color.GREY, connectivity=8)
            # to get the border of the border, make a copy,
            # add yellow where the border is,
            # then find the border of this new object
            obj2 = obj.copy()
            obj2[border] = Color.YELLOW
            border2 = object_neighbors(obj2, background=Color.GREY, connectivity=8)
            _b = bbox(obj2 != Color.GREY)
            assert _b is not None
            r, c, _r1, _c1 = _b
            w = _r1 - r + 1
            h = _c1 - c + 1

            pixels_to_check = [
                (x2, y2) for x2 in range(obj.shape[0]) for y2 in range(obj.shape[1])
                if (border2[x2, y2]
                    # check if pixel is adjacent to a corner
                    and (x2 in [r, r + w-1] or y2 in [c, c + h-1]))
            ]

            if not any(input_grid[r, c] == Color.GREY for r, c in pixels_to_check):
                # good rectangle!
                # fill in the original object with yellow
                obj[obj == Color.BLACK] = Color.YELLOW

        paste(output_grid, obj, 0, 0, transparent=Color.GREY, clip=True)

    return output_grid


def generate_input():
    # create a 10-20x10-20 black grid
    # to make it possible to place rectangles that go offscreen,
    n = np.random.randint(10, 21)
    m = np.random.randint(10, 21)
    n, m = 8, 8
    # add a pixel of padding, which we will remove later
    m += 1
    n += 1
    grid = np.full((m, n), Color.BLACK)

    # add 2-4 grey rectangles with hollow insides
    # each rectangle can be 3-7x3-7
    # if we can't find a space, just stop adding new rectangles
    # num_rectangles = np.random.randint(2, 5)
    num_rectangles = 1

    # store the rectangles so we can add lines to them later
    rectangles = []
    for _ in range(num_rectangles):
        r_n = np.random.randint(3, 8)
        r_m = np.random.randint(3, 8)
        rectangle = np.full((r_n, r_m), Color.GREY)
        try:
            _loc = first_free_location(grid, rectangle, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except ValueError:
            break

        # make the rectangle hollow, but mark it yellow, so we don't place another rectangle inside of it.
        # later we will convert yellow back to black
        rectangle[1:-1, 1:-1] = Color.YELLOW
        paste(grid, rectangle, r, c, clip=True)
        rectangles.append((rectangle, (r, c)))

    # now draw 1-3 lines emanating from the rectangles
    # a line from another rectangle might emanate into the rectangle, so check for existing lines before adding one
    for rectangle, (r, c) in rectangles:
        # num_lines = np.random.randint(1, 4)
        num_lines = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        indices = np.random.choice(len(directions), num_lines, replace=False)
        directions = [directions[i] for i in indices]

        for direction in directions:
            # if there's already a line here, don't draw another one
            # to check, look for grey pixel within 1 pixel of the rectangle
            w, h = rectangle.shape

            bordering_pixels = find_bordering_pixels_in_direction(r, c, w, h, direction)
            bordering_pixels = np.array([(r, c) for r, c in bordering_pixels if 0 <= r < m and 0 <= c < n])
            if len(bordering_pixels) == 0:
                continue

            if any(grid[r, c] == Color.GREY for r, c in bordering_pixels):
                continue

            # now draw a line emanating outwards at a random bordering pixel
            # ignore the extreme most bordering pixels (assumes they are returned in order)
            bordering_pixels = bordering_pixels[1:-1]

            ix = np.random.choice(range(len(bordering_pixels)))
            start_pixel = bordering_pixels[ix]
            draw_line(grid, start_pixel[0], start_pixel[1], direction=direction, color=Color.GREY, stop_at=[Color.GREY])

    # convert yellow back to black
    grid[grid == Color.YELLOW] = Color.BLACK

    # remove the pixel of padding
    grid = grid[1:-1, 1:-1]

    return grid


def find_bordering_pixels_in_direction(r, c, w, h, direction):
    dr, dc = direction
    if dr == 1:  # right
        bordering_pixels = [(r + w, c + i) for i in range(h)]
    elif dr == -1:  # left
        bordering_pixels = [(r - 1, c + i) for i in range(h)]
    elif dc == 1:  # down
        bordering_pixels = [(r + i, c + h) for i in range(w)]
    elif dc == -1:  # up
        bordering_pixels = [(r + i, c - 1) for i in range(w)]

    return np.array(bordering_pixels)


