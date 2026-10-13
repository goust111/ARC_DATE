
import random

import numpy as np
from typing import *
from arc_utils import Color, bbox, bbox_anchor, draw_line, find_components

# concepts:
# attraction, magnetism

# description:
# In the input you will individual pixels are the sides of a grid. Each pixel is in a matching pair, which is a different color, but on the opposite side of the grid.
# To make the output, make each pixel copy its color toward its matching pair until they meet in the middle. Turn the middle point grey.

def main(input_grid):
    """
    March each opposite-side pixel pair toward the center and mark the meeting point gray.

    Input structure:
    - Non-black single pixels appear in pairs on opposite sides of a black grid.
    - Each pair lies on a shared row or column after considering horizontal or vertical reflection.
    - Distances are arranged so each pair has one meeting cell.

    Rule:
    - Detect each single-pixel component.
    - Match it to the one non-background reflected pixel on the opposite side.
    - Move both endpoints toward each other, leaving trails of their own colors and coloring the meeting cell gray.

    Key insight:
    - Opposite-side reflection identifies the pairings, and simultaneous inward motion creates the two-color trails.
    """
    # Plan:
    # 1. Detect the pixels
    # 2. Associate each pixel with its matching pair
    # 3. Make them attract/march toward each other until they meet in the middle, leaving a trail of their own color

    # 1. Find the location of the pixels
    pixel_objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")
    assert len(pixel_objects) % 2 == 0, "There should be an even number of pixels"

    # we're going to draw on top of the input grid
    output_grid = input_grid.copy()

    seen: set[tuple[int, int]] = set()
    H, W = input_grid.shape
    for obj in pixel_objects:
        # 2. Associate each pixel with its matching pair
        # Find the matching position. The legacy seed solver ran under a transpose wrapper;
        # the most robust way here is to probe both possible "opposite side" candidates
        # and pick the one that is non-background.
        _b = bbox(obj != Color.BLACK)
        assert _b is not None
        r, c = bbox_anchor(_b, anchor="center")
        if (r, c) in seen:
            continue

        candidates: list[tuple[int, int]] = []
        # vertical reflection
        rv, cv = H - 1 - r, c
        if 0 <= rv < H and 0 <= cv < W and int(input_grid[rv, cv]) != int(Color.BLACK):
            candidates.append((rv, cv))
        # horizontal reflection
        rh, ch = r, W - 1 - c
        if 0 <= rh < H and 0 <= ch < W and int(input_grid[rh, ch]) != int(Color.BLACK):
            candidates.append((rh, ch))

        assert len(candidates) == 1, "Expected exactly one non-background opposite pixel"
        opposite_r, opposite_c = candidates[0]
        seen.add((r, c))
        seen.add((opposite_r, opposite_c))

        # get the unit vector pointing from one to the other
        dr = int(np.sign(opposite_r - r))
        dc = int(np.sign(opposite_c - c))

        color = input_grid[r, c]
        other_color = input_grid[opposite_r, opposite_c]

        # 3. Make them attract/march toward each other until they meet in the middle (grey when they touch), leaving a trail of their own color
        while (r, c) != (opposite_r, opposite_c):
            # Draw a trail of color
            if output_grid[r, c] == Color.BLACK:
                output_grid[r, c] = color
            if output_grid[opposite_r, opposite_c] == Color.BLACK:
                output_grid[opposite_r, opposite_c] = other_color
            r += dr
            c += dc
            opposite_r -= dr
            opposite_c -= dc

            # Make sure we haven't fallen out of bounds
            if not (0 <= r < input_grid.shape[0] and 0 <= c < input_grid.shape[1] and 0 <= opposite_r < input_grid.shape[0] and 0 <= opposite_c < input_grid.shape[1]):
                break
        # when they meet, turn the middle point grey
        output_grid[r, c] = Color.GREY
    
    return output_grid
        



def generate_input():
    # We are going to generate a grid with pairs that go left-right, then randomly rotate
    # to get the other orientations.

    # Generate a grid with random size, but odd width so there is a unique middle point after marching.
    H = np.random.randint(5, 25)
    # sample an odd width directly
    W = int(random.choice([w for w in range(11, 30) if w % 2 == 1]))

    grid = np.full((H, W), Color.BLACK)

    # Choose rows so that for any pixel, exactly ONE of {vertical-reflection, horizontal-reflection}
    # is non-background (matching the solver's assumption).
    candidate_rows = [r for r in range(1, H - 1) if r < (H - 1 - r)]
    # Avoid adjacent pixels on the same border column (which could merge components).
    candidate_rows = candidate_rows[::2]
    max_pairs = min(6, len(candidate_rows))
    n_pairs = np.random.randint(1, max_pairs + 1)
    rows = random.sample(candidate_rows, k=n_pairs)

    for r in rows:
        color = random.choice([col for col in Color.NOT_BLACK if col != Color.GREY])
        other_color = random.choice([col for col in Color.NOT_BLACK if col not in {color, Color.GREY}])
        grid[r, 0] = int(color)
        grid[r, W - 1] = int(other_color)

    # random rotation
    grid = np.rot90(grid, -(np.random.randint(0, 4)))
    return grid
