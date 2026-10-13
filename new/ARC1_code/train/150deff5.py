import numpy as np
import random
from arc_utils import Color, bbox, crop_bbox, detect_objects, draw_line, first_free_location, paste


# concepts:
# decomposition, color change

# description:
# In the input you will see grey-colored regions on a medium sized black canvas. These regions are comprised of 2x2 squares and 1x3/3x1 rectangles, but this might be hard to see because regions might be touching.
# To make the output, decompose the input into 2x2 squares and 1x3/3x1 rectangles, and color them as follows:
# 1. Color teal the 2x2 squares
# 2. Color red the 1x3/3x1 rectangles

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decompose gray blocks into squares and length-three rectangles, then recolor by shape.

    Input structure:
    - The input contains touching or separated gray pieces on a black background.
    - Valid primitive pieces are 2x2 squares and 1x3 or 3x1 rectangles.
    - All foreground pixels are gray before decomposition.

    Rule:
    - Detect a non-overlapping decomposition into the allowed primitive dimensions.
    - Recolor each 2x2 square teal.
    - Recolor each length-three rectangle red and paste it back at its original location.

    Key insight:
    - Apparent connected regions may be composites; the output color depends on the recovered
      primitive shape, not the whole connected blob.
    """

    # Decompose the grid into non-overlapping grey regions
    # We need a custom predicate because otherwise the regions are also allowed to include background pixels, and we want it all grey
    decomposition = detect_objects(input_grid, background=Color.BLACK, colors=[Color.GREY],
                                   allowed_dimensions=[(2, 2), (3, 1), (1, 3)], # 2x2 squares and 1x3/3x1 rectangles
                                   predicate=lambda sprite: np.all(sprite == Color.GREY))

    output_grid = np.full(input_grid.shape, Color.BLACK)

    for obj in decomposition:
        b = bbox(obj != Color.BLACK)
        assert b is not None
        r0, c0, r1, c1 = b
        sprite = crop_bbox(obj, b)

        # Color change based on dimensions: 2x2 -> teal, 1x3/3x1 -> red
        if sprite.shape == (2, 2):
            sprite[sprite == Color.GREY] = Color.TEAL
        elif sprite.shape in [(3, 1), (1, 3)]:
            sprite[sprite == Color.GREY] = Color.RED
        else:
            assert 0, "Invalid object found"
        
        # Copy the sprite back into the output grid
        output_grid = paste(output_grid, sprite, r0, c0, transparent=Color.BLACK)

    return output_grid


def generate_input():
    n, m = np.random.randint(10, 15), np.random.randint(10, 15)
    grid = np.zeros((m, n), dtype=int)

    n_objects = np.random.randint(4, 8)

    for _ in range(n_objects):
        # Make a gray rectangle sprite
        color = Color.GREY
        w, h = random.choice([(2, 2), (3, 1), (1, 3)])
        sprite = np.full((w, h), color)

        # place it randomly on the grid, assuming we can find a spot
        try:
            _loc = first_free_location(grid, sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
            if _loc is None:
                raise ValueError("no placement")
            r, c = _loc
        except:
            continue

        paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    return grid
# ============= remove below this point for prompting =============
