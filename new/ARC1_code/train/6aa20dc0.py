
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, colors_present, crop_bbox, draw_line, find_components, paste, random_free_location, random_sprite, scale_sprite

# concepts:
# puzzle pieces, rotation, rescaling

# description:
# In the input you will see a non-black background, a small multicolored object, and fragments of that object rescaled and rotated scattered around.
# To make the output, isolate the small multicolored object and then rescale/rotate/translate to cover the fragments as much as possible, matching color whenever the fragment has that color.

def main(input_grid):
    """
    Reconstruct scattered fragments by placing scaled and rotated copies of the multicolor template.

    Input structure:
    - The background is a non-black color.
    - One object is the multicolor template with the largest variety of colors.
    - Other objects are partial scaled and rotated fragments of that template.

    Rule:
    - Treat the most common non-black color as the background.
    - Try rotations and scale factors of the template against the remaining fragments.
    - Iteratively paste the best color-compatible placement and erase the matched fragment pixels.

    Key insight:
    - The complete template is distinguished by color variety, while fragments are partial observations of transformed copies.
    """
    # Plan:
    # 1. Detect the object, separating the small multicolored puzzle piece from its fragments
    # 2. Rescale/rotate/translate the small object
    # 3. Find the transformation covering as much of the fragments as possible, matching colors whenever they overlap
    # 4. Copy the resulting transformation to the output grid; delete the fragments from the input
    # 5. Repeat until all the fragments are gone

    # 1. object detection
    # because the background is not black, set it to be the most common color
    background = max(Color.NOT_BLACK, key=lambda color: np.sum(input_grid == color))
    # detect the objects and figure out which is the template puzzle piece, which is going to be the one with the largest variety of colors
    objects = find_components(input_grid, bg=background, connectivity=8, monochromatic=False, return_format="objects")
    template_object = max(objects, key=lambda obj: len(colors_present(obj, exclude={ background })))
    _b = bbox(template_object != background)
    assert _b is not None
    template_sprite = crop_bbox(template_object, _b)

    output_grid = np.full_like(input_grid, background)

    # 2. rescale/rotate/translate the small object to cover as much of the fragments as possible, matching colors whenever they overlap
    rescaled_and_rotated = [ np.rot90(scale_sprite(template_sprite, scale), k=rot)
                            for scale in [1, 2, 3, 4]
                            for rot in range(4) ]
    # A placement solution is a tuple of (x, y, sprite) where x, y is the top-left corner of the rotated/scaled sprite
    possible_solutions = [ (r, c, sprite)
                          for sprite in rescaled_and_rotated
                          for r in range(output_grid.shape[0] - sprite.shape[0])
                          for c in range(output_grid.shape[1] - sprite.shape[1]) ]
    
    # Keep on looping until we are out of things to copy to the output
    while np.any(input_grid != background):

        def score_solution(r, c, sprite):
            # The score is -inf if the placement violates non-background colors
            # Otherwise it is the number of pixels that match in color between the sprite and the input
            test_canvas = np.full_like(input_grid, background)
            paste(test_canvas, sprite, r, c, transparent=Color.BLACK, clip=True)

            if np.any( (test_canvas != background) & (input_grid != background) & (test_canvas != input_grid) ):
                return float("-inf")
            
            return np.sum( (test_canvas == input_grid) & (input_grid != background) )
        
        # Remove -inf solutions, and zero solutions
        possible_solutions = [ solution for solution in possible_solutions if score_solution(*solution) > 0 ]
        
        best_r, best_c, best_sprite = max(possible_solutions, key=lambda solution: score_solution(*solution))

        # 4. Copy the resulting transformation to the output grid; delete the fragments from the input
        # Copy output
        paste(output_grid, best_sprite, best_r, best_c, transparent=Color.BLACK, clip=True)
        # Delete from input
        for dr, dc in np.argwhere(best_sprite != background):
            input_grid[best_r + dr, best_c + dc] = background
        
        # 5. Repeat until all the fragments are gone
    
    return output_grid


def generate_input():
    # Create a multicolored template puzzle piece, plus 1-2 fragments that have strictly fewer colors
    # (so the solver's `template_object = max(..., key=len(colors_present))` picks the true template).

    background = int(random.choice(Color.NOT_BLACK))
    H, W = int(np.random.randint(18, 30)), int(np.random.randint(18, 30))
    grid = np.full((H, W), background, dtype=int)
    occupied = np.zeros((H, W), dtype=int)

    # Build a small contiguous template with many distinct colors (no BLACK, no background).
    template_h = int(random.choice([3, 4]))
    template_w = int(random.choice([3, 4]))
    available = [int(c) for c in Color.NOT_BLACK if int(c) != background]
    n_colors = int(min(len(available), max(3, min(template_h * template_w, 7))))
    palette = random.sample(available, k=n_colors)

    template_puzzle_piece = np.full((template_h, template_w), background, dtype=int)
    cells = [(r, c) for r in range(template_h) for c in range(template_w)]
    random.shuffle(cells)
    for i, (r, c) in enumerate(cells):
        template_puzzle_piece[r, c] = palette[i % n_colors]

    r, c = random_free_location(occupied, np.ones_like(template_puzzle_piece), transparent=0, border=1, padding=0, padding_connectivity=8)
    occupied[r : r + template_h, c : c + template_w] = 1
    paste(grid, template_puzzle_piece, r, c, transparent=background, clip=True)

    n_fragments = int(np.random.randint(1, 2 + 1))
    for _ in range(n_fragments):
        fragment = template_puzzle_piece.copy()

        # Remove an entire color so fragments have strictly fewer colors than the template.
        drop_color = random.choice(palette)
        fragment[fragment == drop_color] = background

        # Additional random occlusion.
        fragment[np.random.rand(*fragment.shape) < 0.2] = background

        # If we accidentally removed everything, skip this fragment.
        if not bool((fragment != background).any()):
            continue

        # Random rotation and (bounded) scaling.
        fragment = np.rot90(fragment, k=int(np.random.randint(0, 4)))
        fragment = scale_sprite(fragment, int(np.random.randint(1, 4)))

        try:
            fh, fw = fragment.shape
            r, c = random_free_location(occupied, np.ones((fh, fw), dtype=int), transparent=0, border=1, padding=2, padding_connectivity=8)
        except ValueError:
            break
        occupied[r : r + fh, c : c + fw] = 1
        paste(grid, fragment, r, c, transparent=background, clip=True)

    return grid
