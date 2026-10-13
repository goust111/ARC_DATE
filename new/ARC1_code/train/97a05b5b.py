
import numpy as np
from typing import *
from arc_utils import Color, bbox, colors_present, crop_bbox, draw_line, find_components, paste, random_free_location, random_sprite
import random

# concepts:
# puzzle pieces, rotation

# description:
# In the input you will see a red object with holes in it, and several multicolored 3x3 "puzzle pieces"
# To make the output, crop to just the red object and then put each puzzle piece in the red object so that the red pixels in the puzzle piece perfectly fit into the holes in the red object. You can rotate and translate the puzzle pieces to make them fit.

def main(input_grid):
    """
    Crop the red object and fit each multicolor puzzle piece into its matching red-shaped hole.

    Input structure:
    - The input contains one fully red object with black holes.
    - It also contains several multicolor puzzle pieces on the black background.
    - Puzzle pieces may include red cells that act as the hole-fitting stencil.

    Rule:
    - Crop the red object to form the output canvas.
    - Sort puzzle pieces by the number of red cells so larger stencils are placed first.
    - Try rotations and translations, placing each piece where its red cells land only on black holes.

    Key insight:
    - Red cells inside a piece are not paint for the red object; they are the stencil that tells
      which hole shape the piece can occupy.
    """
    # Plan:
    # 1. Detect the object, separating the puzzle pieces from the red object with holes
    # 2. Crop to just the red object
    # 3. Put each puzzle piece so that its red pixels perfectly overlay holes in the red object (considering translations and rotations)

    # 1. Separate puzzle pieces from the red object (remembering that puzzle pieces might have some red in them also: they are not monochromatic---but the red thing is fully red)
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=8, monochromatic=False, return_format="objects")
    red_objects = [ obj for obj in objects if set(colors_present(obj, exclude={ Color.BLACK })) == {Color.RED} ]
    puzzle_pieces = [ obj for obj in objects if set(colors_present(obj, exclude={ Color.BLACK })) != {Color.RED} ]

    assert len(red_objects) == 1, "There should be exactly one fully red object"
    red_object = red_objects[0]

    # 2. The output begins by cropping to just the red object
    _b = bbox(red_object != Color.BLACK)
    assert _b is not None
    output_grid = crop_bbox(red_object, _b)

    # 3. Put each puzzle piece so that its red pixels perfectly overlay holes in the red object
    # Put in the pieces which have the most red first (otherwise a small puzzle piece might go in a big hole)
    puzzle_pieces.sort(key=lambda piece: np.sum(piece == Color.RED), reverse=True)

    for puzzle_piece in puzzle_pieces:
        # Extract just the sprite of this piece, and then figure out where it should go, including its position and rotation
        _b = bbox(puzzle_piece != Color.BLACK)
        assert _b is not None
        piece_sprite = crop_bbox(puzzle_piece, _b)
        possible_sprites = [ piece_sprite, np.rot90(piece_sprite), np.rot90(piece_sprite, k=2), np.rot90(piece_sprite, k=3) ]

        # A placement solution is a tuple (x, y, sprite) where x, y is the top-left corner of the rotated sprite
        possible_solutions = [ (r, c, sprite)
                              for sprite in possible_sprites
                              for r in range(output_grid.shape[0] - piece_sprite.shape[0] + 1)
                              for c in range(output_grid.shape[1] - piece_sprite.shape[1] + 1) ]
        
        # Filter placement solutions to only those where the red pixels of the piece fit into the holes of the red object
        def valid_solution(r, c, sprite):
            # Make a canvas to try putting down the sprite
            test_canvas = np.full_like(output_grid, Color.BLACK)
            paste(test_canvas, sprite, r, c, transparent=Color.BLACK, clip=True)
            # Check if every red pixel in the placed test object is also red in the red object
            red_pixels = [ (r, c) for r, c in np.argwhere(test_canvas == Color.RED) ]
            return all( output_grid[red_r, red_c] == Color.BLACK for red_r, red_c in red_pixels )
                
        possible_solutions = [ solution for solution in possible_solutions if valid_solution(*solution) ]
        if len(possible_solutions) == 0:
            assert False, "No solution found for puzzle piece"

        # Pick the first solution and blit the sprite into the output grid
        r, c, sprite = list(possible_solutions)[0]
        paste(output_grid, sprite, r, c, transparent=Color.BLACK, clip=True)
    
    return output_grid

def generate_input():
    # Choose 2-4 distinct piece shapes with different red-pixel counts (so the greedy solver order is unambiguous).
    # We keep holes small to avoid disconnecting the red object.
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    shape_templates: list[list[tuple[int, int]]] = [
        [cell for cell in all_cells if cell != (1, 1)],  # 8 pixels (1 hole at center)
        [cell for cell in all_cells if cell not in {(1, 1), (0, 0)}],  # 7 pixels (2 holes)
        [cell for cell in all_cells if cell not in {(1, 1), (0, 0), (2, 2)}],  # 6 pixels (3 holes)
        [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)],  # plus shape (5 pixels / 4 holes)
    ]
    n_pieces = np.random.randint(2, 5)
    min_side = 10 if n_pieces <= 2 else (12 if n_pieces == 3 else 14)
    canvas_h, canvas_w = np.random.randint(min_side, 20), np.random.randint(min_side, 20)
    red_canvas = np.full((canvas_h, canvas_w), Color.RED)
    chosen_templates = random.sample(shape_templates, k=n_pieces)

    pieces: list[np.ndarray] = []

    # Reserve non-overlapping 3x3 blocks on a coarse lattice (guaranteed fit, no retries).
    row_starts = list(range(1, canvas_h - 3, 5))
    col_starts = list(range(1, canvas_w - 3, 5))
    block_positions = [(r, c) for r in row_starts for c in col_starts]
    block_positions = random.sample(block_positions, k=n_pieces)

    for template, (r0, c0) in zip(chosen_templates, block_positions):
        non_red_color = random.choice([c for c in Color.NOT_BLACK if c != Color.RED])

        piece = np.full((3, 3), Color.RED, dtype=int)
        for rr, cc in template:
            piece[int(rr), int(cc)] = int(non_red_color)
        pieces.append(piece)

        # Punch holes using a rotated version of the piece (solver is allowed to rotate pieces).
        rotated_piece = np.rot90(piece, k=np.random.randint(0, 4))

        holes = (rotated_piece == Color.RED)
        red_canvas[r0 : r0 + 3, c0 : c0 + 3][holes] = Color.BLACK

    # Create the final input grid.
    # Place the red canvas and all pieces in a guaranteed non-overlapping layout (no retries).
    sep = 2  # empty columns between the canvas and the pieces
    min_W = canvas_w + sep + 3
    min_H = max(canvas_h, (n_pieces - 1) * 4 + 3)
    H = np.random.randint(min_H, 30 + 1)
    W = np.random.randint(min_W, 30 + 1)
    grid = np.full((H, W), Color.BLACK)

    # Place the red canvas on the left.
    r_canvas = np.random.randint(0, H - canvas_h + 1)
    c_canvas = 0
    paste(grid, red_canvas, r_canvas, c_canvas, transparent=Color.BLACK, clip=True)

    # Place all pieces in a vertical strip on the right, spaced out.
    c_piece = canvas_w + sep
    r0 = np.random.randint(0, H - ((n_pieces - 1) * 4 + 3) + 1)
    for i, piece in enumerate(pieces):
        r = r0 + i * 4
        paste(grid, piece, r, c_piece, transparent=Color.BLACK, clip=True)

    # Random rotation for variety.
    grid = np.rot90(grid, k=np.random.randint(0, 4))
    return grid
