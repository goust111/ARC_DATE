import numpy as np
import random
from arc_utils import Color, bbox, bbox_mask, collision, contact, crop_bbox, detect_objects, draw_line, paste, random_sprite


# concepts:
# puzzle pieces, contact

# description:
# In the input you will see blue puzzles pieces lying beneath a red shape with holes on its underside.
# To make the output, move the gray puzzle pieces upward so that they fit into the holes on the underside. They need to fully plug the holes.
# Finally change the grey puzzle pieces to be blue.

def main(input_grid):
    """
    Move gray puzzle pieces into holes in a red shape and recolor the placed pieces blue.

    Input structure:
    - The input has one red object on a black background.
    - Gray puzzle pieces are separate from the red object.
    - Black gaps inside the red object's bounding area are holes to be plugged.

    Rule:
    - Crop each gray puzzle piece.
    - Greedily place pieces where they do not collide with the red object and cover the most hole cells.
    - Paste all placed pieces and then recolor gray pixels to blue.

    Key insight:
    - The red object's bounding area identifies the target holes, so the best placements maximize plugged black cells.
    """
    # Plan:
    # 1. Detect the puzzle pieces and red thing
    # 2. Extract the sprites of each puzzle piece
    # 3. Try moving the puzzle pieces (but not so much that they collide with the red thing). You can translate but not rotate. Plug as much of the holes in the red thing as possible.
    # 4. Change the color of the puzzle pieces to blue (just change everything that's grey to blue)

    # 1. Separate puzzle pieces from the red object
    bg = Color.BLACK
    red_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=8, background=bg)
    assert len(red_objects) == 1, "There should be exactly one fully red object"
    red_object = red_objects[0]
    
    puzzle_pieces = detect_objects(grid=input_grid, colors=[Color.GREY], monochromatic=True, connectivity=4, background=bg)

    # 2. Extract sprites by cropping
    sprites = []
    for piece in puzzle_pieces:
        b = bbox(piece != bg)
        assert b is not None
        sprites.append(crop_bbox(piece, b))

    # Output begins with just the red and then we add stuff to it
    output_grid = red_object.copy()
    red_bbox_rect = bbox_mask(red_object != bg)

    # 3. Try moving the puzzle pieces, plugging as much of the red object is possible
    sprites_to_move = list(sprites)
    while sprites_to_move:
        possible_solutions = [
            (r0, c0, sprite)
            for sprite in sprites_to_move
            for r0 in range(output_grid.shape[0] - sprite.shape[0] + 1)
            for c0 in range(output_grid.shape[1] - sprite.shape[1] + 1)
        ]

        def score_solution(r0, c0, sprite):
            # The score is -inf if it collides with the red object
            # Otherwise it is the number of black pixels that are plugged by the sprite

            # Make a canvas by trying putting down the sprite
            test_canvas = np.full_like(output_grid, bg)
            test_canvas = paste(test_canvas, sprite, int(r0), int(c0), transparent=bg)

            # Check for collision
            if collision(object1=test_canvas, object2=output_grid, background=bg):
                return float("-inf")
            
            # Count the number of black pixels that are plugged by the sprite, only counting those within the bounding box of the red object
            red_object_mask = red_object != bg
            test_object_mask = test_canvas != bg
            plugged_pixels = test_object_mask & ~red_object_mask & red_bbox_rect
            
            return np.sum(plugged_pixels)
        
        best_r, best_c, best_sprite = max(possible_solutions, key=lambda solution: score_solution(*solution))

        # Blit the sprite into the output grid
        output_grid = paste(output_grid, best_sprite, int(best_r), int(best_c), transparent=bg)

        # Remove the sprite from the list of sprites to move
        sprites_to_move = [ sprite for sprite in sprites_to_move if sprite is not best_sprite ]

    # 4. grey->blue
    output_grid[output_grid == Color.GREY] = Color.BLUE

    return output_grid

def generate_input():
    # Red rectangle at the top, grey puzzle pieces below it.

    rect_h = int(np.random.randint(5, 10))
    rect_w = int(np.random.randint(15, 30))
    red_rectangle = np.full((rect_h, rect_w), Color.RED, dtype=int)

    # Full grid: same width as rectangle, extra height for pieces underneath.
    H = rect_h + int(np.random.randint(8, 15))
    W = rect_w
    grid = np.full((H, W), Color.BLACK, dtype=int)

    paste(grid, red_rectangle, 0, 0, transparent=Color.BLACK, clip=False)

    n_holes = int(np.random.randint(2, 4))
    pieces: list[np.ndarray] = []
    for _ in range(n_holes):
        ph = int(np.random.randint(1, 4 + 1))
        pw = int(np.random.randint(1, 4 + 1))
        piece = np.full((ph, pw), Color.GREY, dtype=int)
        ph, pw = piece.shape

        # Chisel a hole on the underside (bottom edge) of the red rectangle.
        r_hole = rect_h - ph
        c_hole = int(np.random.randint(0, rect_w - pw + 1))
        grid[r_hole : r_hole + ph, c_hole : c_hole + pw] = Color.BLACK

        pieces.append(piece)

    # Place the puzzle pieces below the red rectangle (don't overlap each other).
    for piece in pieces:
        ph, pw = piece.shape
        placed = False
        for _try in range(200):
            r0 = int(np.random.randint(rect_h + 1, H - ph + 1))
            c0 = int(np.random.randint(0, W - pw + 1))
            if not collision(object1=grid, object2=piece, r2=r0, c2=c0, background=Color.BLACK):
                paste(grid, piece, r0, c0, transparent=Color.BLACK, clip=True)
                placed = True
                break
        if not placed:
            break

    return grid
# ============= remove below this point for prompting =============
