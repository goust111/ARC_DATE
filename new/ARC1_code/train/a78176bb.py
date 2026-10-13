import numpy as np
import random
from arc_utils import Color, bbox, collision, colors_present, contact, crop_bbox, draw_line, find_components, object_neighbors, paste


# concepts:
# diagonal lines

# description:
# In the input you will see a grid with a diagonal line and gray objects touching it. The grey objects are all right triangles.
# To make the output grid, draw additional diagonal lines in the same color emanating from the tip of the grey objects. Delete the grey objects.

def main(input_grid):
    """
    Extend the colored diagonal through gray triangle tips and remove the guides.

    Input structure:
    - The grid contains one colored diagonal line on black background.
    - Gray right-triangle objects touch that diagonal.
    - The diagonal color is the only non-gray foreground color.

    Rule:
    - Determine the diagonal color and orientation.
    - For each gray object, find the bordering tip farthest from the original diagonal.
    - Draw a same-colored diagonal through that tip in both directions, then erase gray pixels.

    Key insight:
    - Gray triangles are construction guides whose remote tips seed the new parallel diagonals.
    """
    # Plan:
    # 1. Find the diagonal (it's the only one) and greys
    # 2. For each grey, find the tip
    # 3. ... and draw a diagonal line in the same color and same direction
    # 4. Delete the greys
    
    # 1. Input parsing: Find the grey objects, and then extract by color everything that is not grey
    bg = Color.BLACK
    grey_objects = [
        obj
        for obj in find_components(input_grid, bg=bg, connectivity=4, monochromatic=True, return_format="objects")
        if Color.GREY in colors_present(obj, exclude={bg})
    ]
    # extracting the diagonal by color: we know it's just everything that's not grey
    diagonal_object = input_grid.copy()
    diagonal_object[diagonal_object == Color.GREY] = bg

    # Parse out the color and directionbof the diagonal
    diagonal_color = colors_present(diagonal_object, exclude={bg})[0]
    b = bbox(diagonal_object != bg)
    assert b is not None
    diag_sprite = crop_bbox(diagonal_object, b)
    if int(diag_sprite[0, 0]) == diagonal_color:
        diagonal_direction = (1,1) # down-right
    else:
        diagonal_direction = (-1,1) # up-right

    # We draw on top of the input, so copy it
    output_grid = input_grid.copy()

    # 2. Find the tips of the grey objects
    for grey_object in grey_objects:
        # The tip is the bordering pixel farthest away from the diagonal
        bordering_pixels_mask = object_neighbors(grey_object, connectivity=8, background=bg)

        def distance_to_object(r: int, c: int, obj: np.ndarray) -> float:
            return float(
                min(np.linalg.norm([r - r2, c - c2]) for r2, c2 in np.argwhere(obj != bg))
            )

        tip = max(
            np.argwhere(bordering_pixels_mask),
            key=lambda rc: distance_to_object(int(rc[0]), int(rc[1]), diagonal_object),
        )
        tip_r, tip_c = map(int, tip)

        # 3. Draw the diagonal line
        L = max(output_grid.shape)
        draw_line(output_grid, tip_r, tip_c, direction=diagonal_direction, length=L, color=diagonal_color)
        draw_line(
            output_grid,
            tip_r,
            tip_c,
            direction=(-diagonal_direction[0], -diagonal_direction[1]),
            length=L,
            color=diagonal_color,
        )

    # 4. Delete grey
    output_grid[output_grid == Color.GREY] = bg

    return output_grid

def generate_input():
    # Generate the background grid
    grid_len = np.random.randint(10, 20)
    grid = np.full((grid_len, grid_len), Color.BLACK)

    diagonal_color = random.choice([ color for color in Color.NOT_BLACK if color != Color.GREY ])
    diagonal_direction = (1,1) # rotate at the end in order to get variety of orientations5
    diagonal_r, diagonal_c = random.choice(np.argwhere(grid == Color.BLACK))
    draw_line(grid, diagonal_r, diagonal_c, direction=diagonal_direction, color=diagonal_color)
    draw_line(grid, diagonal_r, diagonal_c, direction=(-diagonal_direction[0], -diagonal_direction[1]), color=diagonal_color)

    # Randomly generate grey objects and place them on the grid so they are touching the diagonal
    num_grey_objects = random.choice([1, 2, 3])
    for _ in range(num_grey_objects):
        # the grey sprite is a right-triangle of random size
        grey_size = np.random.randint(2, 5)
        grey_sprite = np.full((grey_size, grey_size), Color.BLACK)
        for r in range(grey_size):
            for c in range(grey_size):
                if r<=c: grey_sprite[r, c] = Color.GREY
        
        # random orientation
        if np.random.rand() < 0.5: grey_sprite = grey_sprite.T

        # find a placement so it touches (without overlapping the diagonal)
        diagonal_mask = (grid == diagonal_color).astype(int)
        H, W = grid.shape
        h, w = grey_sprite.shape
        candidates: list[tuple[int, int]] = []
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                if collision(object1=grid, object2=grey_sprite, r2=r, c2=c, background=Color.BLACK):
                    continue
                if contact(object1=diagonal_mask, object2=grey_sprite, r2=r, c2=c, background=Color.BLACK):
                    candidates.append((r, c))
        if not candidates:
            continue
        r, c = random.choice(candidates)
        paste(grid, grey_sprite, r, c, transparent=Color.BLACK, clip=True)
    
    # Randomly rotate the grid
    grid = np.rot90(grid, k=-(np.random.randint(4)))

    return grid
# ============= remove below this point for prompting =============
