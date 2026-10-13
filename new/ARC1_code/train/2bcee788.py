import numpy as np
import random
from arc_utils import Color, MirrorSymmetry, bbox, bbox_anchor, collision, colors_present, draw_line, find_components, orbit, paste, random_free_location, random_sprite, translate


# concepts:
# mirror symmetry, non-black background, indicator pixels

# description:
# In the input you will see an object with some red pixels attached to it on one side.
# To make the output, mirror the object to cover the red pixels. Then change the background to green.
 
def main(input_grid):
    """
    Mirror the main object across the side indicated by red marker pixels and turn the background green.

    Input structure:
    - A black grid contains one non-red main object.
    - A connected red marker object touches one side of the main object.
    - The red side indicates where the mirrored copy should appear.

    Rule:
    - Separate the main object from the red pixels.
    - Infer a mirror axis just outside the touched side of the main object's bounding box.
    - Reflect every main-object pixel across that axis, then recolor all remaining black background green.

    Key insight:
    - The red pixels are directional markers to be covered by the reflection, not output objects to preserve.
    """
    # Plan:
    # 1. Find the main object and the red pixels
    # 2. Calculate the axis over which to mirror depending on which side the red pixels are on
    # 3. Do the mirroring
    # 4. Change background to green
    
    # 1. Input parsing
    bg = Color.BLACK
    objects = find_components(input_grid, bg=bg, connectivity=8, monochromatic=True, return_format="objects")
    assert len(objects) == 2, "There should be exactly two objects"

    # Find the main object
    main_object = next(obj for obj in objects if Color.RED not in colors_present(obj, exclude={bg}))
    # Find the red pixels
    red_pixels = next(obj for obj in objects if Color.RED in colors_present(obj, exclude={bg}))

    # 2. Axis calculation

    # Figure out what side of the object the red pixels are on
    b = bbox(main_object != bg)
    assert b is not None
    r1, c1 = bbox_anchor(b, anchor="upper left")
    r2, c2 = bbox_anchor(b, anchor="lower right")
    # on the right?
    if collision(object1=translate(main_object, dr=0, dc=1, background=bg), object2=red_pixels, background=bg):
        # the +/- 0.5 is to clobber the red pixels, otherwise we'd reflect over them and leave them be, which would also be a reasonable thing to do
        symmetry = MirrorSymmetry(mirror_r=None, mirror_c=c2 + 0.5)
    # on the left?
    elif collision(object1=translate(main_object, dr=0, dc=-1, background=bg), object2=red_pixels, background=bg):
        symmetry = MirrorSymmetry(mirror_r=None, mirror_c=c1 - 0.5)
    # on the top?
    elif collision(object1=translate(main_object, dr=-1, dc=0, background=bg), object2=red_pixels, background=bg):
        symmetry = MirrorSymmetry(mirror_r=r1 - 0.5, mirror_c=None)
    # on the bottom?
    elif collision(object1=translate(main_object, dr=1, dc=0, background=bg), object2=red_pixels, background=bg):
        symmetry = MirrorSymmetry(mirror_r=r2 + 0.5, mirror_c=None)
    else:
        assert False, "Red pixels are not on any side of the main object"
    
    # 3. Mirror the main object
    output_grid = main_object.copy()
    for r, c in np.argwhere(main_object != bg):
        rr = int(r)
        cc = int(c)
        for r2, c2 in orbit(output_grid, rr, cc, symmetries=[symmetry]):
            if 0 <= r2 < output_grid.shape[0] and 0 <= c2 < output_grid.shape[1]:
                output_grid[r2, c2] = main_object[rr, cc]
    
    # 4. Change the background to green
    output_grid[output_grid == bg] = Color.GREEN

    return output_grid

def generate_input():
    # Make an empty black grid of random size, and then put a non-red object somewhere. Put some red pixels on one side.
    background = Color.BLACK
    grid = np.full((random.randint(10, 25), random.randint(10, 25)), background)

    object_color = random.choice([color for color in Color.NOT_BLACK if color != Color.RED])
    h, w = random.randint(3, 6), random.randint(3, 6)
    sprite = np.full((h, w), object_color, dtype=int)
    # Put red on one side (the rightmost column). We'll randomly rotate at the end for variety.
    sprite[:, -1] = Color.RED

    # placed randomly but a little bit away from the border
    r, c = random_free_location(grid, sprite, transparent=Color.BLACK, border=1, padding=0, padding_connectivity=8)
    paste(grid, sprite, r, c, transparent=Color.BLACK, clip=True)

    # randomly rotate
    grid = np.rot90(grid,-(random.randint(0, 4)))

    return grid

    

# ============= remove below this point for prompting =============
