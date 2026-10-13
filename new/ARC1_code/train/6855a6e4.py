
import numpy as np
from typing import *
from arc_utils import Color, MirrorSymmetry, bbox, bbox_anchor, colors_present, draw_line, find_components, first_free_location, paste, random_sprite
import random

# concepts:
# symmetry, mirror

# description:
# In the input you will see two objects on each outer side of two red frames.
# To make the output, you need to mirror the two objects by flipping them over the symmetry of the red frames, making them inside the red frames. Each object flips over the frame closest to it.

def main(input_grid):
    """
    Mirror gray objects across their nearest red frame so the reflected copies land inside the frames.

    Input structure:
    - The input is a black grid with red frame objects and gray objects outside them.
    - All objects are aligned along one row or one column.
    - Each gray object belongs to the closest red frame.

    Rule:
    - Detect the red frames and gray objects.
    - Infer whether the reflection axis is horizontal or vertical from object alignment.
    - Reflect each gray object across the center line of its closest frame and redraw the frame.

    Key insight:
    - The red frames define local mirror axes; each gray object moves by reflection, not by free translation.
    """
    # Extract the framework
    frame_color = Color.RED
    object_color = Color.GRAY 
    background = Color.BLACK

    # Create an empty grid
    n, m = input_grid.shape
    output_grid = np.zeros((n, m), dtype=int)

    # parse the input
    objects = find_components(input_grid, bg=background, connectivity=8, monochromatic=True, return_format="objects")
    frames = [ obj for obj in objects if frame_color in colors_present(obj, exclude={ background }) ]
    mirrored_objects = [ obj for obj in objects if object_color in colors_present(obj, exclude={ background }) ]

    # determine if we are doing horizontal or vertical mirroring
    # if all the objects have the same X coordinate, we are doing vertical mirroring
    # if all the objects have the same Y coordinate, we are doing horizontal mirroring
    def _center(b: tuple[int, int, int, int]) -> tuple[float, float]:
        r0, c0, r1, c1 = map(int, b)
        return (r0 + r1) / 2.0, (c0 + c1) / 2.0

    centers: list[tuple[float, float]] = []
    for obj in objects:
        _b = bbox(obj != background)
        assert _b is not None
        centers.append(_center(_b))
    r_positions = [rc[0] for rc in centers]
    c_positions = [rc[1] for rc in centers]

    # Legacy note: the seed solver ran under a transpose wrapper, so the axis test swaps here.
    if all(c == c_positions[0] for c in c_positions):
        orientation = "vertical"
    elif all(r == r_positions[0] for r in r_positions):
        orientation = "horizontal"
    else: raise ValueError(f"The objects are not aligned in a single axis")

    # Flip each other object over its closest frame
    for mirrored_object in mirrored_objects:
        # Find the closest frame
        def distance_between_objects(obj1, obj2):
            _b = bbox(obj1 != background)
            assert _b is not None
            r1, c1 = _center(_b)
            _b = bbox(obj2 != background)
            assert _b is not None
            r2, c2 = _center(_b)
            return (r1 - r2) ** 2 + (c1 - c2) ** 2        
        closest_frame = min(frames, key=lambda frame: distance_between_objects(frame, mirrored_object))

        # Build a symmetry object for flipping over the closest frame
        _b = bbox(closest_frame != background)
        assert _b is not None
        frame_r, frame_c = _center(_b)
        _b = bbox(mirrored_object != background)
        assert _b is not None
        this_r, this_c = _center(_b)
        # Make it one pixel past the middle of frame
        frame_c += 0.5 if this_c > frame_c else -0.5
        frame_r += 0.5 if this_r > frame_r else -0.5
        # Conjugation note: legacy seeds ran under a transpose wrapper.
        # When objects are aligned vertically (same column), we mirror across a horizontal line (mirror_r).
        # When objects are aligned horizontally (same row), we mirror across a vertical line (mirror_c).
        symmetry = MirrorSymmetry(
            mirror_r=frame_r if orientation == "vertical" else None,
            mirror_c=frame_c if orientation == "horizontal" else None,
        )
        
        # Flip the object over the symmetry
        for r, c in np.argwhere(mirrored_object != background):
            r2, c2 = symmetry.apply(r, c)
            output_grid[r2, c2] = mirrored_object[r, c]
        
        # Draw the frame
        output_grid = paste(output_grid, closest_frame, 0, 0, transparent=Color.BLACK, clip=True)

    return output_grid

def generate_input():
    # Create a 2D background
    n, m = np.random.randint(15, 30), np.random.randint(15, 30)
    grid = np.zeros((m, n), dtype=int)

    # Get a framework for pattern
    frame_length = np.random.randint(5, 10)
    frame_width = 2
    frame_color = Color.RED

    # Draw a half of the framework
    frame_sprite = np.zeros((frame_length, frame_width), dtype=int)
    # horizontal bar and two vertical pixels
    frame_sprite[:, 0] = frame_color
    frame_sprite[0, 1] = frame_color
    frame_sprite[-1, 1] = frame_color

    # Select the interval for two half frameworks that form one entire framework
    frame_interval = np.random.randint(4, 6)
    
    # Calculate the sizes of two things that we are going to mirror within framwork
    pattern_color = Color.GRAY
    pattern_length = frame_length - 2
    pattern_width = frame_interval // 2
    pattern_width_with_padding = frame_interval // 2 + 1

    # Generate two patterns out the framwork
    pattern_1 = random_sprite(m=pattern_length, n=pattern_width, color_palette=[pattern_color])
    pattern_2 = random_sprite(m=pattern_length, n=pattern_width, color_palette=[pattern_color])

    # place the pattern in the frame
    whole_frame = np.zeros((frame_length, frame_width * 2 + frame_interval + pattern_width_with_padding * 2), dtype=int)
    # Place the upper half of the frame
    whole_frame = paste(whole_frame, frame_sprite, 0, pattern_width_with_padding, transparent=Color.BLACK, clip=True)
    # Place the upper pattern
    whole_frame = paste(whole_frame, pattern_1, 1, 0, transparent=Color.BLACK, clip=True)
    # Place the lower pattern
    whole_frame = paste(whole_frame, pattern_2, 1, frame_width * 2 + frame_interval + pattern_width_with_padding + 1, transparent=Color.BLACK, clip=True)
    # Place the lower half of the frame
    whole_frame = paste(whole_frame, np.flipud(frame_sprite), 0, frame_width + frame_interval + pattern_width_with_padding, transparent=Color.BLACK, clip=True)

    # place the frame in the background
    _loc = first_free_location(grid, whole_frame, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    if _loc is None:
        raise ValueError("no placement")
    r, c = _loc
    grid = paste(grid, whole_frame, r, c, transparent=Color.BLACK, clip=True)

    # Randomly rotate the grid so that we get both vertical and horizontal arrangements
    if np.random.rand() < 0.5:
        grid = np.rot90(grid, k=-1)
    return grid
