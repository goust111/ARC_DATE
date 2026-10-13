import numpy as np

from arc_utils import bbox_w, find_components, get_bg, paste, random_sprite
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Shift every foreground object to the right by its own bounding-box width.

    Input structure
    - The grid has one background color and one or more 8-connected foreground
      objects.
    - Each object has enough empty space to its right for the shift.

    Rule
    - Find the 8-connected foreground objects.
    - Clear the canvas to the background color.
    - Repaint each object shifted horizontally right by the width of its
      bounding box.

    Key insight
    - The horizontal offset is local to each object: a width-3 object shifts by
      3 columns, a width-1 object shifts by 1 column, and so on.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert objects, "expected at least one foreground object"

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)
    h, w = g.shape

    for obj in objects:
        coords = obj["coords"]
        shift = bbox_w(obj["bbox"])
        dst = coords + np.array([0, shift])

        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((0 <= dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool(occupied[dst[:, 0], dst[:, 1]].any())

        out[dst[:, 0], dst[:, 1]] = g[coords[:, 0], coords[:, 1]]
        occupied[dst[:, 0], dst[:, 1]] = True

    return out

# --------------- generator (o5x) ---------------
# Canvas/object bounds.
GRID_SIZE_RANGE = (5, 12)
OBJECT_COUNT_RANGE = (1, 2)
OBJECT_SIDE_RANGE = (1, 5)
ROW_BAND_GAP = 1
# Sprite appearance.
BACKGROUND_COLOR, OBJECT_COLOR = 0, 8
DENSITY_RANGE = (0.5, 1.0)
SPRITE_SYMMETRY = "not_symmetric"
SPRITE_CONNECTIVITY = 8
# Rejection limit.
MAX_GENERATION_ATTEMPTS = 300

def _gen_row_bands(n, sides):
    if len(sides) == 1:
        return [(random.randint(0, n - sides[0]), sides[0])]

    sides = list(sides)
    random.shuffle(sides)
    total = sum(sides)
    spare = n - total - ROW_BAND_GAP
    top = random.randint(0, spare)
    spare -= top
    middle_extra = random.randint(0, spare)
    r0 = top
    r1 = r0 + sides[0] + ROW_BAND_GAP + middle_extra
    return [(r0, sides[0]), (r1, sides[1])]


def _gen_attempt():
    n = random.randint(*GRID_SIZE_RANGE)
    max_side = min(OBJECT_SIDE_RANGE[1], (n - ROW_BAND_GAP) // 2)
    num_objects = random.randint(*OBJECT_COUNT_RANGE)
    sides = [random.randint(OBJECT_SIDE_RANGE[0], max_side) for _ in range(num_objects)]
    bands = _gen_row_bands(n, sides)

    inp = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)

    for r0, side in bands:
        density = random.uniform(*DENSITY_RANGE)
        sprite = random_sprite(
            side,
            side,
            density=density,
            symmetry=SPRITE_SYMMETRY,
            color_palette=[OBJECT_COLOR],
            connectivity=SPRITE_CONNECTIVITY,
            background=BACKGROUND_COLOR,
        )
        c_out = random.randint(side, n - side)
        paste(out, sprite, r0, c_out, transparent=BACKGROUND_COLOR)
        paste(inp, sprite, r0, c_out - side, transparent=BACKGROUND_COLOR)

    objects = find_components(inp, bg=get_bg(inp), connectivity=SPRITE_CONNECTIVITY, monochromatic=False)
    if len(objects) != num_objects or not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
