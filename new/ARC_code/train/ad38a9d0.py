import numpy as np

from arc_utils import colors_present, find_components, get_bg


SHAPE_TO_COLOR = {
    ((0, 0), (0, 1), (1, 1)): 4,
    ((0, 0), (0, 1), (1, 1), (1, 2)): 8,
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)): 3,
    ((0, 0), (1, 0), (2, 0)): 2,
    ((0, 0), (1, 0)): 9,
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)): 5,
}


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(coords, dtype=int)
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each placeholder object according to its normalized shape while leaving the background fixed.

    Input structure
    A same-size grid with one dominant background color and one foreground
    placeholder color forming separated 4-connected objects.

    Rule
    Replace every placeholder object by the color assigned to its exact
    translation-normalized mask; all background pixels remain unchanged.

    Key insight
    The input objects are a shape legend with repeated instances: identical
    masks always receive identical colors, independent of their position.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1

    out = g.copy()
    comps = find_components(
        g,
        bg=bg,
        colors=[fg_colors[0]],
        connectivity=4,
        monochromatic=True,
    )
    assert comps
    for comp in comps:
        coords = comp["coords"]
        shape = _shape_signature(coords)
        assert shape in SHAPE_TO_COLOR
        out[tuple(coords.T)] = SHAPE_TO_COLOR[shape]
    return out

# --------------- generator (att1) ---------------

# Canvas and colors.
GRID_H = 9
GRID_W = 9
BG_COLOR = 7
FG_COLOR = 6

# Object count and retry limits.
MIN_OBJECTS = 6
MAX_OBJECTS_EXCLUSIVE = 9
GENERATION_RETRIES = 300
RESERVED_PADDING = 1

# Canonical shapes: (cells, height, width), in the same order used for duplicates.
SHAPES = [
    (((0, 0), (0, 1), (1, 1)), 2, 2),  # L-triomino -> 4
    (((0, 0), (0, 1), (1, 1), (1, 2)), 2, 3),  # S-tetromino -> 8
    (((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)), 3, 3),  # plus -> 3
    (((0, 0), (1, 0), (2, 0)), 3, 1),  # vertical-3 -> 2
    (((0, 0), (1, 0)), 2, 1),  # domino -> 9
    (((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)), 2, 3),  # 2x3 rect -> 5
]


def _choose_shape_indices(rng):
    """Return all six shape ids plus 0-2 duplicate ids, then shuffle them."""
    num_objects = rng.integers(MIN_OBJECTS, MAX_OBJECTS_EXCLUSIVE)
    shape_indices = list(range(len(SHAPES)))
    num_duplicates = num_objects - len(SHAPES)
    if num_duplicates > 0:
        shape_indices.extend(rng.integers(0, len(SHAPES), size=num_duplicates).tolist())
    rng.shuffle(shape_indices)
    return shape_indices


def _shuffled_positions(rng, shape_h, shape_w):
    positions = [
        (r, c)
        for r in range(GRID_H - shape_h + 1)
        for c in range(GRID_W - shape_w + 1)
    ]
    rng.shuffle(positions)
    return positions


def _reserve_bbox(occupied, r, c, shape_h, shape_w):
    r0 = max(0, r - RESERVED_PADDING)
    c0 = max(0, c - RESERVED_PADDING)
    r1 = min(GRID_H, r + shape_h + RESERVED_PADDING)
    c1 = min(GRID_W, c + shape_w + RESERVED_PADDING)
    occupied[r0:r1, c0:c1] = True


def _place_shape(grid, occupied, cells, shape_h, shape_w, rng):
    """Place one shape at the first free candidate in a randomized scan."""
    for r, c in _shuffled_positions(rng, shape_h, shape_w):
        if occupied[r:r + shape_h, c:c + shape_w].any():
            continue
        for dr, dc in cells:
            grid[r + dr, c + dc] = FG_COLOR
        _reserve_bbox(occupied, r, c, shape_h, shape_w)
        return True
    return False


def _gen_attempt():
    rng = np.random.default_rng()
    grid = np.full((GRID_H, GRID_W), BG_COLOR, dtype=int)
    occupied = np.zeros((GRID_H, GRID_W), dtype=bool)

    for idx in _choose_shape_indices(rng):
        cells, shape_h, shape_w = SHAPES[idx]
        if not _place_shape(grid, occupied, cells, shape_h, shape_w, rng):
            return None

    out = main(grid)
    if out is None:
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATION_RETRIES} attempts")
