import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_TRIES = 200

# Four 3-cell L orientations in a 2x2 bbox.
L_SHAPES = (
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
)

# The matching cavity orientations inside a 3x3 grey slot.
SLOT_CAVITIES = (
    ((0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 1), (1, 2)),
    ((1, 0), (1, 1), (2, 1)),
    ((1, 1), (1, 2), (2, 1)),
)
ALL_3X3 = tuple((r, c) for r in range(3) for c in range(3))
SLOT_BORDERS = tuple(
    tuple(cell for cell in ALL_3X3 if cell not in cavity) for cavity in SLOT_CAVITIES
)
NORMALIZED_L_SHAPES = {tuple(sorted(shape)) for shape in L_SHAPES}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords):
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _is_l_shape(coords) -> bool:
    return len(coords) == 3 and _normalize(coords) in NORMALIZED_L_SHAPES


def _find_slots(grid: np.ndarray):
    h, w = grid.shape
    slots = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            for cavity, border in zip(SLOT_CAVITIES, SLOT_BORDERS):
                if any(int(patch[r, c]) != GREY for r, c in border):
                    continue
                cavity_values = [int(patch[r, c]) for r, c in cavity]
                marker_values = [v for v in cavity_values if v not in (BLACK, GREY)]
                if len(marker_values) != 1:
                    continue
                if cavity_values.count(BLACK) != 2:
                    continue
                slots.append(
                    {
                        "color": int(marker_values[0]),
                        "cavity": tuple((r0 + r, c0 + c) for r, c in cavity),
                    }
                )
                break
    colors = [slot["color"] for slot in slots]
    assert len(colors) == len(set(colors)), "slot marker colors must be unique"
    return slots


def _find_objects(grid: np.ndarray):
    objects = {}
    for value in np.unique(grid):
        color = int(value)
        if color in (BLACK, GREY):
            continue
        comps = _components(grid == color)
        l_components = [comp for comp in comps if _is_l_shape(comp)]
        if not l_components:
            continue
        assert len(l_components) == 1, "each color should have exactly one movable L object"
        objects[color] = tuple(sorted(l_components[0]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored L object into the grey slot whose hidden marker shares its color.

    Input structure:
    - The background is black.
    - Several grey 3x3 slot patterns appear in the grid; each keeps six grey border cells and
      leaves a 3-cell L-shaped cavity made of two black cells plus one hidden colored marker.
    - Separate monochrome 3-pixel L objects appear elsewhere, one per marker color.

    Rule:
    - Detect every grey slot and read the hidden marker color inside its cavity.
    - Match each slot to the unique standalone L object of the same color.
    - Erase each standalone L object back to black.
    - Fill the slot cavity with that color, which is equivalent to rotating the object until it fits.
    - Keep the grey slot borders and every unrelated pixel unchanged.

    Key insight:
    - The single colored pixel inside a grey slot is a pairing key, while the slot geometry itself
      determines the final orientation of the monochrome L that must be inserted there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    slots = _find_slots(g)
    objects = _find_objects(g)
    assert slots, "expected at least one grey slot"
    assert objects, "expected at least one movable L object"
    assert {slot["color"] for slot in slots} == set(objects), "slot colors and object colors must match"

    out = g.copy()
    for coords in objects.values():
        for r, c in coords:
            out[r, c] = BLACK

    for slot in slots:
        color = int(slot["color"])
        for r, c in slot["cavity"]:
            out[r, c] = color

    return out


def _balanced_orientations(rng: random.Random, count: int):
    values = []
    while len(values) < count:
        batch = list(range(4))
        rng.shuffle(batch)
        values.extend(batch)
    return values[:count]


def _reserve_ok(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not mask[r0:r1, c0:c1].any()


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1):
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    mask[r0:r1, c0:c1] = True


def _choose_position(
    rng: random.Random,
    reserved: np.ndarray,
    height: int,
    width: int,
    edge_margin: int = 1,
):
    h, w = reserved.shape
    row_start = edge_margin
    row_end = h - height - edge_margin
    col_start = edge_margin
    col_end = w - width - edge_margin
    if row_end < row_start or col_end < col_start:
        return None
    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(row_start, row_end)
        left = rng.randint(col_start, col_end)
        if _reserve_ok(reserved, top, left, height, width, pad=1):
            return top, left
    return None


def _draw_slot(grid: np.ndarray, top: int, left: int, orientation: int, color: int, marker_index: int):
    cavity = SLOT_CAVITIES[orientation]
    border = SLOT_BORDERS[orientation]
    for r, c in border:
        grid[top + r, left + c] = GREY
    marker_r, marker_c = cavity[marker_index]
    grid[top + marker_r, left + marker_c] = color
    return tuple((top + r, left + c) for r, c in cavity)


def _draw_object(grid: np.ndarray, top: int, left: int, orientation: int, color: int):
    coords = []
    for r, c in L_SHAPES[orientation]:
        rr = top + r
        cc = left + c
        grid[rr, cc] = color
        coords.append((rr, cc))
    return tuple(coords)


def _dimension_bounds(num_pairs: int):
    h_low = 10 + 2 * num_pairs
    h_high = min(30, 16 + 3 * num_pairs)
    w_low = 11 + 2 * num_pairs
    w_high = min(30, 18 + 3 * num_pairs)
    return h_low, h_high, w_low, w_high


def _gen_attempt(rng: random.Random):
    num_pairs = rng.choice((2, 3, 4, 5))
    h_low, h_high, w_low, w_high = _dimension_bounds(num_pairs)
    height = rng.randint(h_low, h_high)
    width = rng.randint(w_low, w_high)

    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    colors = rng.sample(list(COLOR_POOL), num_pairs)
    slot_orientations = _balanced_orientations(rng, num_pairs)
    object_orientations = _balanced_orientations(rng, num_pairs)

    for color, slot_orientation in zip(colors, slot_orientations):
        pos = _choose_position(rng, reserved, 3, 3, edge_margin=1)
        if pos is None:
            return None
        marker_index = rng.randrange(3)
        _draw_slot(grid, pos[0], pos[1], slot_orientation, color, marker_index)
        _reserve(reserved, pos[0], pos[1], 3, 3, pad=1)

    for color, object_orientation in zip(colors, object_orientations):
        pos = _choose_position(rng, reserved, 2, 2, edge_margin=1)
        if pos is None:
            return None
        _draw_object(grid, pos[0], pos[1], object_orientation, color)
        _reserve(reserved, pos[0], pos[1], 2, 2, pad=1)

    slots = _find_slots(grid)
    objects = _find_objects(grid)
    if len(slots) != num_pairs or len(objects) != num_pairs:
        return None
    if {slot["color"] for slot in slots} != set(objects):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
