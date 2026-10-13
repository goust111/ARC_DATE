import random
from collections import deque

import numpy as np


SYMMETRIC_SHAPES = [
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 1), (2, 2)],
]

ASYMMETRIC_SHAPES = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)],
]

MAX_RETRIES = 500


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components4(mask: np.ndarray):
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def normalize_shape(coords):
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return [tuple(map(int, cell)) for cell in arr[order]]


def shape_dims(coords):
    arr = np.array(coords, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def shape_mask(coords):
    h, w = shape_dims(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def is_vertically_symmetric(coords):
    mask = shape_mask(coords)
    return bool(np.array_equal(mask, np.fliplr(mask)))


def transform_shape(coords, rot, flip_h, flip_v):
    arr = np.array(coords, dtype=int)
    for _ in range(rot):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    if flip_h:
        arr[:, 1] *= -1
    if flip_v:
        arr[:, 0] *= -1
    return normalize_shape(arr)


def random_asymmetric_shape():
    for _ in range(64):
        base = random.choice(ASYMMETRIC_SHAPES)
        coords = transform_shape(
            base,
            rot=random.randrange(4),
            flip_h=bool(random.getrandbits(1)),
            flip_v=bool(random.getrandbits(1)),
        )
        if not is_vertically_symmetric(coords):
            return coords
    raise RuntimeError("failed to sample an asymmetric shape")


def reflect_coords(coords, axis_col):
    return [(int(r), int(2 * axis_col - c)) for r, c in coords]


def place_shape(grid, coords, top, left, color):
    for r, c in coords:
        grid[top + r, left + c] = color


def absolute_coords(coords, top, left):
    return [(int(top + r), int(left + c)) for r, c in coords]


def expanded_mask(mask: np.ndarray):
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def analyze_colors(grid: np.ndarray):
    colors = [int(v) for v in np.unique(grid) if int(v) != 0]
    info = []
    for color in colors:
        mask = grid == color
        box = bbox(mask)
        assert box is not None
        cropped = crop_bbox(mask.astype(np.uint8), box).astype(bool)
        comps = components4(mask)
        info.append(
            {
                "color": color,
                "mask": mask,
                "bbox": box,
                "cropped": cropped,
                "symmetric": bool(np.array_equal(cropped, np.fliplr(cropped))),
                "components": comps,
                "area": int(mask.sum()),
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect every non-anchor object across the vertical axis defined by the unique symmetric object.

    Input structure:
    - The grid has black background and 3 or 4 non-black monochromatic objects, one color per object.
    - Exactly one object's bbox-cropped shape is invariant under left-right reflection.
    - The remaining objects are 4-connected but not vertically symmetric.

    Rule:
    - For each non-black color, crop its occupied pixels to that color's bbox and test whether the crop equals its left-right mirror.
    - The unique passing color is the anchor; its vertical mirror axis is the center column of its bbox, using floor((left + right) / 2).
    - Erase every non-anchor pixel and redraw it at the reflected coordinate (r, 2 * axis_col - c), discarding any reflected pixel that leaves the grid.
    - Keep the anchor unchanged.

    Key insight:
    - The symmetric object is not moved; instead, its own geometry and position determine the global reflection axis used to move all other objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    info = analyze_colors(g)
    anchors = [item for item in info if item["symmetric"]]
    assert len(anchors) == 1
    anchor = anchors[0]
    axis_col = (anchor["bbox"][1] + anchor["bbox"][3]) // 2

    for item in info:
        if item["color"] == anchor["color"]:
            continue
        out[item["mask"]] = 0
        rows, cols = np.where(item["mask"])
        for r, c in zip(rows, cols):
            new_c = 2 * axis_col - int(c)
            if 0 <= new_c < g.shape[1]:
                out[int(r), new_c] = item["color"]
    return out


def _choose_anchor_shape():
    shape = normalize_shape(random.choice(SYMMETRIC_SHAPES))
    assert is_vertically_symmetric(shape)
    return shape


def _anchor_position(h, w, shape):
    sh, sw = shape_dims(shape)
    row_lo = max(1, h // 2 - 3)
    row_hi = min(h - sh - 1, h // 2 + 1)
    if row_lo > row_hi:
        row_lo, row_hi = 1, h - sh - 1
    top = random.randint(row_lo, row_hi)

    col_lo = max(2, w // 2 - sw - 1)
    col_hi = min(w - sw - 2, w // 2)
    if col_lo > col_hi:
        col_lo, col_hi = 1, w - sw - 1
    left = random.randint(col_lo, col_hi)
    return top, left


def _side_schedule(count):
    if count == 2:
        sides = ["left", "right"]
    else:
        sides = ["left", "right", random.choice(["left", "right"])]
    random.shuffle(sides)
    return sides


def _valid_candidate(grid, occupied, reserved, coords, axis_col, top, left):
    abs_in = absolute_coords(coords, top, left)
    abs_out = reflect_coords(abs_in, axis_col)
    h, w = grid.shape

    if len(set(abs_in)) != len(abs_in) or len(set(abs_out)) != len(abs_out):
        return None
    if set(abs_in) & set(abs_out):
        return None

    for r, c in abs_in + abs_out:
        if not (0 <= r < h and 0 <= c < w):
            return None

    padded_occupied = expanded_mask(occupied)
    padded_reserved = expanded_mask(reserved)
    for r, c in abs_in:
        if padded_occupied[r, c] or reserved[r, c]:
            return None
    for r, c in abs_out:
        if occupied[r, c] or padded_reserved[r, c]:
            return None
    return abs_in, abs_out


def _sample_position(grid, occupied, reserved, coords, axis_col, side):
    h, w = grid.shape
    sh, sw = shape_dims(coords)
    max_c = sw - 1

    if side == "left":
        c_lo = max(1, 2 * axis_col - (w - 2))
        c_hi = axis_col - max_c - 2
    else:
        c_lo = axis_col + 2
        c_hi = min(w - sw - 1, 2 * axis_col - max_c - 1)
    if c_lo > c_hi:
        return None

    rows = list(range(1, h - sh))
    cols = list(range(c_lo, c_hi + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    attempts = 0
    for top in rows:
        for left in cols:
            attempts += 1
            candidate = _valid_candidate(grid, occupied, reserved, coords, axis_col, top, left)
            if candidate is not None:
                return top, left, candidate
            if attempts >= 150:
                return None
    return None


def _generate_attempt():
    h = random.randint(14, 21)
    w = random.randint(15, 23)
    grid = np.zeros((h, w), dtype=int)

    n_objects = random.choice([3, 4])
    colors = random.sample(list(range(1, 10)), n_objects)

    anchor_shape = _choose_anchor_shape()
    anchor_top, anchor_left = _anchor_position(h, w, anchor_shape)
    place_shape(grid, anchor_shape, anchor_top, anchor_left, colors[0])

    occupied = np.zeros_like(grid, dtype=bool)
    reserved = np.zeros_like(grid, dtype=bool)
    for r, c in absolute_coords(anchor_shape, anchor_top, anchor_left):
        occupied[r, c] = True

    axis_col = anchor_left + (shape_dims(anchor_shape)[1] - 1) // 2
    mover_specs = []
    sides = _side_schedule(n_objects - 1)
    for side, color in zip(sides, colors[1:]):
        coords = random_asymmetric_shape()
        sampled = _sample_position(grid, occupied, reserved, coords, axis_col, side)
        if sampled is None:
            return None
        top, left, (abs_in, abs_out) = sampled
        for r, c in abs_in:
            occupied[r, c] = True
        for r, c in abs_out:
            reserved[r, c] = True
        place_shape(grid, coords, top, left, color)
        mover_specs.append((color, coords, top, left, side))

    if not mover_specs:
        return None
    if len({side for _, _, _, _, side in mover_specs}) < 2:
        return None
    return grid


def _validate_instance(grid: np.ndarray):
    info = analyze_colors(grid)
    if len(info) not in {3, 4}:
        return False
    if any(len(item["components"]) != 1 for item in info):
        return False
    if any(not (4 <= item["area"] <= 8) for item in info):
        return False

    anchors = [item for item in info if item["symmetric"]]
    if len(anchors) != 1:
        return False
    anchor = anchors[0]
    axis_col = (anchor["bbox"][1] + anchor["bbox"][3]) // 2

    movers = [item for item in info if item["color"] != anchor["color"]]
    left_seen = False
    right_seen = False
    target_mask = np.zeros_like(grid, dtype=bool)
    for item in movers:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(item["mask"])]
        reflected = reflect_coords(coords, axis_col)
        if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in reflected):
            return False
        if any(grid[r, c] != 0 for r, c in reflected):
            return False
        if set(coords) & set(reflected):
            return False
        if max(c for _, c in coords) < axis_col:
            left_seen = True
        if min(c for _, c in coords) > axis_col:
            right_seen = True
        for r, c in reflected:
            if target_mask[r, c]:
                return False
            target_mask[r, c] = True
    if not (left_seen and right_seen):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[anchor["mask"]], grid[anchor["mask"]]):
        return False

    for item in movers:
        if np.any(out[item["mask"]] != 0):
            return False
        for r, c in reflect_coords(np.argwhere(item["mask"]), axis_col):
            if out[r, c] != item["color"]:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _generate_attempt()
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
