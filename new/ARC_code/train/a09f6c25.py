from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_color(mask: np.ndarray) -> int:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    diagonal = bool(
        mask.shape[0] == mask.shape[1]
        and (np.array_equal(mask, mask.T) or np.array_equal(mask, np.fliplr(mask).T))
    )

    assert sum((horizontal, vertical, diagonal)) == 1, "object must have one symmetry category"
    if horizontal:
        return 1
    if vertical:
        return 3
    return 6


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each non-singleton red object by its symmetry axis and erase isolated red pixels.

    Input structure:
    - The grid has one dominant background color and red 4-connected foreground objects.
    - Every non-singleton red object has exactly one reflection symmetry category:
      horizontal, vertical, or diagonal.
    - Single red pixels are noise.

    Rule:
    - Start from an all-background output of the same shape.
    - Recolor horizontally symmetric red objects blue, vertically symmetric objects
      green, and diagonally symmetric objects pink.
    - Leave isolated red pixels as background.

    Key insight: The target color is not tied to object position or size; it is a
    label for the object's reflection symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert colors - {int(bg)} == {2}, "expected red as the only foreground color"

    out = np.full_like(g, int(bg))
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[2])
    assert objects, "expected at least one red object"
    assert sum(int(obj["area"]) for obj in objects) == int((g == 2).sum())

    for obj in objects:
        coords = obj["coords"]
        assert int(obj["color"]) == 2
        if int(obj["area"]) == 1:
            continue

        color = _symmetry_color(_component_mask(coords, obj["bbox"]))
        out[coords[:, 0], coords[:, 1]] = color

    return out

# --------------- generator (att2) ---------------

# Canvas and palette.
GRID_ROWS = (17, 28)
GRID_COLS = (19, 27)
BACKGROUND_COLORS = [0, 4, 5, 7, 8, 9]
FOREGROUND_COLOR = 2
OUTPUT_COLOR_BY_KIND = {"H": 1, "V": 3, "D": 6}

# Object count and size constraints.
OBJECT_COUNT_RANGE = (3, 7)
H_COUNT_OPTIONS = (2, 3)
V_COUNT_OPTIONS = (1, 2)
D_COUNT_OPTIONS = (0, 1, 2)
AREA_MIN = 6
AREA_MAX = 28
AREA_MIN_NUM = 2
AREA_MIN_DEN = 5
AREA_MAX_PERCENT = 73
AREA_PERCENT_DEN = 100

# Symmetric mask dimensions and growth retries.
HV_HEIGHT_RANGE = range(3, 11)
HV_WIDTH_RANGE = range(3, 12)
D_SIZE_RANGE = range(4, 9)
MASK_RETRIES = 30

# Placement, noise, and generator retries.
MULTI_MIN_L1 = 4
SINGLETON_COUNT_RANGE = (3, 9)
SINGLETON_MIN_L1_TO_MULTI = 3
SINGLETON_MIN_L1_TO_SINGLETON = 2
GENERATION_RETRIES = 300


def _gen_area_bounds(h, w):
    cells = h * w
    lower = (AREA_MIN_NUM * cells + AREA_MIN_DEN - 1) // AREA_MIN_DEN
    upper = (AREA_MAX_PERCENT * cells) // AREA_PERCENT_DEN
    return max(AREA_MIN, lower), min(AREA_MAX, upper)


def _gen_orbit(kind, shape, r, c):
    h, w = shape
    if kind == "H":
        cells = {(r, c), (h - 1 - r, c)}
    elif kind == "V":
        cells = {(r, c), (r, w - 1 - c)}
    else:
        cells = {(r, c), (c, r)}
    return tuple(sorted(cells))


def _gen_symmetry_label(mask):
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    diagonal = bool(
        mask.shape[0] == mask.shape[1]
        and (np.array_equal(mask, mask.T) or np.array_equal(mask, np.fliplr(mask).T))
    )
    if sum((horizontal, vertical, diagonal)) != 1:
        return None
    if horizontal:
        return "H"
    if vertical:
        return "V"
    return "D"


def _gen_initial_mask(kind, h, w):
    mask = np.zeros((h, w), dtype=bool)
    forbidden = set()

    if kind == "H":
        axis_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
        mask[axis_rows, :] = True
        choices = [c for c in range(w) if c != w - 1 - c]
        arm_col = random.choice(choices)
        mask[:, arm_col] = True
        mirror_col = w - 1 - arm_col
        forbidden.update(_gen_orbit("H", mask.shape, 0, mirror_col))
    elif kind == "V":
        axis_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
        mask[:, axis_cols] = True
        choices = [r for r in range(h) if r != h - 1 - r]
        arm_row = random.choice(choices)
        mask[arm_row, :] = True
        mirror_row = h - 1 - arm_row
        forbidden.update(_gen_orbit("V", mask.shape, mirror_row, 0))
    else:
        for i in range(h):
            mask[i, i] = True
        for i in range(h - 1):
            mask[i, i + 1] = True
            mask[i + 1, i] = True

    return mask, forbidden


def _gen_base_area(kind, h, w):
    if kind == "H":
        return (1 if h % 2 else 2) * w + h - (1 if h % 2 else 2)
    if kind == "V":
        return (1 if w % 2 else 2) * h + w - (1 if w % 2 else 2)
    return 3 * h - 2


def _gen_dim_options(kind):
    if kind == "D":
        candidates = [(n, n) for n in D_SIZE_RANGE]
    else:
        candidates = [(h, w) for h in HV_HEIGHT_RANGE for w in HV_WIDTH_RANGE]

    options = []
    for h, w in candidates:
        lower, upper = _gen_area_bounds(h, w)
        if _gen_base_area(kind, h, w) <= upper and lower <= upper:
            options.append((h, w))
    return options


def _gen_frontier_orbits(mask, kind, forbidden, limit):
    h, w = mask.shape
    seen = set()
    choices = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            orbit = _gen_orbit(kind, mask.shape, r, c)
            if orbit in seen or any(cell in forbidden for cell in orbit):
                continue
            seen.add(orbit)
            new_cells = [cell for cell in orbit if not mask[cell]]
            if not new_cells or len(new_cells) > limit:
                continue
            touches = False
            for rr, cc in new_cells:
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        touches = True
                        break
                if touches:
                    break
            if touches:
                choices.append(orbit)
    return choices


def _gen_add_orbit(mask, orbit):
    for r, c in orbit:
        mask[r, c] = True


def _gen_grow_mask(mask, kind, lower, target, upper, forbidden):
    while int(mask.sum()) < target:
        area = int(mask.sum())
        choices = _gen_frontier_orbits(mask, kind, forbidden, target - area)
        if not choices:
            break
        _gen_add_orbit(mask, random.choice(choices))

    while int(mask.sum()) < lower:
        area = int(mask.sum())
        choices = _gen_frontier_orbits(mask, kind, forbidden, upper - area)
        if not choices:
            return False
        _gen_add_orbit(mask, random.choice(choices))

    return lower <= int(mask.sum()) <= upper


def _gen_mask(kind):
    options = _gen_dim_options(kind)
    for _ in range(MASK_RETRIES):
        h, w = random.choice(options)
        mask, forbidden = _gen_initial_mask(kind, h, w)
        lower, upper = _gen_area_bounds(h, w)
        lower = max(lower, int(mask.sum()))
        if lower > upper:
            continue
        target = random.randint(lower, upper)
        if not _gen_grow_mask(mask, kind, lower, target, upper, forbidden):
            continue
        if _gen_symmetry_label(mask) == kind:
            return mask
    return None


def _gen_counts():
    combos = [
        (h, v, d)
        for h in H_COUNT_OPTIONS
        for v in V_COUNT_OPTIONS
        for d in D_COUNT_OPTIONS
        if OBJECT_COUNT_RANGE[0] <= h + v + d <= OBJECT_COUNT_RANGE[1]
    ]
    total = random.randint(*OBJECT_COUNT_RANGE)
    return random.choice([combo for combo in combos if sum(combo) == total])


def _gen_min_l1(a, b):
    if len(a) == 0 or len(b) == 0:
        return 999
    return int(np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2).min())


def _gen_valid_positions(shape, mask, placed):
    rows, cols = shape
    h, w = mask.shape
    local = np.argwhere(mask)
    positions = []
    for r0 in range(rows - h + 1):
        for c0 in range(cols - w + 1):
            coords = local + np.array([r0, c0])
            if placed is not None and _gen_min_l1(coords, placed) < MULTI_MIN_L1:
                continue
            positions.append((r0, c0))
    return positions


def _gen_place_patch(grid, mask, r0, c0, color, bg):
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    paste(grid, patch, r0, c0, transparent=bg)


def _gen_scatter_singletons(inp, multi_coords, count):
    rows, cols = inp.shape
    singles = []
    for _ in range(count):
        single_arr = np.array(singles, dtype=int) if singles else np.empty((0, 2), dtype=int)
        choices = []
        for r in range(rows):
            for c in range(cols):
                if inp[r, c] == FOREGROUND_COLOR:
                    continue
                cell = np.array([[r, c]])
                if _gen_min_l1(cell, multi_coords) < SINGLETON_MIN_L1_TO_MULTI:
                    continue
                if _gen_min_l1(cell, single_arr) < SINGLETON_MIN_L1_TO_SINGLETON:
                    continue
                choices.append((r, c))
        if not choices:
            return False
        r, c = random.choice(choices)
        inp[r, c] = FOREGROUND_COLOR
        singles.append((r, c))
    return True


def _gen_objects(h_count, v_count, d_count):
    """Build all object masks, returning None if a constrained mask fails."""
    objects = []
    for kind, count in (("H", h_count), ("V", v_count), ("D", d_count)):
        for _ in range(count):
            mask = _gen_mask(kind)
            if mask is None:
                return None
            objects.append((kind, mask))
    random.shuffle(objects)
    objects.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return objects


def _gen_place_objects(inp, out, objects, bg):
    """Place masks into input/output grids and return all multi-cell coords."""
    placed_parts = []
    for kind, mask in objects:
        placed = np.concatenate(placed_parts, axis=0) if placed_parts else None
        positions = _gen_valid_positions(inp.shape, mask, placed)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _gen_place_patch(inp, mask, r0, c0, FOREGROUND_COLOR, bg)
        _gen_place_patch(out, mask, r0, c0, OUTPUT_COLOR_BY_KIND[kind], bg)
        placed_parts.append(np.argwhere(mask) + np.array([r0, c0]))
    return np.concatenate(placed_parts, axis=0)


def _gen_attempt():
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    bg = random.choice(BACKGROUND_COLORS)
    h_count, v_count, d_count = _gen_counts()

    objects = _gen_objects(h_count, v_count, d_count)
    if objects is None:
        return None

    inp = np.full((rows, cols), bg, dtype=int)
    out = np.full((rows, cols), bg, dtype=int)
    multi_coords = _gen_place_objects(inp, out, objects, bg)
    if multi_coords is None:
        return None

    singleton_count = random.randint(*SINGLETON_COUNT_RANGE)
    if int(multi_coords.shape[0]) + singleton_count >= (rows * cols + 1) // 2:
        return None
    if not _gen_scatter_singletons(inp, multi_coords, singleton_count):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid a09f6c25 input")
