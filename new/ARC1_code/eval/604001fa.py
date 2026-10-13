from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, colors_present, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location
import itertools
import random


def _infer_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected one object color and one marker color"

    marker_colors: list[int] = []
    for color in colors:
        comps = find_components(g, bg=bg, colors=[color], connectivity=4)
        if comps and all(
            comp["area"] == 3 and bbox_h(comp["bbox"]) == 2 and bbox_w(comp["bbox"]) == 2
            for comp in comps
        ):
            marker_colors.append(int(color))

    assert len(marker_colors) == 1, "expected a unique 2x2 L-triomino marker color"
    marker_color = marker_colors[0]
    object_colors = [int(color) for color in colors if int(color) != marker_color]
    assert len(object_colors) == 1, "expected a unique object color"
    return object_colors[0], marker_color


def _color_for_marker(g: np.ndarray, marker: dict, bg: int) -> int:
    missing_to_color = {
        (0, 0): Color.GREEN,
        (1, 0): Color.YELLOW,
        (1, 1): Color.PINK,
        (0, 1): Color.TEAL,
    }
    patch = crop_bbox(g, marker["bbox"])
    assert patch.shape == (2, 2), "marker must occupy a 2x2 bounding box"
    missing = np.argwhere(patch == bg)
    assert missing.shape == (1, 2), "marker must be a 2x2 L with one missing corner"
    key = tuple(map(int, missing[0]))
    assert key in missing_to_color, "marker orientation must map to a recolor"
    return int(missing_to_color[key])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each orange L marker and recolor the blue object directly below it by the marker orientation.

    Input structure:
    - A background grid contains one color of 8-connected foreground objects.
    - A second foreground color forms one 2x2 L-triomino marker above each object.
    - Each marker's bottom row is immediately above one object's bounding box, with overlapping columns.

    Rule:
    - Remove all marker pixels to the background.
    - For each 8-connected object, find the unique marker directly above it.
    - Recolor the object by the missing corner of the marker's 2x2 box: top-left -> green, bottom-left -> yellow, bottom-right -> pink, top-right -> teal.

    Key insight:
    The small L marker is an orientation code for the object's new color; the object's own shape is preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    object_color, marker_color = _infer_colors(g, bg)

    markers = find_components(g, bg=bg, colors=[marker_color], connectivity=4)
    objects = find_components(g, bg=bg, colors=[object_color], connectivity=8)
    assert markers and len(markers) == len(objects), "expected one marker for each object"

    out = g.copy()
    out[g == marker_color] = bg

    used_marker_bboxes: set[tuple[int, int, int, int]] = set()
    for obj in objects:
        or0, oc0, _or1, oc1 = map(int, obj["bbox"])
        candidates = []
        for marker in markers:
            _mr0, mc0, mr1, mc1 = map(int, marker["bbox"])
            if mr1 + 1 == or0 and max(mc0, oc0) <= min(mc1, oc1):
                candidates.append(marker)

        assert len(candidates) == 1, "each object must have one directly-above marker"
        marker = candidates[0]
        used_marker_bboxes.add(tuple(map(int, marker["bbox"])))
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = _color_for_marker(g, marker, bg)

    assert len(used_marker_bboxes) == len(markers), "all markers must be assigned"
    return out

# --------------- generator (o5x) ---------------

# Palette and marker orientation codes.
_GEN_BG = 0
_GEN_OBJECT_COLOR = 1
_GEN_MARKER_COLOR = 7
_GEN_REQUIRED_INPUT_COLORS = {_GEN_BG, _GEN_OBJECT_COLOR, _GEN_MARKER_COLOR}
_GEN_MISSING_TO_COLOR = {(0, 0): 3, (1, 0): 4, (1, 1): 6, (0, 1): 8}
_GEN_MISSING_CELLS = tuple(_GEN_MISSING_TO_COLOR)
_GEN_MARKER_SHAPE = (2, 2)
_GEN_L_TRIOMINO_AREA = 3

# Object mask size, area, and disconnected-4c weighting.
_GEN_OBJECT_DIMS = ((3, 4), (3, 3), (4, 4))
_GEN_OBJECT_DIM_WEIGHTS = (8, 1, 1)
_GEN_AREA_RANGE_BY_DIMS = {(3, 3): (6, 6), (3, 4): (7, 9), (4, 4): (9, 10)}
_GEN_C4_WEIGHTS = {2: 5, 3: 5, 4: 2}
_GEN_OTHER_C4_WEIGHT = 1

# Group assembly offsets: 2x2 marker directly above the object.
_GEN_MARKER_COL_OFFSET = 1
_GEN_OBJECT_TOP = 2

# Canvas, layout, and placement spacing.
_GEN_GROUP_COUNT_RANGE = (1, 4)
_GEN_GRID_MAX_SIZE = (20, 16)
_GEN_MIN_SIZE_BY_LAYOUT = {"one": (8, 6), "two_tall": (11, 6), "two_wide": (8, 9), "many": (12, 10)}
_GEN_TWO_GROUP_TALL_PROB = 0.5
_GEN_GROUP_PADDING = 1
_GEN_PADDING_CONNECTIVITY = 8

# Retry limit and exhaustive object-mask memoization.
_GEN_MAX_ATTEMPTS = 300
_GEN_MASK_CACHE = {}


def _gen_component_is_l_triomino(coords: np.ndarray) -> bool:
    if coords.shape[0] != _GEN_L_TRIOMINO_AREA:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)) == _GEN_MARKER_SHAPE


def _gen_valid_object_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, h - 1, w - 1):
        return False

    if len(components_mask(mask, connectivity=8)) != 1:
        return False
    comps4 = components_mask(mask, connectivity=4)
    if len(comps4) <= 1:
        return False
    return not all(_gen_component_is_l_triomino(comp) for comp in comps4)


def _gen_object_target_area(h: int, w: int) -> int:
    lo, hi = _GEN_AREA_RANGE_BY_DIMS[(h, w)]
    return lo if lo == hi else random.randint(lo, hi)


def _gen_valid_masks(h: int, w: int, target_area: int) -> list[tuple[np.ndarray, int]]:
    key = (h, w, target_area)
    if key in _GEN_MASK_CACHE:
        return _GEN_MASK_CACHE[key]

    valid = []
    for cells in itertools.combinations(range(h * w), target_area):
        mask = np.zeros(h * w, dtype=bool)
        mask[list(cells)] = True
        mask = mask.reshape((h, w))
        if _gen_valid_object_mask(mask):
            valid.append((mask, len(components_mask(mask, connectivity=4))))

    _GEN_MASK_CACHE[key] = valid
    return valid


def _gen_object_patch() -> np.ndarray | None:
    h, w = random.choices(_GEN_OBJECT_DIMS, weights=_GEN_OBJECT_DIM_WEIGHTS, k=1)[0]
    target_area = _gen_object_target_area(h, w)
    valid_masks = _gen_valid_masks(h, w, target_area)
    if not valid_masks:
        return None
    masks, c4_counts = zip(*valid_masks)
    weights = [_GEN_C4_WEIGHTS.get(c4, _GEN_OTHER_C4_WEIGHT) for c4 in c4_counts]
    mask = random.choices(masks, weights=weights, k=1)[0]
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = _GEN_OBJECT_COLOR
    return patch


def _gen_marker_patch(missing: tuple[int, int]) -> np.ndarray:
    patch = np.full(_GEN_MARKER_SHAPE, _GEN_MARKER_COLOR, dtype=int)
    patch[missing] = _GEN_BG
    return patch


def _gen_group_patches() -> tuple[np.ndarray, np.ndarray] | None:
    """Return aligned input/output patches for one marker-object group."""
    obj = _gen_object_patch()
    if obj is None:
        return None

    missing = random.choice(_GEN_MISSING_CELLS)
    marker = _gen_marker_patch(missing)
    h, w = obj.shape
    inp = np.zeros((h + _GEN_OBJECT_TOP, w), dtype=int)
    out = np.zeros_like(inp)

    paste(inp, marker, 0, _GEN_MARKER_COL_OFFSET, transparent=_GEN_BG)
    paste(inp, obj, _GEN_OBJECT_TOP, 0, transparent=_GEN_BG)

    recolored = np.zeros_like(obj)
    recolored[obj != _GEN_BG] = _GEN_MISSING_TO_COLOR[missing]
    paste(out, recolored, _GEN_OBJECT_TOP, 0, transparent=_GEN_BG)
    return inp, out


def _gen_grid_size(n_groups: int) -> tuple[int, int]:
    if n_groups == 1:
        min_rows, min_cols = _GEN_MIN_SIZE_BY_LAYOUT["one"]
    elif n_groups == 2 and random.random() < _GEN_TWO_GROUP_TALL_PROB:
        min_rows, min_cols = _GEN_MIN_SIZE_BY_LAYOUT["two_tall"]
    elif n_groups == 2:
        min_rows, min_cols = _GEN_MIN_SIZE_BY_LAYOUT["two_wide"]
    else:
        min_rows, min_cols = _GEN_MIN_SIZE_BY_LAYOUT["many"]
    max_rows, max_cols = _GEN_GRID_MAX_SIZE
    return random.randint(min_rows, max_rows), random.randint(min_cols, max_cols)


def _gen_place_group(grid, expected, reserved, inp_patch, out_patch) -> bool:
    """Place one group and update the input, expected output, and reservation masks."""
    reserve_patch = np.ones(inp_patch.shape, dtype=int)
    try:
        r0, c0 = random_free_location(
            reserved,
            reserve_patch,
            transparent=_GEN_BG,
            padding=_GEN_GROUP_PADDING,
            padding_connectivity=_GEN_PADDING_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(grid, inp_patch, r0, c0, transparent=_GEN_BG)
    paste(expected, out_patch, r0, c0, transparent=_GEN_BG)
    paste(reserved, reserve_patch, r0, c0, transparent=_GEN_BG)
    return True


def _gen_matches_solver(grid: np.ndarray, expected: np.ndarray) -> bool:
    """Check that generated inputs exercise the integrated solver oracle exactly."""
    if set(map(int, np.unique(grid))) != _GEN_REQUIRED_INPUT_COLORS:
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.randint(*_GEN_GROUP_COUNT_RANGE)
    rows, cols = _gen_grid_size(n_groups)
    grid = np.zeros((rows, cols), dtype=int)
    expected = np.zeros_like(grid)
    reserved = np.zeros_like(grid)

    for _ in range(n_groups):
        patches = _gen_group_patches()
        if patches is None:
            return None
        inp_patch, out_patch = patches
        if not _gen_place_group(grid, expected, reserved, inp_patch, out_patch):
            return None

    if not _gen_matches_solver(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 604001fa input")
