import numpy as np
from scipy.ndimage import binary_fill_holes

from arc_utils import Color, colors_present, find_components, get_bg, paste, random_free_location
import random


def _mask_in_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    filled = binary_fill_holes(mask).astype(bool)
    return bool(np.any(filled & ~mask))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every foreground object that encloses a background hole to teal.

    Input structure
    - The grid has one dominant background color and exactly one foreground color.
    - The foreground pixels form separate 4-connected objects.
    - Teal is absent from the input and is reserved as the output mark.

    Rule
    - Keep objects whose bbox contains no enclosed background region unchanged.
    - For each foreground object with a 4-connected background hole inside its bbox, recolor all of that object's pixels to teal.
    - Leave background pixels unchanged, including the holes themselves.

    Key insight
    - The changed objects are selected by topology, not by size or position: they are precisely the objects whose mask has an interior hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert Color.TEAL not in set(map(int, g.ravel())), "teal must be output-only"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    foreground_color = foreground_colors[0]

    out = g.copy()
    selected = 0
    components = find_components(g, bg=bg, colors=[foreground_color], connectivity=4, monochromatic=True)
    assert components, "expected at least one foreground object"

    for component in components:
        coords = component["coords"]
        obj_mask = _mask_in_bbox(coords, component["bbox"])
        if _has_enclosed_hole(obj_mask):
            out[coords[:, 0], coords[:, 1]] = Color.TEAL
            selected += 1

    assert selected > 0, "expected at least one hole-containing object"
    assert np.any(out != g), "expected a visible recolor"
    return out

# --------------- generator (o5x) ---------------

# Palette and grid bounds.
GEN_BG_COLOR, GEN_FG_COLOR = 9, 1
GRID_ROWS_RANGE, GRID_COLS_RANGE = (9, 16), (11, 15)

# Object count and foreground density constraints.
OBJECT_COUNT_RANGE, HOLE_OBJECT_COUNT_RANGE = (2, 7), (1, 3)
NOHOLE_OBJECT_MAX = 4
HOLE_OBJECT_MIN_AREA, OBJECT_MAX_AREA = 8, 17
FOREGROUND_DENSITY_RANGE = (0.18, 0.30)

# Local object mask construction constraints.
PATCH_CANVAS_SHAPE = (9, 7)
HOLE_BOX_ROWS_RANGE, HOLE_BOX_COLS_RANGE = (3, 9), (3, 7)
HOLE_SIZE_RANGE = (1, 8)
NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Optional one-pixel dent used only for max-area hole objects.
DENT_TARGET_AREA, DENT_PROBABILITY = 17, 0.35
DENT_BOXES, DENT_SIDES = ((4, 6), (6, 4)), ("top", "bottom", "left", "right")

# Placement and retry constraints.
PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 1, 4
GENERATION_RETRIES = 300

def _gen_crop_mask(mask):
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _gen_hole_size(mask):
    filled = binary_fill_holes(mask).astype(bool)
    return int(np.sum(filled & ~mask))


def _gen_neighbors(mask, r, c):
    h, w = mask.shape
    out = []
    for dr, dc in NEIGHBOR_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _gen_frontier(mask, forbidden=None):
    if forbidden is None:
        forbidden = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or forbidden[r, c]:
                continue
            if sum(bool(mask[rr, cc]) for rr, cc in _gen_neighbors(mask, r, c)) == 1:
                cells.append((r, c))
    return cells


def _gen_patch_from_mask(mask, bg, fg):
    cropped = _gen_crop_mask(mask)
    patch = np.full(cropped.shape, bg, dtype=int)
    patch[cropped] = fg
    return patch


def _gen_hole_patch(bg, fg, target_area):
    valid = []
    for h in range(HOLE_BOX_ROWS_RANGE[0], HOLE_BOX_ROWS_RANGE[1] + 1):
        for w in range(HOLE_BOX_COLS_RANGE[0], HOLE_BOX_COLS_RANGE[1] + 1):
            perim = 2 * h + 2 * w - 4
            hole = (h - 2) * (w - 2)
            if perim <= target_area and HOLE_SIZE_RANGE[0] <= hole <= HOLE_SIZE_RANGE[1]:
                valid.append((h, w))
    if not valid:
        return None

    make_dent = target_area == DENT_TARGET_AREA and random.random() < DENT_PROBABILITY
    if make_dent:
        h, w = random.choice(DENT_BOXES)
    else:
        h, w = random.choice(valid)

    canvas_h, canvas_w = PATCH_CANVAS_SHAPE
    mask = np.zeros(PATCH_CANVAS_SHAPE, dtype=bool)
    r0 = random.randint(0, canvas_h - h)
    c0 = random.randint(0, canvas_w - w)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True

    forbidden = np.zeros_like(mask, dtype=bool)
    forbidden[r0 + 1 : r1, c0 + 1 : c1] = True

    if make_dent:
        side = random.choice(DENT_SIDES)
        if side == "top":
            dent = (r0 + 1, random.randint(c0 + 1, c1 - 1))
        elif side == "bottom":
            dent = (r1 - 1, random.randint(c0 + 1, c1 - 1))
        elif side == "left":
            dent = (random.randint(r0 + 1, r1 - 1), c0 + 1)
        else:
            dent = (random.randint(r0 + 1, r1 - 1), c1 - 1)
        mask[dent] = True
        forbidden[dent] = False

    while int(mask.sum()) < target_area:
        cells = _gen_frontier(mask, forbidden)
        if not cells:
            return None
        r, c = random.choice(cells)
        mask[r, c] = True

    cropped = _gen_crop_mask(mask)
    if cropped.shape[0] > PATCH_CANVAS_SHAPE[0] or cropped.shape[1] > PATCH_CANVAS_SHAPE[1]:
        return None
    hole_size = _gen_hole_size(cropped)
    if int(cropped.sum()) != target_area or hole_size == 0:
        return None
    if not HOLE_SIZE_RANGE[0] <= hole_size <= HOLE_SIZE_RANGE[1]:
        return None
    return _gen_patch_from_mask(mask, bg, fg)


def _gen_nohole_patch(bg, fg, target_area):
    canvas_h, canvas_w = PATCH_CANVAS_SHAPE
    mask = np.zeros(PATCH_CANVAS_SHAPE, dtype=bool)
    mask[random.randrange(canvas_h), random.randrange(canvas_w)] = True

    while int(mask.sum()) < target_area:
        cells = _gen_frontier(mask)
        random.shuffle(cells)
        chosen = None
        for r, c in cells:
            trial = mask.copy()
            trial[r, c] = True
            cropped = _gen_crop_mask(trial)
            if cropped.shape[0] <= PATCH_CANVAS_SHAPE[0] and cropped.shape[1] <= PATCH_CANVAS_SHAPE[1]:
                chosen = (r, c)
                break
        if chosen is None:
            return None
        mask[chosen] = True

    cropped = _gen_crop_mask(mask)
    if int(cropped.sum()) != target_area or _gen_hole_size(cropped) > 0:
        return None
    return _gen_patch_from_mask(mask, bg, fg)


def _gen_allocate_areas(k_hole, k_nohole, total_area):
    mins = [HOLE_OBJECT_MIN_AREA] * k_hole + [1] * k_nohole
    maxs = [OBJECT_MAX_AREA] * (k_hole + k_nohole)
    areas = mins[:]
    remaining = total_area - sum(areas)
    if remaining < 0:
        return None
    while remaining:
        choices = [i for i, area in enumerate(areas) if area < maxs[i]]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, maxs[i] - areas[i]))
        areas[i] += add
        remaining -= add
    hole_areas = areas[:k_hole]
    nohole_areas = areas[k_hole:]
    random.shuffle(hole_areas)
    random.shuffle(nohole_areas)
    return hole_areas, nohole_areas


def _gen_object_counts():
    total_objects = random.randint(*OBJECT_COUNT_RANGE)
    k_hole_min = max(HOLE_OBJECT_COUNT_RANGE[0], total_objects - NOHOLE_OBJECT_MAX)
    k_hole_max = min(HOLE_OBJECT_COUNT_RANGE[1], total_objects - 1)
    k_hole = random.randint(k_hole_min, k_hole_max)
    k_nohole = total_objects - k_hole
    return total_objects, k_hole, k_nohole


def _gen_target_pixels(h, w, total_objects, k_hole, k_nohole):
    min_pixels = HOLE_OBJECT_MIN_AREA * k_hole + k_nohole
    lo = max(min_pixels, int(np.ceil(FOREGROUND_DENSITY_RANGE[0] * h * w)))
    hi = min(OBJECT_MAX_AREA * total_objects, int(np.floor(FOREGROUND_DENSITY_RANGE[1] * h * w)))
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_make_objects(bg, fg, hole_areas, nohole_areas):
    """Return shuffled (patch, has_hole) pairs, or None if any sampled patch is invalid."""
    objects = []
    for area in hole_areas:
        patch = _gen_hole_patch(bg, fg, area)
        if patch is None:
            return None
        objects.append((patch, True))
    for area in nohole_areas:
        patch = _gen_nohole_patch(bg, fg, area)
        if patch is None:
            return None
        objects.append((patch, False))

    random.shuffle(objects)
    return objects


def _gen_render_scene(h, w, bg, fg, total_objects, objects):
    grid = np.full((h, w), bg, dtype=int)
    expected = grid.copy()

    for patch, has_hole in sorted(objects, key=lambda item: int(np.sum(item[0] == fg)), reverse=True):
        try:
            r, c = random_free_location(grid, patch, transparent=bg, padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_CONNECTIVITY)
        except ValueError:
            return None
        paste(grid, patch, r, c, transparent=bg)
        if has_hole:
            out_patch = np.full(patch.shape, bg, dtype=int)
            out_patch[patch == fg] = Color.TEAL
            paste(expected, out_patch, r, c, transparent=bg)
        else:
            paste(expected, patch, r, c, transparent=bg)

    if get_bg(grid) != bg:
        return None
    comps = find_components(grid, bg=bg, colors=[fg], connectivity=4, monochromatic=True)
    if len(comps) != total_objects:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _gen_attempt():
    bg = GEN_BG_COLOR
    fg = GEN_FG_COLOR
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    total_objects, k_hole, k_nohole = _gen_object_counts()

    target_pixels = _gen_target_pixels(h, w, total_objects, k_hole, k_nohole)
    if target_pixels is None:
        return None
    allocated = _gen_allocate_areas(k_hole, k_nohole, target_pixels)
    if allocated is None:
        return None

    hole_areas, nohole_areas = allocated
    objects = _gen_make_objects(bg, fg, hole_areas, nohole_areas)
    if objects is None:
        return None

    return _gen_render_scene(h, w, bg, fg, total_objects, objects)


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b2862040 input")
