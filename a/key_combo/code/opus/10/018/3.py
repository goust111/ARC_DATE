import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS = ("down", "right", "left")
MARKER_PREFIX = (RED, GREEN, YELLOW)

_GEN_CALL_INDEX = 0


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords, bbox=None):
    if bbox is None:
        bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _tight_mask(mask):
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask):
    return tuple(tuple(int(cell) for cell in row) for row in mask.astype(int))


def _dihedral_variants(mask):
    base = _tight_mask(mask.astype(bool))
    variants = []
    seen = set()
    for flipped in (base, np.fliplr(base)):
        for k in range(4):
            variant = _tight_mask(np.rot90(flipped, k))
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _canonical_shape_key(mask):
    return min(_mask_key(variant) for variant in _dihedral_variants(mask))


def _find_color_components(grid, color):
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox(coords)
            mask = _coords_to_mask(coords, bbox)
            comps.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "r0": bbox[0],
                    "c0": bbox[1],
                    "r1": bbox[2],
                    "c1": bbox[3],
                    "signature": _canonical_shape_key(mask),
                }
            )
    return comps


def _read_markers(grid):
    marker_cols = [c for c in range(grid.shape[1]) if int(grid[0, c]) != BG]
    marker_colors = [int(grid[0, c]) for c in marker_cols]
    assert 2 <= len(marker_colors) <= 3, "expected two or three top-row markers"
    assert marker_colors == list(MARKER_PREFIX[: len(marker_colors)]), "markers must be the red/green/yellow prefix"
    assert np.count_nonzero(grid[0] != BG) == len(marker_colors), "top row should contain only the legend markers"
    return marker_colors, marker_cols


def _classify_objects(objects):
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj["signature"]].append(obj)

    classes = []
    for signature, members in grouped.items():
        sort_key = min((member["r0"], member["c0"]) for member in members)
        classes.append(
            {
                "signature": signature,
                "members": members,
                "sort_key": sort_key,
            }
        )
    classes.sort(key=lambda cls: cls["sort_key"])
    return classes


def _fits_mask(occupied, mask, r0, c0):
    h, w = mask.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    patch = occupied[r0 : r0 + h, c0 : c0 + w]
    return not np.any(patch & mask)


def _mark_mask(occupied, mask, r0, c0):
    h, w = mask.shape
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask


def _paint_mask(grid, mask, r0, c0, color):
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _sort_key_for_direction(obj, direction):
    if direction == "down":
        return (-obj["r1"], obj["r0"], obj["c0"])
    if direction == "right":
        return (-obj["c1"], obj["r0"], obj["c0"])
    return (obj["c0"], obj["r0"], obj["r1"])


def _boundary_destination(obj, shape):
    H, W = shape
    h, w = obj["mask"].shape
    if obj["direction"] == "down":
        return H - h, obj["c0"]
    if obj["direction"] == "right":
        return obj["r0"], W - w
    return obj["r0"], 0


def _slide_destination(obj, occupied, shape):
    H, W = shape
    h, w = obj["mask"].shape
    if obj["direction"] == "down":
        for nr in range(H - h, obj["r0"] - 1, -1):
            if _fits_mask(occupied, obj["mask"], nr, obj["c0"]):
                return nr, obj["c0"]
    elif obj["direction"] == "right":
        for nc in range(W - w, obj["c0"] - 1, -1):
            if _fits_mask(occupied, obj["mask"], obj["r0"], nc):
                return obj["r0"], nc
    else:
        for nc in range(0, obj["c0"] + 1):
            if _fits_mask(occupied, obj["mask"], obj["r0"], nc):
                return obj["r0"], nc
    raise AssertionError("no valid slide destination")


def _transform(grid, return_details=False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_colors, _marker_cols = _read_markers(g)
    objects = _find_color_components(g, BLUE)
    assert objects, "expected at least one blue object"

    classes = _classify_objects(objects)
    assert len(classes) == len(marker_colors), "legend length must match the number of dihedral classes"

    annotated = []
    class_details = []
    for class_index, cls in enumerate(classes):
        direction = DIRS[class_index]
        out_color = marker_colors[class_index]
        orientations = {_mask_key(member["mask"]) for member in cls["members"]}
        class_details.append(
            {
                "index": class_index,
                "direction": direction,
                "out_color": out_color,
                "sort_key": cls["sort_key"],
                "count": len(cls["members"]),
                "orientations": orientations,
            }
        )
        for member in cls["members"]:
            annotated.append(
                {
                    **member,
                    "class_index": class_index,
                    "direction": direction,
                    "out_color": out_color,
                }
            )

    out = np.full_like(g, BG)
    occupied = np.zeros_like(g, dtype=bool)
    placed = []

    for direction in DIRS[: len(marker_colors)]:
        group = [obj.copy() for obj in annotated if obj["direction"] == direction]
        group.sort(key=lambda obj: _sort_key_for_direction(obj, direction))
        for obj in group:
            dest_r0, dest_c0 = _slide_destination(obj, occupied, g.shape)
            _paint_mask(out, obj["mask"], dest_r0, dest_c0, obj["out_color"])
            _mark_mask(occupied, obj["mask"], dest_r0, dest_c0)
            obj["dest_r0"] = dest_r0
            obj["dest_c0"] = dest_c0
            placed.append(obj)

    if return_details:
        return out, {"marker_colors": marker_colors, "classes": class_details, "placed": placed}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group blue objects by dihedral-equivalent shape, then slide each class in the legend-defined direction.

    Input structure:
    - The background is black, the movable objects are blue, and the only other colored cells are 2-3 singleton legend pixels on the top row.
    - Each blue object is a separated 4-connected component whose tight binary mask may appear in different rotations or reflections.
    - The left-to-right legend length gives the number of shape classes, while the classes themselves are ordered by the smallest top-left bbox position among their members.

    Rule:
    - Extract every blue object, normalize its tight mask, and group objects that are equivalent under the 8 dihedral transforms.
    - Sort the resulting classes by the lexicographically smallest member bbox top-left; class 1 gets the first legend slot, class 2 the second, class 3 the third.
    - Map legend slot 1/2/3 to slide directions down/right/left, slide every object of that class rigidly toward its assigned boundary, and for objects sharing a direction place the ones already nearer that boundary first so later ones stop just before overlap.
    - Start from a blank black canvas and recolor the moved objects with their class legend color.

    Key insight:
    - Object orientation is a distractor: only dihedral shape equivalence reveals the true grouping, and that grouping simultaneously determines both motion direction and final color.
    """
    return _transform(grid)


def _mask_from_strings(rows):
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


BASE_SHAPES = {
    "L3": _mask_from_strings(["#.", "##"]),
    "L4": _mask_from_strings(["#..", "###"]),
    "T4": _mask_from_strings(["###", ".#."]),
    "S4": _mask_from_strings([".##", "##."]),
    "P5": _mask_from_strings(["##.", "###"]),
    "U5": _mask_from_strings(["#.#", "###"]),
}

SHAPE_COMBOS = {
    2: [
        ("L4", "T4"),
        ("L4", "S4"),
        ("T4", "S4"),
    ],
    3: [
        ("P5", "L4", "T4"),
        ("U5", "L4", "S4"),
        ("P5", "T4", "S4"),
    ],
}


def _choose_class_shapes(class_count):
    names = list(random.choice(SHAPE_COMBOS[class_count]))
    random.shuffle(names)
    return [BASE_SHAPES[name] for name in names]


def _choose_class_counts(class_count):
    if class_count == 2:
        return list(random.choice(((3, 3), (4, 3))))
    return list(random.choice(((2, 2, 2), (3, 2, 2))))


def _sample_oriented_masks(base_mask, count):
    variants = _dihedral_variants(base_mask)
    random.shuffle(variants)

    chosen = []
    if len(variants) >= 2 and count >= 2:
        chosen.append(variants[0].copy())
        chosen.append(variants[1].copy())
    while len(chosen) < count:
        chosen.append(random.choice(variants).copy())
    random.shuffle(chosen)
    return chosen


def _band_width(masks):
    return sum(mask.shape[1] for mask in masks) + max(0, len(masks) - 1) * 2


def _stack_height(masks):
    return sum(mask.shape[0] for mask in masks)


def _marker_columns(W, class_count):
    while True:
        cols = sorted(random.sample(range(1, W - 1), class_count))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)):
            return cols


def _can_place_input(grid, occupied, mask, r0, c0):
    h, w = mask.shape
    H, W = grid.shape
    if r0 < 1 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if r0 == 0:
        return False
    return _fits_mask(occupied, mask, r0, c0)


def _place_down_class(grid, occupied, masks, row_anchor, col_choices):
    placements = []
    next_row = row_anchor
    for mask in masks:
        h, _w = mask.shape
        row_options = list(range(next_row, min(next_row + 2, grid.shape[0] - h - 1) + 1))
        col_options = col_choices[:]
        random.shuffle(row_options)
        random.shuffle(col_options)
        picked = None
        for r0 in row_options:
            for c0 in col_options:
                if _can_place_input(grid, occupied, mask, r0, c0):
                    picked = (r0, c0)
                    break
            if picked is not None:
                break
        if picked is None:
            return None
        placements.append(picked)
        _mark_mask(occupied, mask, picked[0], picked[1])
        next_row = picked[0] + h + random.randint(1, 2)
    return placements


def _place_horizontal_class(grid, occupied, masks, row_anchor, start_c_min, start_c_max):
    placements = []
    current_c = random.randint(start_c_min, start_c_max)
    for mask in masks:
        h, w = mask.shape
        row_options = [row_anchor, row_anchor + 1]
        random.shuffle(row_options)
        picked = None
        for r0 in row_options:
            for c0 in range(current_c, min(current_c + 1, grid.shape[1] - w - 2) + 1):
                if _can_place_input(grid, occupied, mask, r0, c0):
                    picked = (r0, c0)
                    break
            if picked is not None:
                break
        if picked is None:
            return None
        placements.append(picked)
        _mark_mask(occupied, mask, picked[0], picked[1])
        current_c = picked[1] + w + random.randint(1, 2)
    return placements


def _build_attempt(class_count):
    class_masks = [_sample_oriented_masks(base, count) for base, count in zip(_choose_class_shapes(class_count), _choose_class_counts(class_count))]
    down_masks = class_masks[0]
    right_masks = class_masks[1]
    left_masks = class_masks[2] if class_count == 3 else []

    right_row = 5
    left_row = 9

    H_min = max(
        15,
        _stack_height(down_masks) + max(right_row + max(mask.shape[0] for mask in right_masks), (left_row + max(mask.shape[0] for mask in left_masks)) if left_masks else 0) + 3,
    )
    W_min = max(
        15,
        4 + max(mask.shape[1] for mask in down_masks) + max(_band_width(right_masks), _band_width(left_masks) if left_masks else 0) + 4,
    )
    H = random.randint(H_min, min(24, H_min + 3))
    W = random.randint(W_min, min(26, W_min + 3))

    grid = np.full((H, W), BG, dtype=int)
    marker_colors = list(MARKER_PREFIX[:class_count])
    for color, col in zip(marker_colors, _marker_columns(W, class_count)):
        grid[0, col] = color

    occupied = np.zeros((H, W), dtype=bool)

    down_c_base = random.randint(1, max(1, W // 5))
    down_col_choices = [c for c in (down_c_base, down_c_base + 1) if c < W - 1]
    down_places = _place_down_class(grid, occupied, down_masks, row_anchor=2, col_choices=down_col_choices)
    if down_places is None:
        return None

    down_right_edge = max(c0 + mask.shape[1] for (c0, mask) in zip([c for _, c in down_places], down_masks))
    right_total = _band_width(right_masks)
    right_start_min = max(down_right_edge + 2, W // 3)
    right_start_max = max(right_start_min, W - right_total - 3)
    right_places = _place_horizontal_class(grid, occupied, right_masks, right_row, right_start_min, right_start_max)
    if right_places is None:
        return None

    left_places = []
    if left_masks:
        left_total = _band_width(left_masks)
        left_start_min = max(down_right_edge + 2, W - left_total - 4)
        left_start_max = max(left_start_min, W - left_total - 2)
        left_places = _place_horizontal_class(grid, occupied, left_masks, left_row, left_start_min, left_start_max)
        if left_places is None:
            return None

    for masks, places in zip(class_masks, (down_places, right_places, left_places if left_masks else [])):
        if len(masks) != len(places):
            return None
        for mask, (r0, c0) in zip(masks, places):
            _paint_mask(grid, mask, r0, c0, BLUE)

    return grid


def _class_orientation_ok(details):
    for cls in details["classes"]:
        if cls["count"] >= 2 and len(cls["orientations"]) < 2:
            return False
    return True


def _collision_is_active(details, shape):
    for obj in details["placed"]:
        if (obj["dest_r0"], obj["dest_c0"]) != _boundary_destination(obj, shape):
            return True
    return False


def _validate_generated_grid(grid, class_count):
    if grid is None:
        return False
    try:
        out, details = _transform(grid, return_details=True)
    except AssertionError:
        return False
    if len(details["classes"]) != class_count:
        return False
    if len(details["placed"]) < 6 or len(details["placed"]) > 8:
        return False
    if any(cls["count"] < 2 for cls in details["classes"]):
        return False
    if not _class_orientation_ok(details):
        return False
    if not _collision_is_active(details, grid.shape):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX

    preferred_class_count = 2 if _GEN_CALL_INDEX % 2 == 0 else 3
    _GEN_CALL_INDEX += 1

    for _ in range(400):
        grid = _build_attempt(preferred_class_count)
        if _validate_generated_grid(grid, preferred_class_count):
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 3")
