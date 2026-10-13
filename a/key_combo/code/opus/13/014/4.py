import itertools
import random

import numpy as np


BG = 0
CUE_COLOR = 5
FILL_COLOR = 4
OBJECT_COLORS = [1, 2, 3, 6, 7, 8, 9]
AREA_FOR_ROTATION = {0: 4, 1: 5, 2: 6, 3: 3}

SHAPE_LIBRARY = {
    3: [
        ((0, 0), (0, 1), (1, 0)),
    ],
    4: [
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ],
    5: [
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ],
    6: [
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)),
    ],
}

MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords) -> tuple[int, int, int, int]:
    coords = list(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    coords = list(coords)
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_to_mask(shape) -> np.ndarray:
    _, _, r1, c1 = _bbox(shape)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _rotate_normalized(shape, k: int) -> tuple[tuple[int, int], ...]:
    mask = _shape_to_mask(shape)
    rot = np.rot90(mask, -int(k) % 4)
    return _normalize(map(tuple, np.argwhere(rot)))


def _rotate_absolute(coords, bbox, k: int) -> tuple[tuple[int, int], ...]:
    k = int(k) % 4
    if k == 0:
        return tuple(sorted(map(tuple, coords)))

    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    rotated = []
    for r, c in coords:
        dr = r - center_r
        dc = c - center_c
        for _ in range(k):
            dr, dc = dc, -dr
        rr = int(round(center_r + dr))
        cc = int(round(center_c + dc))
        rotated.append((rr, cc))
    assert len(set(rotated)) == len(rotated)
    return tuple(sorted(rotated))


def _find_components(grid: np.ndarray, allowed_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in allowed_colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = tuple(sorted(coords))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                    "norm": _normalize(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0], comp["color"]))
    return comps


def _valid_match(obj_norm, cue_norm) -> bool:
    return any(_rotate_normalized(obj_norm, k) == cue_norm for k in range(4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored object to a same-shape grey cue, decode a rotation from cue size, then rotate and complete the object.

    Input structure:
    - The background is black.
    - Several separated non-grey colored objects appear on the left.
    - Several separated grey cue sprites appear elsewhere; each cue matches exactly one object up to translation and quarter-turn rotation.

    Rule:
    - Extract each colored object's visible shape and each grey cue's visible shape, both translation-normalized.
    - Find the unique cue matched to each object by equality up to 0/90/180/270-degree rotation.
    - Rotate the colored object clockwise by `(matched cue area mod 4) * 90` around the center of its own bbox.
    - In the rotated object's new bbox, fill every remaining black cell with yellow, preserve the object's original color, and erase all grey cues to black.

    Key insight:
    - The cue's geometry identifies which object it belongs to, while the cue's pixel count encodes the rotation that determines where the yellow completion must go.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_components(g, set(OBJECT_COLORS))
    cues = _find_components(g, {CUE_COLOR})
    assert objects and len(objects) == len(cues)

    valid_perms = []
    for perm in itertools.permutations(range(len(cues))):
        if all(_valid_match(objects[i]["norm"], cues[perm[i]]["norm"]) for i in range(len(objects))):
            valid_perms.append(perm)
    assert len(valid_perms) == 1
    perm = valid_perms[0]

    out = g.copy()
    for comp in objects + cues:
        for r, c in comp["coords"]:
            out[r, c] = BG

    for obj_index, cue_index in enumerate(perm):
        obj = objects[obj_index]
        cue = cues[cue_index]
        rot_k = cue["area"] % 4
        rotated = _rotate_absolute(obj["coords"], obj["bbox"], rot_k)
        for r, c in rotated:
            out[r, c] = obj["color"]
        r0, c0, r1, c1 = _bbox(rotated)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r, c] == BG:
                    out[r, c] = FILL_COLOR

    return out


def _touch_or_overlap(a, b, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_record(rot_k: int, color: int) -> dict | None:
    area = AREA_FOR_ROTATION[rot_k]
    candidates = list(SHAPE_LIBRARY[area])
    random.shuffle(candidates)
    for base in candidates:
        if rot_k and _rotate_normalized(base, rot_k) == _normalize(base):
            continue
        obj_shape = _rotate_normalized(base, random.randrange(4))
        cue_shape = _rotate_normalized(base, random.randrange(4))
        return {
            "rot_k": rot_k,
            "color": color,
            "obj_shape": obj_shape,
            "cue_shape": cue_shape,
        }
    return None


def _place_shape(bboxes, shape, row_range, col_range, pad: int) -> tuple[int, int] | None:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    r_choices = list(range(row_range[0], row_range[1] - h + 2))
    c_choices = list(range(col_range[0], col_range[1] - w + 2))
    random.shuffle(r_choices)
    random.shuffle(c_choices)
    for r0 in r_choices:
        for c0 in c_choices:
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_touch_or_overlap(bbox, other, pad=pad) for other in bboxes):
                continue
            return r0, c0
    return None


def _stamp(grid: np.ndarray, shape, anchor, color: int) -> tuple[int, int, int, int]:
    r0, c0 = anchor
    coords = []
    for r, c in shape:
        rr = r0 + r
        cc = c0 + c
        grid[rr, cc] = color
        coords.append((rr, cc))
    return _bbox(coords)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    rot_subset = random.sample([0, 1, 2, 3], 3)
    colors = random.sample(OBJECT_COLORS, 3)
    records = []
    for rot_k, color in zip(rot_subset, colors):
        record = _sample_record(rot_k, color)
        if record is None:
            return None
        records.append(record)
    random.shuffle(records)

    divider = random.randint(w // 2 - 1, w // 2 + 1)
    left_boxes = []
    right_boxes = []

    for record in sorted(records, key=lambda rec: (-len(rec["obj_shape"]), rec["color"])):
        obj_anchor = _place_shape(left_boxes, record["obj_shape"], (0, h - 1), (1, max(2, divider - 4)), pad=1)
        if obj_anchor is None:
            return None
        obj_bbox = _stamp(grid, record["obj_shape"], obj_anchor, record["color"])
        left_boxes.append(obj_bbox)
        record["obj_bbox"] = obj_bbox

    for record in sorted(records, key=lambda rec: (-len(rec["cue_shape"]), rec["color"])):
        cue_anchor = _place_shape(
            right_boxes,
            record["cue_shape"],
            (0, h - 1),
            (min(w - 3, divider + 2), w - 2),
            pad=1,
        )
        if cue_anchor is None:
            return None
        cue_bbox = _stamp(grid, record["cue_shape"], cue_anchor, CUE_COLOR)
        right_boxes.append(cue_bbox)
        record["cue_bbox"] = cue_bbox

    if any(box[1] >= divider for box in left_boxes):
        return None
    if any(box[1] <= divider for box in right_boxes):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int((out == FILL_COLOR).sum()) < 3:
        return None
    if len(_find_components(grid, set(OBJECT_COLORS))) != 3 or len(_find_components(grid, {CUE_COLOR})) != 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
