import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_TO_COLOR = {
    (0, 0): GREEN,   # top-left missing
    (0, 1): TEAL,    # top-right missing
    (1, 0): YELLOW,  # bottom-left missing
    (1, 1): PINK,    # bottom-right missing
}

BODY_PALETTE = (BLUE, RED, MAROON)
ORIENTATIONS = tuple(MARKER_TO_COLOR)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, bbox


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = _components_from_mask(grid == int(color))
    out = []
    for coords in comps:
        out.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _is_marker_component(comp: dict) -> bool:
    if comp["color"] != ORANGE or comp["area"] != 3:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _decode_marker(grid: np.ndarray, marker: dict) -> int:
    r0, c0, r1, c1 = marker["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape == (2, 2)
    missing = np.argwhere(patch == BLACK)
    assert missing.shape == (1, 2)
    return int(MARKER_TO_COLOR[tuple(map(int, missing[0]))])


def _enclosed_black_components(patch: np.ndarray) -> list[np.ndarray]:
    black = patch == BLACK
    h, w = patch.shape
    seen = np.zeros_like(black, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return _components_from_mask(black & ~seen)


def _extract_holes(grid: np.ndarray) -> list[dict]:
    holes = []
    for grey in _find_color_components(grid, GREY):
        r0, c0, r1, c1 = grey["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        for local in _enclosed_black_components(patch):
            global_coords = local.copy()
            global_coords[:, 0] += r0
            global_coords[:, 1] += c0
            mask, bbox = _coords_to_mask(global_coords)
            holes.append({"coords": global_coords, "bbox": bbox, "mask": mask})
    return holes


def _analyze_scene(grid: np.ndarray) -> dict:
    markers = [comp for comp in _find_color_components(grid, ORANGE) if _is_marker_component(comp)]

    body_components = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY, ORANGE):
            continue
        body_components.extend(_find_color_components(grid, color))

    color_counts = Counter(int(comp["color"]) for comp in body_components)
    assert color_counts, "expected colored sprites"
    ranked = color_counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "distractor color must be uniquely most common"
    distractor_color = int(ranked[0][0])

    holes = _extract_holes(grid)
    assert holes, "expected at least one hole"

    for hole in holes:
        key = (hole["mask"].shape, hole["mask"].astype(np.uint8).tobytes())
        hole["key"] = key
    assert len({hole["key"] for hole in holes}) == len(holes), "hole shapes must be unique"

    valid = []
    distractors = []
    used_markers = set()
    for comp in body_components:
        mask, bbox = _coords_to_mask(comp["coords"])
        comp["mask"] = mask
        comp["key"] = (mask.shape, mask.astype(np.uint8).tobytes())
        if int(comp["color"]) == distractor_color:
            distractors.append(comp)
            continue

        r0, c0, _r1, c1 = bbox
        candidates = []
        for idx, marker in enumerate(markers):
            mr0, mc0, mr1, mc1 = marker["bbox"]
            if mr1 + 1 == r0 and c0 <= mc0 and mc1 <= c1:
                candidates.append((idx, marker))
        assert len(candidates) == 1, "each valid block must have exactly one directly-above marker"
        idx, marker = candidates[0]
        assert idx not in used_markers, "marker reused by multiple blocks"
        used_markers.add(idx)
        valid.append(
            {
                **comp,
                "marker": marker,
                "target_color": _decode_marker(grid, marker),
            }
        )

    assert len(used_markers) == len(markers), "all markers must certify one valid block"
    assert len(valid) == len(holes), "holes and valid blocks must match 1:1"
    return {
        "markers": markers,
        "bodies": body_components,
        "valid": valid,
        "distractors": distractors,
        "holes": holes,
        "distractor_color": distractor_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove distractors, recolor each certified puzzle piece by its L-marker orientation, and embed it into the matching hole.

    Input structure:
    - Black background contains 2-3 grey rectangular objects with enclosed black holes.
    - Elsewhere, monochrome colored sprites are scattered; the uniquely most frequent non-grey sprite color is the distractor color.
    - Every non-distractor puzzle piece has one orange 2x2 L-triomino marker directly above its bounding box; distractors have no marker.

    Rule:
    - Ignore all sprites whose color is the most common non-grey sprite color.
    - For each remaining piece, read the missing corner of its directly-above orange 2x2 L marker: top-left -> green, top-right -> teal, bottom-left -> yellow, bottom-right -> pink.
    - Erase all orange markers and all original external sprites, then place each recolored valid piece into the unique grey-hole whose binary shape exactly matches the piece mask.

    Key insight:
    - A piece is identified only after combining two independent clues: color-frequency excludes distractors, and the surviving piece's orange L both certifies it and encodes its output color; placement still depends only on exact shape equality with a hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    scene = _analyze_scene(g)

    out = g.copy()
    out[g == ORANGE] = BLACK
    for comp in scene["bodies"]:
        rr, cc = comp["coords"][:, 0], comp["coords"][:, 1]
        out[rr, cc] = BLACK

    remaining_holes = {idx: hole for idx, hole in enumerate(scene["holes"])}
    for piece in scene["valid"]:
        matches = [idx for idx, hole in remaining_holes.items() if hole["key"] == piece["key"]]
        assert len(matches) == 1, "piece must match exactly one remaining hole"
        hole = remaining_holes.pop(matches[0])
        r0, c0, r1, c1 = hole["bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[hole["mask"]] = int(piece["target_color"])

    assert not remaining_holes
    return out


def _enumerate_hole_masks() -> list[np.ndarray]:
    masks = []
    for h, w in ((2, 2), (2, 3), (3, 2)):
            area = h * w
            for bits in range(1 << area):
                mask = np.array([(bits >> i) & 1 for i in range(area)], dtype=bool).reshape(h, w)
                count = int(mask.sum())
                if count < 3 or count > 6 or count == area:
                    continue
                coords = np.argwhere(mask)
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, h - 1, w - 1):
                    continue
                if len(_components_from_mask(mask)) != 1:
                    continue
                masks.append(mask.copy())
    random.shuffle(masks)
    return masks


HOLE_MASKS = _enumerate_hole_masks()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _render_body(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _render_valid_patch(mask: np.ndarray, color: int, missing_corner: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h + 2, w), dtype=int)
    patch[2:, :][mask] = int(color)
    marker_left = 0 if w == 2 else random.randint(0, w - 2)
    patch[0:2, marker_left : marker_left + 2] = ORANGE
    patch[missing_corner[0], marker_left + missing_corner[1]] = BLACK
    return patch


def _bbox_from_patch(patch: np.ndarray) -> tuple[int, int]:
    return patch.shape


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + h + pad)
    c1 = min(reserved.shape[1], left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    if top < 0 or left < 0 or top + h > reserved.shape[0] or left + w > reserved.shape[1]:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + h + pad)
    c1 = min(reserved.shape[1], left + w + pad)
    return not reserved[r0:r1, c0:c1].any()


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = canvas[top : top + h, left : left + w]
    m = patch != BLACK
    view[m] = patch[m]


def _choose_unique_masks(k: int) -> list[np.ndarray]:
    chosen = []
    used = set()
    pool = HOLE_MASKS.copy()
    random.shuffle(pool)
    for mask in pool:
        key = _mask_key(mask)
        if key in used:
            continue
        used.add(key)
        chosen.append(mask.copy())
        if len(chosen) == k:
            return chosen
    raise RuntimeError("not enough unique masks")


def _place_holes_in_grey(obj_h: int, obj_w: int, masks: list[np.ndarray]):
    for _ in range(200):
        taken = np.zeros((obj_h, obj_w), dtype=bool)
        placements = []
        ok = True
        for mask in masks:
            mh, mw = mask.shape
            positions = []
            for top in range(1, obj_h - mh):
                for left in range(1, obj_w - mw):
                    coords = np.argwhere(mask)
                    coords = coords + np.array([[top, left]])
                    conflict = False
                    for rr, cc in coords:
                        rr = int(rr)
                        cc = int(cc)
                        if taken[rr, cc]:
                            conflict = True
                            break
                        for nr, nc in _neighbors4(rr, cc):
                            if 0 <= nr < obj_h and 0 <= nc < obj_w and taken[nr, nc]:
                                conflict = True
                                break
                        if conflict:
                            break
                    if not conflict:
                        positions.append((top, left))
            if not positions:
                ok = False
                break
            top, left = random.choice(positions)
            placements.append((mask.copy(), top, left))
            taken[top : top + mh, left : left + mw] |= mask
        if ok:
            return placements
    return None


def _build_grey_patch(obj_h: int, obj_w: int, masks: list[np.ndarray]):
    placements = _place_holes_in_grey(obj_h, obj_w, masks)
    if placements is None:
        return None
    patch = np.full((obj_h, obj_w), GREY, dtype=int)
    holes = []
    for mask, top, left in placements:
        patch[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = BLACK
        holes.append({"mask": mask.copy(), "top": top, "left": left})
    return patch, holes


def _sample_hole_partition() -> list[int]:
    return random.choice(
        [
            [2, 2],
            [1, 1, 2],
            [1, 2, 2],
        ]
    )


def _build_expected(base: np.ndarray, hole_records: list[dict], target_colors: list[int]) -> np.ndarray:
    out = base.copy()
    for hole, color in zip(hole_records, target_colors):
        r0, c0 = hole["abs_top"], hole["abs_left"]
        h, w = hole["mask"].shape
        out[r0 : r0 + h, c0 : c0 + w][hole["mask"]] = int(color)
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        scene = _analyze_scene(grid)
    except Exception:
        return False

    if len(scene["valid"]) != meta["valid_count"]:
        return False
    if len(scene["holes"]) != meta["valid_count"]:
        return False
    if len(scene["markers"]) != meta["valid_count"]:
        return False
    if scene["distractor_color"] != meta["distractor_color"]:
        return False

    decoded = sorted(int(piece["target_color"]) for piece in scene["valid"])
    if decoded != sorted(meta["target_colors"]):
        return False

    hole_keys = sorted(hole["key"] for hole in scene["holes"])
    if hole_keys != sorted(meta["hole_keys"]):
        return False

    distractor_keys = {comp["key"] for comp in scene["distractors"]}
    if not any(key in distractor_keys for key in meta["hole_keys"]):
        return False

    if meta["valid_count"] >= 4 and set(meta["target_colors"]) != {GREEN, YELLOW, PINK, TEAL}:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(22, 29)
    cols = random.randint(22, 29)
    canvas = np.zeros((rows, cols), dtype=int)
    base = np.zeros_like(canvas)
    reserved = np.zeros_like(canvas, dtype=bool)

    hole_partition = _sample_hole_partition()
    total_valid = sum(hole_partition)
    hole_masks = _choose_unique_masks(total_valid)

    hole_records = []
    valid_masks = []
    cursor = 0
    for hole_count in hole_partition:
        if hole_count == 1:
            obj_h = random.randint(5, 7)
            obj_w = random.randint(6, 8)
        else:
            obj_h = random.randint(6, 7)
            obj_w = random.randint(7, 8)
        local_masks = [hole_masks[cursor + i] for i in range(hole_count)]
        cursor += hole_count
        built = _build_grey_patch(obj_h, obj_w, local_masks)
        if built is None:
            return None
        patch, local_holes = built

        for _ in range(200):
            top = random.randint(1, rows - obj_h - 1)
            left = random.randint(1, cols - obj_w - 1)
            if _can_place(reserved, top, left, obj_h, obj_w, pad=2):
                _paste(canvas, patch, top, left)
                _paste(base, patch, top, left)
                _reserve(reserved, top, left, obj_h, obj_w, pad=2)
                for hole in local_holes:
                    valid_masks.append(hole["mask"].copy())
                    hole_records.append(
                        {
                            "mask": hole["mask"].copy(),
                            "abs_top": top + hole["top"],
                            "abs_left": left + hole["left"],
                        }
                    )
                break
        else:
            return None

    distractor_color = random.choice(BODY_PALETTE)
    valid_colors = [color for color in BODY_PALETTE if color != distractor_color]
    if len(valid_colors) < 2:
        return None

    # Same original color can lead to different outputs; first four pieces cover all marker mappings.
    if total_valid == 4:
        missing_corners = list(random.sample(ORIENTATIONS, 4))
    else:
        missing_corners = list(random.sample(ORIENTATIONS, 4)) + [random.choice(ORIENTATIONS)]
    target_colors = [MARKER_TO_COLOR[corner] for corner in missing_corners]

    base_valid_colors = [valid_colors[0], valid_colors[0], valid_colors[1], valid_colors[1]]
    if total_valid > 4:
        base_valid_colors.append(random.choice(valid_colors))
    random.shuffle(base_valid_colors)

    valid_patches = []
    for mask, color, corner in zip(valid_masks, base_valid_colors, missing_corners):
        valid_patches.append(_render_valid_patch(mask, color, corner))

    distractor_count = max(Counter(base_valid_colors).values()) + random.randint(1, 2)
    distractor_masks = [random.choice(valid_masks).copy()]
    while len(distractor_masks) < distractor_count:
        distractor_masks.append(random.choice(HOLE_MASKS).copy())

    pieces = []
    for patch in valid_patches:
        pieces.append(patch)
    for mask in distractor_masks:
        pieces.append(_render_body(mask, distractor_color))
    random.shuffle(pieces)

    for patch in pieces:
        ph, pw = _bbox_from_patch(patch)
        placed = False
        for _ in range(300):
            top = random.randint(0, rows - ph)
            left = random.randint(0, cols - pw)
            if _can_place(reserved, top, left, ph, pw, pad=1):
                _paste(canvas, patch, top, left)
                _reserve(reserved, top, left, ph, pw, pad=1)
                placed = True
                break
        if not placed:
            return None

    expected = _build_expected(base, hole_records, target_colors)
    meta = {
        "valid_count": total_valid,
        "distractor_color": distractor_color,
        "target_colors": target_colors,
        "hole_keys": [_mask_key(mask) for mask in valid_masks],
    }
    if not _semantic_checks(canvas, expected, meta):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
