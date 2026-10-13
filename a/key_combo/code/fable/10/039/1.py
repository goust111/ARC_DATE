from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
NON_BG = tuple(range(1, 10))
DICT_GLYPH_SIZE = 3
DICT_GAP_ROW = 1
DICT_ZONE_HEIGHT = 5
SCENE_SEPARATOR_ROWS = 1
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    norm = _normalize_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in norm.astype(int))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _detect_dictionary_rows(grid: np.ndarray) -> tuple[int, int]:
    nonempty_rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert nonempty_rows.size > 0, "missing foreground"
    label_row = int(nonempty_rows[-1])
    top = label_row - (DICT_ZONE_HEIGHT - 1)
    assert top >= 0, "dictionary does not fit"
    assert np.all(grid[label_row - 1] == BG), "expected blank row above labels"
    if top > 0:
        assert np.all(grid[top - 1] == BG), "dictionary must be isolated from the scene"
    return top, label_row


def _parse_dictionary(grid: np.ndarray) -> tuple[int, int, dict[tuple[tuple[int, ...], ...], int], list[int]]:
    top, label_row = _detect_dictionary_rows(grid)
    mapping: dict[tuple[tuple[int, ...], ...], int] = {}
    label_cols = [int(c) for c in np.flatnonzero(grid[label_row] != BG)]
    assert len(label_cols) >= 2, "expected multiple dictionary entries"
    label_colors = [int(grid[label_row, c]) for c in label_cols]
    assert len(set(label_colors)) == len(label_colors), "label colors must be unique"

    for c in label_cols:
        assert 0 < c < grid.shape[1] - 1, "label must sit under a 3x3 glyph"
        patch = grid[top : top + DICT_GLYPH_SIZE, c - 1 : c + 2]
        glyph_mask = patch != BG
        assert np.any(glyph_mask), "every label needs a glyph above it"
        key = _mask_key(glyph_mask)
        assert key not in mapping, "glyph masks must be distinct"
        mapping[key] = int(grid[label_row, c])

    return top, label_row, mapping, label_cols


def _rectangle_hole_key(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, ...], ...]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = {int(v) for v in patch.flatten() if int(v) != BG}
    assert len(colors) == 1, "rectangle must be monochrome apart from holes"
    color = next(iter(colors))
    assert np.all(patch[0, :] == color)
    assert np.all(patch[-1, :] == color)
    assert np.all(patch[:, 0] == color)
    assert np.all(patch[:, -1] == color)
    assert np.all((patch == BG) | (patch == color))
    hole_mask = patch == BG
    assert np.any(hole_mask), "rectangle must contain at least one hole"
    hole_box = _bbox(hole_mask)
    assert hole_box is not None
    hr0, hc0, hr1, hc1 = hole_box
    assert hr1 - hr0 + 1 <= 3 and hc1 - hc0 + 1 <= 3, "hole bbox must fit in 3x3"
    return _mask_key(hole_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each holed rectangle using the label below the matching dictionary glyph, then erase the dictionary.

    Input structure:
    - Black background contains several large monochrome rectangles whose interiors have a few black hole pixels.
    - Each rectangle is otherwise solid and axis-aligned; the holes lie strictly inside the rectangle and their minimal bounding box is at most 3x3.
    - A separated bottom dictionary strip shows several 3x3 glyph masks, each with a unique single-pixel label one row below it.

    Rule:
    - For every rectangle, collect its hole pixels and normalize them by cropping to the holes' minimal bounding box.
    - Find the dictionary glyph whose filled mask matches that normalized hole mask exactly, using translation only and no rotation or mirroring.
    - Recolor the whole rectangle with that glyph's label color and fill the holes.
    - Clear the entire dictionary strip, including glyphs, blank gap row, and labels, to background.

    Key insight:
    - The important pattern is the shape of the missing pixels inside each rectangle; the visible glyph colors are decoys, and the true output color is encoded only by the label under the matching glyph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    dict_top, label_row, mapping, _label_cols = _parse_dictionary(g)
    scene = g[: max(dict_top - SCENE_SEPARATOR_ROWS, 0), :]
    comps = _components(scene != BG)
    assert len(comps) >= 2, "expected multiple scene rectangles"

    out = g.copy()
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        key = _rectangle_hole_key(scene, coords)
        assert key in mapping, "every rectangle hole must match a dictionary glyph"
        out[r0 : r1 + 1, c0 : c1 + 1] = mapping[key]

    out[dict_top : label_row + 1, :] = BG
    return out


def _all_normalized_masks() -> list[np.ndarray]:
    unique: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << 9):
        raw = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        key = _mask_key(raw)
        unique.setdefault(key, _normalize_mask(raw))
    return [unique[key] for key in sorted(unique)]


def _hole_keeps_rectangle_connected(mask: np.ndarray) -> bool:
    h, w = mask.shape
    solid = np.ones((h + 2, w + 2), dtype=bool)
    solid[1 : 1 + h, 1 : 1 + w][mask] = False
    return len(_components(solid)) == 1


ALL_MASKS = [mask for mask in _all_normalized_masks() if _hole_keeps_rectangle_connected(mask)]
MASK_BY_KEY = {_mask_key(mask): mask.copy() for mask in ALL_MASKS}
MASKS_BY_SIGNATURE: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
ASYMMETRIC_MIRROR_PAIRS: list[tuple[np.ndarray, np.ndarray]] = []
for mask in ALL_MASKS:
    h, w = mask.shape
    signature = (h, w, int(mask.sum()))
    MASKS_BY_SIGNATURE[signature].append(mask)

mirror_pair_keys: set[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = set()
for mask in ALL_MASKS:
    flipped = np.fliplr(mask)
    if np.array_equal(mask, flipped):
        continue
    k1 = _mask_key(mask)
    k2 = _mask_key(flipped)
    if k1 == k2:
        continue
    pair_key = tuple(sorted((k1, k2)))
    if pair_key in mirror_pair_keys:
        continue
    mirror_pair_keys.add(pair_key)
    ASYMMETRIC_MIRROR_PAIRS.append((MASK_BY_KEY[pair_key[0]], MASK_BY_KEY[pair_key[1]]))


def _choose_masks() -> tuple[list[np.ndarray], np.ndarray]:
    pair_a, pair_b = random.choice(ASYMMETRIC_MIRROR_PAIRS)
    used_count = random.choice((2, 3))
    used = [pair_a.copy(), pair_b.copy()]
    signature = (pair_a.shape[0], pair_a.shape[1], int(pair_a.sum()))

    if used_count == 3:
        candidates = [
            mask
            for mask in MASKS_BY_SIGNATURE[signature]
            if _mask_key(mask) not in {_mask_key(pair_a), _mask_key(pair_b)}
        ]
        if not candidates:
            candidates = [mask for mask in ALL_MASKS if _mask_key(mask) not in {_mask_key(pair_a), _mask_key(pair_b)}]
        used.append(random.choice(candidates).copy())

    used_keys = {_mask_key(mask) for mask in used}
    same_signature_unused = [
        mask for mask in MASKS_BY_SIGNATURE[signature] if _mask_key(mask) not in used_keys
    ]
    if same_signature_unused:
        unused = same_signature_unused[0].copy()
    else:
        unused = next(mask.copy() for mask in ALL_MASKS if _mask_key(mask) not in used_keys)
    return used, unused


def _choose_color_sets(entry_count: int, rect_count: int) -> tuple[list[int], list[int], list[int]]:
    colors = list(NON_BG)
    random.shuffle(colors)
    label_colors = colors[:entry_count]
    rect_colors = colors[entry_count : entry_count + rect_count]
    assert len(rect_colors) == rect_count

    body_colors = []
    forced_collision_pool = label_colors + rect_colors
    for idx in range(entry_count):
        if idx == 0:
            body_colors.append(random.choice(forced_collision_pool))
        else:
            body_colors.append(random.choice(list(NON_BG)))
    return label_colors, rect_colors, body_colors


def _pick_rect_count(used_count: int) -> int:
    extra = 1 if used_count == 2 or random.random() < 0.65 else 0
    return min(4, used_count + extra)


def _rect_size_for_mask(mask: np.ndarray) -> tuple[int, int]:
    hole_h, hole_w = mask.shape
    min_h = max(5, hole_h + 2)
    min_w = max(5, hole_w + 2)
    height = random.randint(min_h, min(9, min_h + 3))
    width = random.randint(min_w, min(10, min_w + 4))
    return height, width


def _make_rect_specs(used_masks: list[np.ndarray], rect_count: int) -> list[dict]:
    specs = [{"mask": mask.copy()} for mask in used_masks]
    while len(specs) < rect_count:
        specs.append({"mask": random.choice(used_masks).copy()})
    random.shuffle(specs)

    repeated_seen = False
    for idx, spec in enumerate(specs):
        height, width = _rect_size_for_mask(spec["mask"])
        spec["height"] = height
        spec["width"] = width
        spec["index"] = idx
        key = _mask_key(spec["mask"])
        if sum(_mask_key(other["mask"]) == key for other in specs) > 1:
            repeated_seen = True
    assert repeated_seen or rect_count == len(used_masks)
    return specs


def _repeated_shapes_have_distinct_sizes(rect_specs: list[dict]) -> bool:
    grouped: dict[tuple[tuple[int, ...], ...], list[tuple[int, int]]] = defaultdict(list)
    for spec in rect_specs:
        grouped[_mask_key(spec["mask"])].append((spec["height"], spec["width"]))
    repeated_groups = [sizes for sizes in grouped.values() if len(sizes) >= 2]
    if not repeated_groups:
        return True
    return any(len(set(sizes)) >= 2 for sizes in repeated_groups)


def _place_rectangles(scene_h: int, width: int, rect_specs: list[dict]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(rect_specs)), key=lambda idx: rect_specs[idx]["height"] * rect_specs[idx]["width"], reverse=True)
    for idx in order:
        spec = rect_specs[idx]
        h = spec["height"]
        w = spec["width"]
        candidates = [(r, c) for r in range(scene_h - h + 1) for c in range(width - w + 1)]
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            ok = True
            for or0, oc0, or1, oc1 in boxes:
                if not (box[2] + 2 < or0 or or1 + 2 < box[0] or box[3] + 2 < oc0 or oc1 + 2 < box[1]):
                    ok = False
                    break
            if ok:
                placed = (r0, c0)
                break
        if placed is None:
            return None
        placements.append(placed)
        boxes.append((placed[0], placed[1], placed[0] + h - 1, placed[1] + w - 1))

    ordered = [None] * len(rect_specs)
    for order_idx, spec_idx in enumerate(order):
        ordered[spec_idx] = placements[order_idx]
    return ordered


def _carve_hole(grid: np.ndarray, r0: int, c0: int, height: int, width: int, mask: np.ndarray) -> tuple[int, int]:
    hole_h, hole_w = mask.shape
    hole_r = random.randint(r0 + 1, r0 + height - hole_h - 1)
    hole_c = random.randint(c0 + 1, c0 + width - hole_w - 1)
    patch = grid[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w]
    patch[mask] = BG
    return hole_r, hole_c


def _dictionary_layout(width: int, entry_count: int) -> list[int] | None:
    min_width = entry_count * 3 + (entry_count - 1)
    if min_width > width - 2:
        return None
    slack = width - 2 - min_width
    gap_slots = entry_count
    extras = [0] * gap_slots
    for _ in range(slack):
        extras[random.randrange(gap_slots)] += 1

    centers = []
    col = 1 + extras[0]
    for idx in range(entry_count):
        centers.append(col + 1)
        col += 3
        if idx < entry_count - 1:
            col += 1 + extras[idx + 1]
    return centers


def _stamp_glyph(slot: np.ndarray, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    slot[:h, :w][mask] = color


def _expected_from_specs(shape: tuple[int, int], rect_specs: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec, (r0, c0) in zip(rect_specs, placements):
        out[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = spec["label_color"]
    return out


def _validate_scene(grid: np.ndarray, scene_h: int, rect_specs: list[dict], placements: list[tuple[int, int]], mapping: dict[tuple[tuple[int, ...], ...], int]) -> None:
    scene = grid[:scene_h, :]
    comps = _components(scene != BG)
    assert len(comps) == len(rect_specs)
    seen_keys = []
    for coords in comps:
        key = _rectangle_hole_key(scene, coords)
        assert key in mapping
        seen_keys.append(key)
    used_keys = [_mask_key(spec["mask"]) for spec in rect_specs]
    assert sorted(seen_keys) == sorted(used_keys)

    if len(set(used_keys)) < len(rect_specs):
        grouped: dict[tuple[tuple[int, ...], ...], list[tuple[int, int]]] = defaultdict(list)
        for spec, pos in zip(rect_specs, placements):
            grouped[_mask_key(spec["mask"])].append((spec["height"], spec["width"]))
        repeated_groups = [sizes for sizes in grouped.values() if len(sizes) >= 2]
        assert repeated_groups
        assert any(len(set(sizes)) >= 2 for sizes in repeated_groups), "repeated hole shape should appear in different rectangle sizes"


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    used_masks, unused_mask = _choose_masks()
    rect_count = _pick_rect_count(len(used_masks))
    rect_specs = _make_rect_specs(used_masks, rect_count)
    if not _repeated_shapes_have_distinct_sizes(rect_specs):
        return None
    entry_masks = [mask.copy() for mask in used_masks] + [unused_mask.copy()]
    entry_count = len(entry_masks)

    label_colors, rect_colors, body_colors = _choose_color_sets(entry_count, rect_count)
    for idx, spec in enumerate(rect_specs):
        spec["rect_color"] = rect_colors[idx]

    entry_meta = []
    for idx, mask in enumerate(entry_masks):
        entry_meta.append(
            {
                "mask": mask.copy(),
                "label_color": label_colors[idx],
                "body_color": body_colors[idx],
                "key": _mask_key(mask),
            }
        )
    mapping = {entry["key"]: entry["label_color"] for entry in entry_meta}

    for spec in rect_specs:
        spec["label_color"] = mapping[_mask_key(spec["mask"])]

    width = random.randint(max(16, entry_count * 5 + 2), 30)
    scene_h = random.randint(9, 24)
    height = scene_h + SCENE_SEPARATOR_ROWS + DICT_ZONE_HEIGHT
    if height > 30:
        return None

    centers = _dictionary_layout(width, entry_count)
    if centers is None:
        return None
    placements = _place_rectangles(scene_h, width, rect_specs)
    if placements is None:
        return None

    dict_top = scene_h + SCENE_SEPARATOR_ROWS
    label_row = dict_top + DICT_ZONE_HEIGHT - 1
    inp = np.zeros((height, width), dtype=int)

    for spec, (r0, c0) in zip(rect_specs, placements):
        inp[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = spec["rect_color"]
        _carve_hole(inp, r0, c0, spec["height"], spec["width"], spec["mask"])

    for entry, center_c in zip(entry_meta, centers):
        patch = inp[dict_top : dict_top + 3, center_c - 1 : center_c + 2]
        _stamp_glyph(patch, entry["mask"], entry["body_color"])
        inp[label_row, center_c] = entry["label_color"]

    expected = _expected_from_specs((height, width), rect_specs, placements)

    solver_out = main(inp)
    if not np.array_equal(solver_out, expected):
        return None

    signature = (used_masks[0].shape[0], used_masks[0].shape[1], int(used_masks[0].sum()))
    dict_keys = [entry["key"] for entry in entry_meta]
    assert len(dict_keys) == len(set(dict_keys))
    assert any(
        (mask.shape[0], mask.shape[1], int(mask.sum())) == signature
        for mask in entry_masks[1:]
    ), "dictionary should include same-signature confusers"
    assert any(entry["body_color"] in set(label_colors + rect_colors) for entry in entry_meta), "need a decoy color collision"
    _validate_scene(inp, scene_h, rect_specs, placements, mapping)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _attempt_generate()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid hole-dictionary input")
