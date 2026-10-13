import numpy as np
from itertools import combinations


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6
TEAL = 8

SPRITE_COLORS = (BLUE, RED, GREEN, PINK)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _bbox_from_coords(coords):
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid, bbox):
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _nonzero_bbox(grid):
    coords = np.argwhere(grid != BLACK)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _neighbors4(cell):
    r, c = cell
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _normalize_shape(coords):
    arr = np.array(sorted((int(r), int(c)) for r, c in coords), dtype=int)
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_set(shape):
    return {(int(r), int(c)) for r, c in shape}


def _shape_area(shape):
    return len(shape)


def _is_connected(coords):
    coord_set = {tuple(map(int, cell)) for cell in coords}
    start = next(iter(coord_set))
    stack = [start]
    seen = {start}
    while stack:
        cell = stack.pop()
        for nxt in _neighbors4(cell):
            if nxt in coord_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coord_set)


def _find_components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _anchor_candidates(shape):
    shape_cells = _shape_set(shape)
    frontier = set()
    for cell in shape_cells:
        for nbr in _neighbors4(cell):
            if nbr not in shape_cells:
                frontier.add(nbr)
    return sorted(frontier)


def _relative_shape(shape, anchor):
    ar, ac = anchor
    return {(int(r - ar), int(c - ac)) for r, c in shape}


def _sprite_from_shape(shape, color, anchor):
    all_cells = list(shape) + [anchor]
    r0, c0, r1, c1 = _bbox_from_coords(all_cells)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in shape:
        sprite[r - r0, c - c0] = int(color)
    ar, ac = anchor
    sprite[ar - r0, ac - c0] = GREY
    return sprite


def _shape_patch(shape, color):
    r0, c0, r1, c1 = _bbox_from_coords(shape)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in shape:
        patch[r - r0, c - c0] = int(color)
    return patch


def _random_connected_shape(rng, area):
    coords = {(0, 0)}
    while len(coords) < area:
        frontier = sorted(
            nbr
            for cell in coords
            for nbr in _neighbors4(cell)
            if nbr not in coords
        )
        coords.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize_shape(coords)


def _mutate_shape(shape, rng):
    original = _shape_set(shape)
    frontier = sorted(
        nbr
        for cell in original
        for nbr in _neighbors4(cell)
        if nbr not in original
    )
    removable = list(original)
    rng.shuffle(removable)
    frontier_arr = list(frontier)
    rng.shuffle(frontier_arr)

    for remove_cell in removable:
        remaining = original - {remove_cell}
        if not _is_connected(remaining):
            continue
        for add_cell in frontier_arr:
            if add_cell in remaining:
                continue
            candidate = remaining | {add_cell}
            normalized = _normalize_shape(candidate)
            if normalized != shape and _is_connected(candidate):
                return normalized

    target_areas = [len(shape), max(3, len(shape) - 1), min(7, len(shape) + 1)]
    for target_area in target_areas:
        for _ in range(40):
            candidate = _random_connected_shape(rng, target_area)
            if candidate != shape:
                return candidate
    raise RuntimeError("failed to mutate shape")


def _classify_components(grid):
    components = _find_components(np.asarray(grid) != BLACK)
    templates = []
    sprites = []
    for coords in components:
        values = {int(grid[r, c]) for r, c in coords}
        bbox = _bbox_from_coords(coords)
        if values == {TEAL}:
            shape = _normalize_shape(coords)
            templates.append({"bbox": bbox, "shape": shape, "coords": coords})
            continue

        assert GREY in values and TEAL not in values, "expected anchored colored sprites and one teal template"
        fg_values = values - {GREY}
        assert len(fg_values) == 1, "sprite must have exactly one non-grey color"
        fg_color = next(iter(fg_values))
        grey_cells = [(r, c) for r, c in coords if int(grid[r, c]) == GREY]
        assert len(grey_cells) == 1, "sprite must have exactly one grey anchor"
        anchor_abs = grey_cells[0]
        shape_cells = [(r, c) for r, c in coords if int(grid[r, c]) != GREY]
        assert 3 <= len(shape_cells) <= 7, "sprite body size must be 3..7"
        assert any(
            (anchor_abs[0] + dr, anchor_abs[1] + dc) in set(shape_cells)
            for dr, dc in DIR4
        ), "grey anchor must touch the body in 4-neighborhood"
        shape = _normalize_shape(shape_cells)
        crop = _crop_bbox(grid, bbox)
        local_anchor = (anchor_abs[0] - bbox[0], anchor_abs[1] - bbox[1])
        rel_pixels = []
        for rr in range(crop.shape[0]):
            for cc in range(crop.shape[1]):
                color = int(crop[rr, cc])
                if color != BLACK:
                    rel_pixels.append((rr - local_anchor[0], cc - local_anchor[1], color))
        rel_shape = {
            (rr - local_anchor[0], cc - local_anchor[1])
            for rr in range(crop.shape[0])
            for cc in range(crop.shape[1])
            if int(crop[rr, cc]) == fg_color
        }
        sprites.append(
            {
                "bbox": bbox,
                "color": int(fg_color),
                "shape": shape,
                "crop": crop,
                "anchor_abs": anchor_abs,
                "anchor_local": local_anchor,
                "rel_pixels": rel_pixels,
                "rel_shape": rel_shape,
            }
        )

    assert len(templates) == 1, "expected exactly one teal template"
    template = templates[0]
    assert 3 <= _shape_area(template["shape"]) <= 7, "template size must be 3..7"
    return template, sprites


def _sort_sprites(sprites):
    return sorted(sprites, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _overlay_sprites(sprites):
    assert sprites, "need at least one sprite to overlay"
    translated = [item["rel_pixels"] for item in sprites]
    min_r = min(rr for pixels in translated for rr, _cc, _color in pixels)
    max_r = max(rr for pixels in translated for rr, _cc, _color in pixels)
    min_c = min(cc for pixels in translated for _rr, cc, _color in pixels)
    max_c = max(cc for pixels in translated for _rr, cc, _color in pixels)

    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    origin_r = -min_r
    origin_c = -min_c
    for item in sprites:
        for rr, cc, color in item["rel_pixels"]:
            out[origin_r + rr, origin_c + cc] = int(color)
    out[origin_r, origin_c] = BLACK

    box = _nonzero_bbox(out)
    assert box is not None, "overlay must leave colored pixels after erasing anchor"
    return _crop_bbox(out, box)


def _matched_sprites(template, sprites):
    template_shape = template["shape"]
    matched = [item for item in sprites if item["shape"] == template_shape]
    unmatched = [item for item in sprites if item["shape"] != template_shape]
    return matched, unmatched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter sprites by template-shape match, align the survivors by their grey anchors, and crop the composite.

    Input structure:
    - A black background contains one cyan template component and 5-7 separate anchored sprite components.
    - Each sprite has exactly one grey anchor pixel and one other color from blue, red, green, or magenta.
    - A sprite body is a 4-connected 3-7 pixel shape; its grey anchor is 4-neighbor adjacent to that body.

    Rule:
    - Extract the cyan template shape as translation-normalized body coordinates.
    - For each anchored sprite, compare only its non-grey body shape after the same normalization and keep exactly those sprites whose body shape matches the template.
    - Sort the kept sprites by bbox top-left from left to right, then top to bottom; align all grey anchors to the first kept sprite's anchor, overlay later sprites over earlier ones, erase the aligned grey anchor, and crop to the minimal non-black bbox.

    Key insight:
    - The cyan object is only a body-shape filter, while the visible output is created solely by anchor-based registration of the matching sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    template, sprites = _classify_components(g)
    matched, _unmatched = _matched_sprites(template, sprites)
    assert 2 <= len(matched) <= 3, "expected 2-3 matching sprites"
    return _overlay_sprites(_sort_sprites(matched))


def _patch_nonblack_coords(patch, top_left):
    r0, c0 = top_left
    coords = []
    for rr, cc in np.argwhere(np.asarray(patch) != BLACK):
        coords.append((int(r0 + rr), int(c0 + cc)))
    return coords


def _bbox_overlap(box1, box2):
    return not (
        box1[2] < box2[0]
        or box2[2] < box1[0]
        or box1[3] < box2[1]
        or box2[3] < box1[1]
    )


def _place_patch(canvas, patch, blocked, placed_boxes, rng):
    h, w = patch.shape
    max_r = canvas.shape[0] - h
    max_c = canvas.shape[1] - w
    for _ in range(300):
        r0 = int(rng.integers(0, max_r + 1))
        c0 = int(rng.integers(0, max_c + 1))
        coords = _patch_nonblack_coords(patch, (r0, c0))
        if any(coord in blocked for coord in coords):
            continue
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_bbox_overlap(bbox, other) for other in placed_boxes):
            continue
        for rr, cc in coords:
            canvas[rr, cc] = int(patch[rr - r0, cc - c0])
        placed_boxes.append(bbox)
        for rr, cc in coords:
            for dr, dc in DIR8 + ((0, 0),):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]:
                    blocked.add((nr, nc))
        return True
    return False


def _single_sprite_result(sprite):
    alone = _overlay_sprites([sprite])
    return alone


def _translated_shape_overlap(sprite_a, sprite_b):
    return bool(sprite_a["rel_shape"] & sprite_b["rel_shape"])


def _choose_match_anchors(shape, candidates, match_count, rng):
    combos = list(combinations(candidates, match_count))
    rng.shuffle(combos)
    for anchors in combos:
        rel_shapes = [_relative_shape(shape, anchor) for anchor in anchors]
        if not any(
            rel_shapes[i] & rel_shapes[j]
            for i in range(match_count)
            for j in range(i + 1, match_count)
        ):
            continue
        union_all = set().union(*rel_shapes)
        if any(set().union(*(rel_shapes[j] for j in range(match_count) if j != i)) == union_all for i in range(match_count)):
            continue
        return list(anchors)
    return None


def _validate_generated_scene(grid):
    template, sprites = _classify_components(grid)
    assert 5 <= len(sprites) <= 7, "sprite count must be 5..7"
    matched, unmatched = _matched_sprites(template, sprites)
    assert 2 <= len(matched) <= 3, "need 2..3 template matches"
    assert len(unmatched) >= 3, "need distractor sprites"
    assert any(item["color"] in {m["color"] for m in matched} for item in unmatched), "at least one distractor should share a color with a matched sprite"
    assert any(_shape_area(item["shape"]) == _shape_area(template["shape"]) for item in unmatched), "at least one distractor should match the template size"
    matched_sorted = _sort_sprites(matched)
    out = _overlay_sprites(matched_sorted)
    assert np.array_equal(out, main(grid)), "main should match overlay specification"
    assert any(
        _translated_shape_overlap(matched_sorted[i], matched_sorted[j])
        for i in range(len(matched_sorted))
        for j in range(i + 1, len(matched_sorted))
    ), "matching sprites should require visible overlap after anchor alignment"
    for idx in range(len(matched_sorted)):
        subset = matched_sorted[:idx] + matched_sorted[idx + 1 :]
        assert not np.array_equal(_overlay_sprites(subset), out), "each matched sprite must affect the result"
        assert not np.array_equal(_single_sprite_result(matched_sorted[idx]), out), "overlay should not collapse to one sprite"
    assert not np.array_equal(_overlay_sprites(_sort_sprites(sprites)), out), "distractors must matter if not filtered"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        template_area = int(rng.integers(3, 8))
        template_shape = _random_connected_shape(rng, template_area)
        candidate_anchors = _anchor_candidates(template_shape)
        if len(candidate_anchors) < 2:
            continue

        match_count = 2 if rng.random() < 0.5 else 3
        match_anchors = _choose_match_anchors(template_shape, candidate_anchors, match_count, rng)
        if match_anchors is None:
            continue

        sprites = []
        matched_colors = [int(rng.choice(SPRITE_COLORS)) for _ in range(match_count)]
        for color, anchor in zip(matched_colors, match_anchors):
            sprites.append(
                {
                    "patch": _sprite_from_shape(template_shape, color, anchor),
                    "shape": template_shape,
                    "anchor": anchor,
                    "color": color,
                    "matched": True,
                }
            )

        sprite_total = int(rng.integers(5, 8))
        distractor_count = sprite_total - match_count
        near_match_added = False
        for idx in range(distractor_count):
            if not near_match_added or (idx == distractor_count - 1 and not near_match_added):
                shape = _mutate_shape(template_shape, rng)
                near_match_added = True
            else:
                for _inner in range(80):
                    area_choices = [template_area, max(3, template_area - 1), min(7, template_area + 1)]
                    area = int(area_choices[int(rng.integers(len(area_choices)))])
                    shape = _random_connected_shape(rng, area)
                    if shape != template_shape:
                        break
                else:
                    shape = _mutate_shape(template_shape, rng)
            anchors = _anchor_candidates(shape)
            if not anchors:
                break
            anchor = anchors[int(rng.integers(len(anchors)))]
            if idx == 0:
                color = matched_colors[int(rng.integers(len(matched_colors)))]
            else:
                color = int(rng.choice(SPRITE_COLORS))
            sprites.append(
                {
                    "patch": _sprite_from_shape(shape, color, anchor),
                    "shape": shape,
                    "anchor": anchor,
                    "color": color,
                    "matched": False,
                }
            )
        if len(sprites) != sprite_total:
            continue

        template_patch = _shape_patch(template_shape, TEAL)

        max_h = max([item["patch"].shape[0] for item in sprites] + [template_patch.shape[0]])
        max_w = max([item["patch"].shape[1] for item in sprites] + [template_patch.shape[1]])
        height = int(rng.integers(max(18, max_h + 8), 25))
        width = int(rng.integers(max(18, max_w + 8), 25))
        canvas = np.zeros((height, width), dtype=int)
        blocked = set()
        boxes = []

        patches = [{"patch": template_patch}] + [{"patch": item["patch"]} for item in sprites]
        order = list(range(len(patches)))
        rng.shuffle(order)
        for idx in order:
            if not _place_patch(canvas, patches[idx]["patch"], blocked, boxes, rng):
                break
        else:
            try:
                _validate_generated_scene(canvas)
            except AssertionError:
                continue
            return canvas

    raise RuntimeError("failed to generate a valid ARC instance")
