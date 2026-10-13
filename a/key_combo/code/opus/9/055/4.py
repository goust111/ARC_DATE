from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BACKGROUND = 0
FRAME = 5
ROTATION_TO_COLOR = {0: 2, 1: 1, 2: 3, 3: 4}
KEY_COLORS = [1, 2, 3, 4]

MAX_GENERATION_ATTEMPTS = 400
HOLE_AREAS = (3, 4, 5)
SHAPE_DIM_RANGE = range(2, 5)
OUTER_MARGIN_RANGE = (1, 2)
INNER_GAP_RANGE = (1, 2)
KEY_GAP_RANGE = (2, 3)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _normalize_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _crop_to_bbox(np.asarray(mask, dtype=bool))
    return tuple(map(int, norm.shape)), norm.tobytes()


def _orbit_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for k in range(4):
        sig = _shape_signature(np.rot90(mask, k))
        if best is None or sig < best:
            best = sig
    assert best is not None
    return best


def _full_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color != int(bg) and np.all(grid[:, c] == color):
            candidates.append((c, color))
    assert len(candidates) == 1, "expected exactly one full separator column"
    return candidates[0]


def _extract_template(left: np.ndarray, bg: int, frame: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components(left == int(frame))
    assert comps, "expected a template on the left side"
    coords = max(comps, key=len)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return left[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _extract_holes(template: np.ndarray, bg: int) -> list[dict]:
    holes = []
    for coords in _components(template == int(bg)):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if r0 == 0 or c0 == 0 or r1 == template.shape[0] - 1 or c1 == template.shape[1] - 1:
            continue
        holes.append(
            {
                "coords": coords,
                "mask": _normalize_from_coords(coords),
            }
        )
    assert holes, "expected enclosed holes in the template"
    return holes


def _extract_keys(right: np.ndarray, bg: int, frame: int) -> list[dict]:
    keys = []
    for color in sorted(int(v) for v in np.unique(right) if int(v) not in {int(bg), int(frame)}):
        for coords in _components(right == int(color)):
            keys.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "mask": _normalize_from_coords(coords),
                }
            )
    assert keys, "expected right-side key objects"
    return keys


def _match_holes_to_keys(holes: list[dict], keys: list[dict]) -> dict[int, tuple[int, int]]:
    hole_candidates: list[list[tuple[int, int]]] = []
    for hole in holes:
        candidates = []
        for key_index, key in enumerate(keys):
            for k in range(4):
                if np.array_equal(np.rot90(key["mask"], k), hole["mask"]):
                    candidates.append((key_index, k))
        assert candidates, "every hole must have at least one matching key"
        hole_candidates.append(candidates)

    order = sorted(range(len(holes)), key=lambda idx: len(hole_candidates[idx]))
    used_keys: set[int] = set()
    partial: dict[int, tuple[int, int]] = {}
    solutions: list[dict[int, tuple[int, int]]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(partial.copy())
            return
        hole_index = order[pos]
        for key_index, k in hole_candidates[hole_index]:
            if key_index in used_keys:
                continue
            used_keys.add(key_index)
            partial[hole_index] = (key_index, k)
            dfs(pos + 1)
            used_keys.remove(key_index)
            del partial[hole_index]

    dfs(0)
    assert len(solutions) == 1, "shape matching must yield a unique assignment"
    return solutions[0]


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], dict[int, tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())
    sep_col, frame = _full_separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    template, _bbox = _extract_template(left, bg, frame)
    holes = _extract_holes(template, bg)
    keys = _extract_keys(right, bg, frame)
    assert len(holes) == len(keys)
    matching = _match_holes_to_keys(holes, keys)
    return template, holes, keys, matching, bg


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each template hole with the color named by the key's required rotation.

    Input structure:
    - The background is black, and a full gray separator column splits the grid into left and right regions.
    - The left region contains one gray solid rectangle with several enclosed black hole-shapes.
    - The right region contains one monochromatic key object per hole; key colors are distractors.

    Rule:
    - Crop the gray left template to its bounding box and extract each enclosed hole as a normalized mask.
    - Extract each right-side key object as a normalized mask and find the unique hole it matches after rotating the key by 0, 90, 180, or 270 degrees.
    - Measure rotation with `np.rot90(key, k)`, so `k=1` means one 90-degree counterclockwise quarter-turn in the grid coordinate frame.
    - Fill each hole in the cropped template using the matched rotation amount: 0 -> red(2), 90 -> blue(1), 180 -> green(3), 270 -> yellow(4).

    Key insight:
    - Shape matching identifies which key belongs to which hole, but the fill color is determined only by the quarter-turn needed to rotate that key into the hole, not by the key's own color.
    """
    template, holes, _keys, matching, _bg = _analyze(grid)
    out = template.copy()
    for hole_index, hole in enumerate(holes):
        _key_index, rotation = matching[hole_index]
        color = ROTATION_TO_COLOR[int(rotation)]
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _enumerate_shape_catalog() -> list[np.ndarray]:
    catalog: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h in SHAPE_DIM_RANGE:
        for w in SHAPE_DIM_RANGE:
            n = h * w
            for bits in range(1, 1 << n):
                if bits.bit_count() not in HOLE_AREAS:
                    continue
                mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(h, w)
                if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if len(_components(mask)) != 1:
                    continue
                if any(np.array_equal(mask, np.rot90(mask, k)) for k in range(1, 4)):
                    continue
                key = _orbit_signature(mask)
                if key not in catalog:
                    catalog[key] = mask.copy()
    return list(catalog.values())


SHAPE_CATALOG = _enumerate_shape_catalog()


def _sample_shape_set() -> list[np.ndarray]:
    while True:
        shapes = random.sample(SHAPE_CATALOG, 4)
        areas = [int(shape.sum()) for shape in shapes]
        if len(set(areas)) == 4:
            continue
        return [shape.copy() for shape in shapes]


def _sample_key_colors(fill_colors: list[int]) -> list[int]:
    perms = [list(perm) for perm in permutations(KEY_COLORS) if all(int(a) != int(b) for a, b in zip(perm, fill_colors))]
    return random.choice(perms)


def _visible_bbox_duplicates(masks: list[np.ndarray]) -> bool:
    bboxes = [tuple(map(int, mask.shape)) for mask in masks]
    return len(set(bboxes)) < len(bboxes)


def _render_template(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    row_gap = random.randint(*INNER_GAP_RANGE)
    col_gap = random.randint(*INNER_GAP_RANGE)

    top_left, top_right, bottom_left, bottom_right = entries
    row_heights = [
        max(top_left["hole_mask"].shape[0], top_right["hole_mask"].shape[0]),
        max(bottom_left["hole_mask"].shape[0], bottom_right["hole_mask"].shape[0]),
    ]
    col_widths = [
        max(top_left["hole_mask"].shape[1], bottom_left["hole_mask"].shape[1]),
        max(top_right["hole_mask"].shape[1], bottom_right["hole_mask"].shape[1]),
    ]

    template_h = 1 + row_heights[0] + row_gap + row_heights[1] + 1
    template_w = 1 + col_widths[0] + col_gap + col_widths[1] + 1

    template = np.full((template_h, template_w), FRAME, dtype=int)
    expected = template.copy()

    placements = [
        (1, 1, row_heights[0], col_widths[0]),
        (1, 1 + col_widths[0] + col_gap, row_heights[0], col_widths[1]),
        (1 + row_heights[0] + row_gap, 1, row_heights[1], col_widths[0]),
        (1 + row_heights[0] + row_gap, 1 + col_widths[0] + col_gap, row_heights[1], col_widths[1]),
    ]

    for entry, (slot_r0, slot_c0, slot_h, slot_w) in zip(entries, placements):
        hole = entry["hole_mask"]
        hr, hc = hole.shape
        dr = random.randint(0, slot_h - hr)
        dc = random.randint(0, slot_w - hc)
        r0 = slot_r0 + dr
        c0 = slot_c0 + dc
        rr, cc = np.where(hole)
        template[r0 + rr, c0 + cc] = BACKGROUND
        expected[r0 + rr, c0 + cc] = int(entry["fill_color"])
    return template, expected


def _render_keys(entries: list[dict]) -> np.ndarray:
    row_gap = random.randint(*KEY_GAP_RANGE)
    col_gap = random.randint(*KEY_GAP_RANGE)

    top_left, top_right, bottom_left, bottom_right = entries
    row_heights = [
        max(top_left["key_mask"].shape[0], top_right["key_mask"].shape[0]),
        max(bottom_left["key_mask"].shape[0], bottom_right["key_mask"].shape[0]),
    ]
    col_widths = [
        max(top_left["key_mask"].shape[1], bottom_left["key_mask"].shape[1]),
        max(top_right["key_mask"].shape[1], bottom_right["key_mask"].shape[1]),
    ]

    panel_h = 1 + row_heights[0] + row_gap + row_heights[1] + 1
    panel_w = 1 + col_widths[0] + col_gap + col_widths[1] + 1
    panel = np.zeros((panel_h, panel_w), dtype=int)

    placements = [
        (1, 1, row_heights[0], col_widths[0]),
        (1, 1 + col_widths[0] + col_gap, row_heights[0], col_widths[1]),
        (1 + row_heights[0] + row_gap, 1, row_heights[1], col_widths[0]),
        (1 + row_heights[0] + row_gap, 1 + col_widths[0] + col_gap, row_heights[1], col_widths[1]),
    ]

    for entry, (slot_r0, slot_c0, slot_h, slot_w) in zip(entries, placements):
        key_mask = entry["key_mask"]
        kr, kc = key_mask.shape
        dr = random.randint(0, slot_h - kr)
        dc = random.randint(0, slot_w - kc)
        r0 = slot_r0 + dr
        c0 = slot_c0 + dc
        rr, cc = np.where(key_mask)
        panel[r0 + rr, c0 + cc] = int(entry["key_color"])
    return panel


def _compose_input(template: np.ndarray, keys: np.ndarray) -> np.ndarray:
    left_margin_top = random.randint(*OUTER_MARGIN_RANGE)
    left_margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
    left_margin_left = random.randint(*OUTER_MARGIN_RANGE)
    left_margin_right = random.randint(*OUTER_MARGIN_RANGE)
    right_margin_top = random.randint(*OUTER_MARGIN_RANGE)
    right_margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
    right_margin_left = random.randint(*OUTER_MARGIN_RANGE)
    right_margin_right = random.randint(*OUTER_MARGIN_RANGE)

    left_h = template.shape[0] + left_margin_top + left_margin_bottom
    left_w = template.shape[1] + left_margin_left + left_margin_right
    right_h = keys.shape[0] + right_margin_top + right_margin_bottom
    right_w = keys.shape[1] + right_margin_left + right_margin_right

    full_h = max(left_h, right_h)
    full_w = left_w + 1 + right_w
    if full_h > 30 or full_w > 30:
        raise ValueError("grid too large")

    grid = np.zeros((full_h, full_w), dtype=int)
    sep_col = left_w
    grid[:, sep_col] = FRAME

    template_top = random.randint(0, full_h - template.shape[0])
    template_left = random.randint(0, left_w - template.shape[1])
    grid[
        template_top : template_top + template.shape[0],
        template_left : template_left + template.shape[1],
    ] = template

    key_top = random.randint(0, full_h - keys.shape[0])
    key_left = sep_col + 1 + random.randint(0, right_w - keys.shape[1])
    key_slice = grid[key_top : key_top + keys.shape[0], key_left : key_left + keys.shape[1]]
    assert not np.any((key_slice != BACKGROUND) & (keys != BACKGROUND))
    key_slice[:] = np.where(keys != BACKGROUND, keys, key_slice)
    return grid


def _build_candidate() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    shapes = _sample_shape_set()
    hole_orientations = [random.randrange(4) for _ in range(4)]
    rotations = random.sample([0, 1, 2, 3], 4)

    entries = []
    for shape, hole_rot, rotation in zip(shapes, hole_orientations, rotations):
        hole_mask = np.rot90(shape, hole_rot)
        key_mask = np.rot90(hole_mask, (-rotation) % 4)
        entries.append(
            {
                "hole_mask": hole_mask,
                "key_mask": key_mask,
                "rotation": int(rotation),
                "fill_color": int(ROTATION_TO_COLOR[int(rotation)]),
            }
        )

    if not _visible_bbox_duplicates([entry["hole_mask"] for entry in entries]):
        return None

    fill_colors = [entry["fill_color"] for entry in entries]
    key_colors = _sample_key_colors(fill_colors)
    for entry, key_color in zip(entries, key_colors):
        entry["key_color"] = int(key_color)

    random.shuffle(entries)
    template, expected = _render_template(entries)

    key_entries = entries.copy()
    random.shuffle(key_entries)
    keys = _render_keys(key_entries)

    try:
        grid = _compose_input(template, keys)
    except ValueError:
        return None
    return grid, expected, entries


def _validate_candidate(grid: np.ndarray, expected: np.ndarray, entries: list[dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    template, holes, keys, matching, _bg = _analyze(grid)
    if not np.array_equal(template.shape, expected.shape):
        return False
    if len(holes) != 4 or len(keys) != 4:
        return False

    rotations = sorted(int(matching[idx][1]) for idx in range(len(holes)))
    if rotations != [0, 1, 2, 3]:
        return False

    if not all(int(entry["key_color"]) != int(entry["fill_color"]) for entry in entries):
        return False

    hole_masks = [entry["hole_mask"] for entry in entries]
    if len({_orbit_signature(mask) for mask in hole_masks}) != 4:
        return False
    if not _visible_bbox_duplicates(hole_masks):
        return False

    areas = [int(mask.sum()) for mask in hole_masks]
    if len(set(areas)) == 4:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, expected, entries = built
        if _validate_candidate(grid, expected, entries):
            return grid
    raise ValueError("failed to generate a valid ARC sample")
