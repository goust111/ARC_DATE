from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 8
OBJECT_COLORS = (1, 2, 3, 4)

GRID_SIDE_RANGE = (15, 20)
TOTAL_OBJECT_RANGE = (5, 8)
MATCH_OBJECT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 6)
SHAPE_SPAN_MAX = 4
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _monochrome_components(grid: np.ndarray, colors: Iterable[int]) -> list[dict]:
    components: list[dict] = []
    for color in colors:
        for coords in _components_from_mask(grid == int(color)):
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "shape": _normalized_shape(coords),
                }
            )
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0, keepdims=True)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _place_shape(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _drop_distance(coords: np.ndarray, occupied: np.ndarray) -> int:
    h, _w = occupied.shape
    delta = 0
    while True:
        shifted = coords.copy()
        shifted[:, 0] += delta + 1
        if np.any(shifted[:, 0] >= h):
            return delta
        if np.any(occupied[shifted[:, 0], shifted[:, 1]]):
            return delta
        delta += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every object matching the cyan template shape, then let only those objects fall downward.

    Input structure:
    - The grid has black background, one cyan 4-connected template object, and several other small 4-connected monochrome objects.
    - Non-template objects use non-cyan colors and are separated from each other by background.
    - Some non-template objects have the same translation-normalized shape as the cyan template.

    Rule:
    - Extract the cyan template's normalized 4-connected mask.
    - Among all non-black, non-cyan 4-connected objects, mark exactly those with the same normalized mask as movable.
    - Recolor movable objects to cyan, then drop each movable object straight downward as a rigid body until the next step would hit a fixed non-black pixel or the bottom edge; process objects by lower bbox first, then leftmost bbox on ties.
    - Erase the original cyan template itself, while all non-matching objects stay fixed in place and keep their original colors.

    Key insight:
    - Shape matching does not just choose a recolor target; it decides which objects become dynamic while all other objects turn into the terrain that stops the fall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_components = _components_from_mask(g == TEMPLATE_COLOR)
    assert len(template_components) == 1, "expected exactly one cyan template object"
    template_coords = template_components[0]
    template_shape = _normalized_shape(template_coords)

    objects = _monochrome_components(g, OBJECT_COLORS)
    movable = [obj for obj in objects if obj["shape"] == template_shape]
    fixed = [obj for obj in objects if obj["shape"] != template_shape]
    assert len(movable) >= 1

    occupancy = np.zeros_like(g, dtype=bool)
    for obj in fixed:
        coords = obj["coords"]
        occupancy[coords[:, 0], coords[:, 1]] = True

    ordered = sorted(movable, key=lambda obj: (-obj["bbox"][2], obj["bbox"][1]))
    landed: list[np.ndarray] = []
    for obj in ordered:
        coords = obj["coords"]
        delta = _drop_distance(coords, occupancy)
        final_coords = coords.copy()
        final_coords[:, 0] += delta
        occupancy[final_coords[:, 0], final_coords[:, 1]] = True
        landed.append(final_coords)

    out = np.zeros_like(g)
    for obj in fixed:
        _paint_cells(out, obj["coords"], obj["color"])
    for coords in landed:
        _paint_cells(out, coords, TEMPLATE_COLOR)
    return out


def _random_shape(exclude: set[tuple[tuple[int, int], ...]] | None = None, *, irregular: bool) -> tuple[tuple[int, int], ...] | None:
    exclude = set() if exclude is None else set(exclude)
    for _ in range(200):
        target_area = random.randint(*SHAPE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
        h, w = _shape_dims(norm)
        if h > SHAPE_SPAN_MAX or w > SHAPE_SPAN_MAX:
            continue
        if h < 2 or w < 2:
            continue
        if irregular and len(norm) == h * w:
            continue
        if norm in exclude:
            continue
        return norm
    return None


def _mark_occupied(occupied: np.ndarray, coords: np.ndarray) -> None:
    occupied[coords[:, 0], coords[:, 1]] = True


def _coords_clear(coords: np.ndarray, occupied: np.ndarray, *, padding: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        r0 = max(0, int(r) - padding)
        r1 = min(h, int(r) + padding + 1)
        c0 = max(0, int(c) - padding)
        c1 = min(w, int(c) + padding + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _vertical_path_clear(final_coords: np.ndarray, lift: int, occupied: np.ndarray) -> bool:
    for r, c in final_coords:
        for rr in range(int(r) - lift, int(r) + 1):
            if occupied[rr, int(c)]:
                return False
    return True


def _band_is_free(left: int, width: int, used: list[tuple[int, int]]) -> bool:
    right = left + width - 1
    for other_left, other_right in used:
        if not (right + 1 < other_left or other_right + 1 < left):
            return False
    return True


def _top_edge_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    coord_set = {tuple(map(int, cell)) for cell in coords}
    return [(r, c) for r, c in coord_set if (r - 1, c) not in coord_set]


def _bottom_edge_cells(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    shape_set = set(shape)
    return [(r, c) for r, c in shape if (r + 1, c) not in shape_set]


def _corner_positions(h: int, w: int, obj_h: int, obj_w: int) -> list[tuple[int, int]]:
    margin = 2
    return [
        (random.randint(0, margin), random.randint(0, margin)),
        (random.randint(0, margin), random.randint(max(0, w - obj_w - margin), w - obj_w)),
        (random.randint(max(0, h - obj_h - margin), h - obj_h), random.randint(0, margin)),
        (
            random.randint(max(0, h - obj_h - margin), h - obj_h),
            random.randint(max(0, w - obj_w - margin), w - obj_w),
        ),
    ]


def _place_fixed_objects(
    h: int,
    w: int,
    template_coords: np.ndarray,
    fixed_count: int,
    template_shape: tuple[tuple[int, int], ...],
) -> list[dict] | None:
    input_occupied = np.zeros((h, w), dtype=bool)
    _mark_occupied(input_occupied, template_coords)
    fixed_specs: list[dict] = []

    support_target = max(1, min(fixed_count, 3))
    for index in range(fixed_count):
        shape = _random_shape({template_shape}, irregular=False)
        if shape is None:
            return None
        shape_h, shape_w = _shape_dims(shape)
        row_lo = h // 2 if index < support_target else 0
        row_hi = h - shape_h
        candidates = [
            (r, c)
            for r in range(row_lo, row_hi + 1)
            for c in range(w - shape_w + 1)
        ]
        random.shuffle(candidates)
        placed = None
        for top, left in candidates:
            coords = _place_shape(shape, top, left)
            if _coords_clear(coords, input_occupied, padding=1):
                placed = coords
                break
        if placed is None:
            return None
        color = random.choice(OBJECT_COLORS)
        fixed_specs.append({"shape": shape, "coords": placed, "color": color})
        _mark_occupied(input_occupied, placed)
    return fixed_specs


def _place_final_matches(
    h: int,
    w: int,
    template_shape: tuple[tuple[int, int], ...],
    fixed_specs: list[dict],
    match_count: int,
) -> list[dict] | None:
    fixed_occupied = np.zeros((h, w), dtype=bool)
    for spec in fixed_specs:
        _mark_occupied(fixed_occupied, spec["coords"])

    support_cells = []
    for spec in fixed_specs:
        for cell in _top_edge_cells(spec["coords"]):
            support_cells.append(cell)
    if not support_cells:
        return None

    modes = ["obstacle", "bottom"]
    while len(modes) < match_count:
        modes.append(random.choice(["obstacle", "bottom"]))
    random.shuffle(modes)

    match_specs: list[dict] = []
    final_occupied = fixed_occupied.copy()
    used_bands: list[tuple[int, int]] = []
    bottom_cells = _bottom_edge_cells(template_shape)
    shape_h, shape_w = _shape_dims(template_shape)

    for mode in modes:
        placed = None
        if mode == "bottom":
            cols = list(range(w - shape_w + 1))
            random.shuffle(cols)
            top = h - shape_h
            for left in cols:
                if not _band_is_free(left, shape_w, used_bands):
                    continue
                coords = _place_shape(template_shape, top, left)
                if np.any(final_occupied[coords[:, 0], coords[:, 1]]):
                    continue
                placed = coords
                break
        else:
            candidates = [(cell, anchor) for cell in support_cells for anchor in bottom_cells]
            random.shuffle(candidates)
            for (support_r, support_c), (anchor_r, anchor_c) in candidates:
                top = support_r - 1 - anchor_r
                left = support_c - anchor_c
                if top < 0 or left < 0 or top + shape_h > h or left + shape_w > w:
                    continue
                if top + shape_h >= h:
                    continue
                if not _band_is_free(left, shape_w, used_bands):
                    continue
                coords = _place_shape(template_shape, top, left)
                if np.any(final_occupied[coords[:, 0], coords[:, 1]]):
                    continue
                if not np.any(fixed_occupied[coords[:, 0] + 1, coords[:, 1]]):
                    continue
                placed = coords
                break

        if placed is None:
            return None

        final_occupied[placed[:, 0], placed[:, 1]] = True
        left = int(placed[:, 1].min())
        right = int(placed[:, 1].max())
        used_bands.append((left, right))
        match_specs.append(
            {
                "mode": mode,
                "final_coords": placed,
                "input_color": random.choice(OBJECT_COLORS),
            }
        )

    if {"bottom", "obstacle"} - {spec["mode"] for spec in match_specs}:
        return None
    return match_specs


def _lift_matches(
    h: int,
    w: int,
    template_coords: np.ndarray,
    fixed_specs: list[dict],
    match_specs: list[dict],
) -> bool:
    input_occupied = np.zeros((h, w), dtype=bool)
    _mark_occupied(input_occupied, template_coords)
    for spec in fixed_specs:
        _mark_occupied(input_occupied, spec["coords"])

    ordered = sorted(match_specs, key=lambda spec: (spec["final_coords"][:, 0].min(), random.random()))
    for spec in ordered:
        final_coords = spec["final_coords"]
        max_lift = int(final_coords[:, 0].min())
        choices = list(range(1, max_lift + 1))
        random.shuffle(choices)
        placed = None
        for lift in choices:
            input_coords = final_coords.copy()
            input_coords[:, 0] -= lift
            if not _coords_clear(input_coords, input_occupied, padding=1):
                continue
            if not _vertical_path_clear(final_coords, lift, input_occupied):
                continue
            placed = (lift, input_coords)
            break
        if placed is None:
            return False
        lift, input_coords = placed
        spec["lift"] = lift
        spec["input_coords"] = input_coords
        _mark_occupied(input_occupied, input_coords)
    return True


def _expected_output(h: int, w: int, fixed_specs: list[dict], match_specs: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for spec in fixed_specs:
        _paint_cells(out, spec["coords"], spec["color"])
    for spec in match_specs:
        _paint_cells(out, spec["final_coords"], TEMPLATE_COLOR)
    return out


def _build_input(h: int, w: int, template_coords: np.ndarray, fixed_specs: list[dict], match_specs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    _paint_cells(grid, template_coords, TEMPLATE_COLOR)
    for spec in fixed_specs:
        _paint_cells(grid, spec["coords"], spec["color"])
    for spec in match_specs:
        _paint_cells(grid, spec["input_coords"], spec["input_color"])
    return grid


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    template_shape: tuple[tuple[int, int], ...],
    fixed_specs: list[dict],
    match_specs: list[dict],
) -> bool:
    template_components = _components_from_mask(inp == TEMPLATE_COLOR)
    if len(template_components) != 1:
        return False
    if _normalized_shape(template_components[0]) != template_shape:
        return False

    objects = _monochrome_components(inp, OBJECT_COLORS)
    matched = [obj for obj in objects if obj["shape"] == template_shape]
    unmatched = [obj for obj in objects if obj["shape"] != template_shape]
    if len(matched) != len(match_specs) or len(unmatched) != len(fixed_specs):
        return False
    if len(objects) < TOTAL_OBJECT_RANGE[0] or len(objects) > TOTAL_OBJECT_RANGE[1]:
        return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    template_coords = template_components[0]
    if np.any(out[template_coords[:, 0], template_coords[:, 1]] != BACKGROUND):
        return False

    moved = 0
    modes = {spec["mode"] for spec in match_specs}
    for spec in match_specs:
        if spec.get("lift", 0) <= 0:
            return False
        if np.all(spec["input_coords"] == spec["final_coords"]):
            return False
        moved += 1
        if not np.all(out[spec["final_coords"][:, 0], spec["final_coords"][:, 1]] == TEMPLATE_COLOR):
            return False
    return moved >= 1 and modes == {"bottom", "obstacle"}


def _generate_one() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)

    total_objects = random.randint(*TOTAL_OBJECT_RANGE)
    match_count = random.randint(MATCH_OBJECT_RANGE[0], min(MATCH_OBJECT_RANGE[1], total_objects - 2))
    fixed_count = total_objects - match_count

    template_shape = _random_shape(irregular=True)
    if template_shape is None:
        return None
    shape_h, shape_w = _shape_dims(template_shape)

    template_coords = None
    corner_positions = _corner_positions(h, w, shape_h, shape_w)
    random.shuffle(corner_positions)
    for top, left in corner_positions:
        candidate = _place_shape(template_shape, top, left)
        template_coords = candidate
        break
    if template_coords is None:
        return None

    fixed_specs = _place_fixed_objects(h, w, template_coords, fixed_count, template_shape)
    if fixed_specs is None:
        return None

    match_specs = _place_final_matches(h, w, template_shape, fixed_specs, match_count)
    if match_specs is None:
        return None
    if not _lift_matches(h, w, template_coords, fixed_specs, match_specs):
        return None

    inp = _build_input(h, w, template_coords, fixed_specs, match_specs)
    expected = _expected_output(h, w, fixed_specs, match_specs)
    if not _validate_semantics(inp, expected, template_shape, fixed_specs, match_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
