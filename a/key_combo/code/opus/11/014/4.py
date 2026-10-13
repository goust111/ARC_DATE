from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
RED = 2
TEAL = 8

ROTATION_OFFSETS = {
    0: (0, 2),
    1: (2, 0),
    2: (0, -2),
    3: (-2, 0),
}

GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (12, 19)
OBJECT_COUNT_RANGE = (3, 5)
TEMPLATE_ROW_RANGE = (0, 2)
TEMPLATE_COL_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 300
SEARCH_CANDIDATE_CAP = 90


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _normalize_shape(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.array(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _rotate_shape_cw(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape([(-c, r) for r, c in shape])


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_to_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _translate_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _buffer8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _template_variants(template_mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(template_mask, -k) for k in range(4)]


def _build_template_catalog() -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    base_cells = [(r, c) for r in range(3) for c in range(4)]

    for size in range(4, 7):
        for combo in combinations(base_cells, size):
            shape = _normalize_shape(list(combo))
            mask = _shape_to_mask(shape)
            if len(_components(mask, connectivity=4)) != 1:
                continue

            rotations = []
            cur = shape
            for _ in range(4):
                rotations.append(cur)
                cur = _rotate_shape_cw(cur)
            if len(set(rotations)) != 4:
                continue

            valid_shift = True
            for rot_idx, rot_shape in enumerate(rotations):
                cells = set(rot_shape)
                dr, dc = ROTATION_OFFSETS[rot_idx]
                if not cells.isdisjoint(_translate_cells(cells, dr, dc)):
                    valid_shift = False
                    break
            if valid_shift:
                shapes.add(shape)

    return tuple(sorted(shapes))


TEMPLATE_CATALOG = _build_template_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red copy of the teal template according to its rotation state.

    Input structure:
    - A black background contains one small teal template in the upper-left area.
    - Elsewhere, 3-5 red 4-connected objects are exact rotated copies of that template.
    - The template is asymmetric under 90/180/270-degree rotation, so every red copy has a unique rotation state.

    Rule:
    - Extract the teal template as a tight shape mask.
    - For each red object, determine whether it matches the template at 0, 90, 180, or 270 degrees clockwise.
    - Translate the whole red object by 2 cells using the rotation-coded direction:
      0 degrees -> right, 90 degrees -> down, 180 degrees -> left, 270 degrees -> up.
    - Erase all original red positions, redraw the translated red objects, and leave the teal template unchanged.

    Key insight:
    The motion instruction is encoded by geometric orientation rather than by color: the same red color moves differently depending on how the object is rotated relative to the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    template_components = _components(g == TEAL, connectivity=4)
    assert len(template_components) == 1, "expected exactly one teal template"
    template_mask = _component_mask(template_components[0])
    variants = _template_variants(template_mask)

    red_components = _components(g == RED, connectivity=4)
    assert red_components, "expected at least one red object"

    out = g.copy()
    out[g == RED] = BG
    occupied: set[tuple[int, int]] = set()

    for coords in red_components:
        obj_mask = _component_mask(coords)
        matches = [k for k, variant in enumerate(variants) if np.array_equal(obj_mask, variant)]
        assert len(matches) == 1, "each red object must match exactly one template rotation"
        rotation = matches[0]
        dr, dc = ROTATION_OFFSETS[rotation]
        moved = {(int(r) + dr, int(c) + dc) for r, c in coords}
        assert all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in moved), "translated object left the grid"
        assert occupied.isdisjoint(moved), "translated objects must not overlap"
        occupied.update(moved)
        _paint(out, moved, RED)

    return out


def _sample_rotations(count: int) -> list[int]:
    if count >= 4:
        rotations = [0, 1, 2, 3]
        while len(rotations) < count:
            rotations.append(random.randrange(4))
        random.shuffle(rotations)
        return rotations
    return random.sample([0, 1, 2, 3], count)


def _candidate_plans(
    grid_shape: tuple[int, int],
    variant_mask: np.ndarray,
    rotation: int,
) -> list[dict[str, object]]:
    h, w = grid_shape
    variant_shape = _mask_to_shape(variant_mask)
    obj_cells = set(variant_shape)
    obj_h, obj_w = variant_mask.shape
    dr, dc = ROTATION_OFFSETS[rotation]

    r_min = max(0, -dr)
    r_max = min(h - obj_h, h - obj_h - dr)
    c_min = max(0, -dc)
    c_max = min(w - obj_w, w - obj_w - dc)
    if r_min > r_max or c_min > c_max:
        return []

    plans: list[dict[str, object]] = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            source = {(r0 + r, c0 + c) for r, c in obj_cells}
            dest = {(r + dr, c + dc) for r, c in source}
            union = source | dest
            plans.append(
                {
                    "rotation": rotation,
                    "source": source,
                    "dest": dest,
                    "buffer": _buffer8(union, grid_shape),
                }
            )
    return plans


def _place_objects(
    all_candidates: list[list[dict[str, object]]],
    reserved_buffer: set[tuple[int, int]],
    depth: int = 0,
) -> list[dict[str, object]] | None:
    if depth == len(all_candidates):
        return []

    candidates = [
        cand for cand in all_candidates[depth] if cast_buffer(cand["buffer"]).isdisjoint(reserved_buffer)
    ]
    if not candidates:
        return None

    sample_count = min(len(candidates), SEARCH_CANDIDATE_CAP)
    if len(candidates) > sample_count:
        candidates = random.sample(candidates, sample_count)
    else:
        random.shuffle(candidates)

    for cand in candidates:
        next_reserved = reserved_buffer | cast_buffer(cand["buffer"])
        rest = _place_objects(all_candidates, next_reserved, depth + 1)
        if rest is not None:
            return [cand, *rest]
    return None


def cast_buffer(value: object) -> set[tuple[int, int]]:
    return value if isinstance(value, set) else set(value)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    template_shape = random.choice(TEMPLATE_CATALOG)
    template_mask = _shape_to_mask(template_shape)
    th, tw = template_mask.shape
    tr = random.randint(TEMPLATE_ROW_RANGE[0], min(TEMPLATE_ROW_RANGE[1], h - th))
    tc = random.randint(TEMPLATE_COL_RANGE[0], min(TEMPLATE_COL_RANGE[1], w - tw))
    template_cells = {(tr + r, tc + c) for r, c in template_shape}
    _paint(grid, template_cells, TEAL)

    count = random.randint(*OBJECT_COUNT_RANGE)
    rotations = _sample_rotations(count)
    candidate_groups = [_candidate_plans(grid.shape, np.rot90(template_mask, -rot), rot) for rot in rotations]
    if any(not group for group in candidate_groups):
        return None

    placements = _place_objects(candidate_groups, _buffer8(template_cells, grid.shape))
    if placements is None:
        return None

    for placement in placements:
        _paint(grid, cast_buffer(placement["source"]), RED)

    expected = np.full_like(grid, BG)
    _paint(expected, template_cells, TEAL)
    for placement in placements:
        _paint(expected, cast_buffer(placement["dest"]), RED)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if len(_components(grid == RED, connectivity=4)) != count:
        return None
    if len(_components(actual == RED, connectivity=4)) != count:
        return None
    if np.array_equal(grid, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
