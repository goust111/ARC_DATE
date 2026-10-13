from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
REPEATED_OUT = 1
UNIQUE_OUT = 2
INPUT_FG = 5
MAX_GENERATION_ATTEMPTS = 500
GROUP_PATTERNS = ((3,), (4,), (2, 2), (3, 2))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _bbox_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _transform_shape(
    shape: tuple[tuple[int, int], ...], rotate_quarter_turns: int, flip_lr: bool
) -> tuple[tuple[int, int], ...]:
    cells = tuple(shape)
    for _ in range(int(rotate_quarter_turns) % 4):
        height = max(r for r, _ in cells) + 1
        cells = tuple((c, height - 1 - r) for r, c in cells)
    if flip_lr:
        width = max(c for _, c in cells) + 1
        cells = tuple((r, width - 1 - c) for r, c in cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _all_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = {
        _transform_shape(shape, rot, flip)
        for rot in range(4)
        for flip in (False, True)
    }
    return sorted(variants)


BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
)

SHAPE_VARIANTS: list[dict] = []
for family_id, base in enumerate(BASE_SHAPES):
    for variant in _all_variants(base):
        SHAPE_VARIANTS.append(
            {
                "family": int(family_id),
                "shape": variant,
                "area": len(variant),
                "dims": _bbox_dims(variant),
            }
        )


def _analyze(grid: np.ndarray) -> tuple[list[dict], dict[tuple[tuple[int, int], ...], list[dict]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all(np.isin(g, [BG, INPUT_FG])), "input must contain only black background and grey foreground"
    comps = []
    for coords in _components(g == INPUT_FG):
        key = _shape_key(coords)
        comps.append({"coords": coords, "shape": key, "area": int(coords.shape[0])})
    groups: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for comp in comps:
        groups.setdefault(comp["shape"], []).append(comp)
    return comps, groups


def _expected_output(grid: np.ndarray) -> np.ndarray:
    comps, groups = _analyze(grid)
    assert 4 <= len(comps) <= 6, "expected 4-6 components"
    singleton_shapes = [shape for shape, members in groups.items() if len(members) == 1]
    repeated_shapes = [shape for shape, members in groups.items() if len(members) >= 2]
    assert len(singleton_shapes) == 1, "expected exactly one unique shape"
    assert 1 <= len(repeated_shapes) <= 2, "expected one or two repeated-shape groups"

    out = np.asarray(grid, dtype=int).copy()
    unique_shape = singleton_shapes[0]

    for shape, members in groups.items():
        color = UNIQUE_OUT if shape == unique_shape else REPEATED_OUT
        for member in members:
            coords = member["coords"]
            out[coords[:, 0], coords[:, 1]] = color

    unique_coords = groups[unique_shape][0]["coords"]
    h, w = map(int, out.shape)
    for r, c in unique_coords:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
                    out[rr, cc] = UNIQUE_OUT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor repeated grey shapes blue and bloom the single unique grey shape into a red 3x3-neighborhood stamp.

    Input structure:
    - The background is black, and every foreground object is grey.
    - There are 4-6 pairwise separated 4-connected grey components, each with 3-6 pixels.
    - Exactly one component has a normalized tight-bbox shape that appears once; every other shape appears in a repeated group.

    Rule:
    - Extract all grey 4-connected components and normalize each shape by translating its tight bbox to the origin.
    - Recolor every component in a repeated-shape group blue without changing its footprint.
    - Recolor the single unique-shape component red, then for every red source pixel fill the black cells in its centered 3x3 neighborhood with red.
    - Never overwrite non-background cells while expanding, so repeated blue pixels stay blue.

    Key insight:
    - Shape frequency is not the final answer by itself; it decides which operation to apply, so only the singleton shape receives the spatial neighborhood expansion.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _shape_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top) + r, int(left) + c) for r, c in shape]


def _chebyshev_far_enough(existing: list[tuple[int, int]], new_cells: list[tuple[int, int]]) -> bool:
    for r, c in new_cells:
        for rr, cc in existing:
            if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) < 3:
                return False
    return True


def _place_objects(
    height: int, width: int, objects: list[dict]
) -> list[dict] | None:
    occupied: list[tuple[int, int]] = []
    placed: list[dict] = []

    ordered = sorted(
        objects,
        key=lambda obj: (-len(obj["shape"]), -_bbox_dims(obj["shape"])[0] * _bbox_dims(obj["shape"])[1], obj["group"]),
    )
    for obj in ordered:
        shape = obj["shape"]
        sh, sw = _bbox_dims(shape)
        candidates = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
        random.shuffle(candidates)
        if obj["group"] == "unique" and random.random() < 0.5:
            border_first = [
                pos
                for pos in candidates
                if pos[0] == 0 or pos[1] == 0 or pos[0] + sh == height or pos[1] + sw == width
            ]
            inner = [pos for pos in candidates if pos not in border_first]
            candidates = border_first + inner
        chosen = None
        for top, left in candidates:
            cells = _shape_cells(shape, top, left)
            if _chebyshev_far_enough(occupied, cells):
                chosen = (top, left, cells)
                break
        if chosen is None:
            return None
        top, left, cells = chosen
        occupied.extend(cells)
        placed.append(
            {
                "shape": shape,
                "group": obj["group"],
                "top": int(top),
                "left": int(left),
                "cells": cells,
            }
        )
    return placed


def _choose_shape_set(pattern: tuple[int, ...]) -> tuple[list[tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...]] | None:
    repeated_group_count = len(pattern)
    for _ in range(200):
        repeated = random.sample(SHAPE_VARIANTS, repeated_group_count)
        repeated_families = {item["family"] for item in repeated}
        rep_areas = {item["area"] for item in repeated}
        rep_dims = {item["dims"] for item in repeated}
        unique_candidates = [
            item
            for item in SHAPE_VARIANTS
            if item["family"] not in repeated_families
            and item["area"] in rep_areas
            and item["dims"] in rep_dims
        ]
        if not unique_candidates:
            continue
        unique = random.choice(unique_candidates)
        repeated_shapes = [item["shape"] for item in repeated]
        if unique["shape"] in repeated_shapes:
            continue
        if len(set(repeated_shapes + [unique["shape"]])) != repeated_group_count + 1:
            continue
        return repeated_shapes, unique["shape"]
    return None


def _build_attempt() -> np.ndarray | None:
    pattern = random.choice(GROUP_PATTERNS)
    chosen = _choose_shape_set(pattern)
    if chosen is None:
        return None
    repeated_shapes, unique_shape = chosen

    total_objects = sum(pattern) + 1
    min_side = {4: 12, 5: 13, 6: 15}[int(total_objects)]
    height = random.randint(min_side, min(min_side + 3, 18))
    width = random.randint(min_side, min(min_side + 3, 18))

    objects: list[dict] = []
    for group_index, (shape, copies) in enumerate(zip(repeated_shapes, pattern)):
        for _ in range(int(copies)):
            objects.append({"shape": shape, "group": f"repeat_{group_index}"})
    objects.append({"shape": unique_shape, "group": "unique"})
    random.shuffle(objects)

    placed = _place_objects(height, width, objects)
    if placed is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for obj in placed:
        for r, c in obj["cells"]:
            grid[int(r), int(c)] = INPUT_FG
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    comps, groups = _analyze(grid)
    if not (4 <= len(comps) <= 6):
        return False
    if not all(3 <= int(comp["area"]) <= 6 for comp in comps):
        return False

    unique_shapes = [shape for shape, members in groups.items() if len(members) == 1]
    repeated_shapes = [shape for shape, members in groups.items() if len(members) >= 2]
    if len(unique_shapes) != 1 or not (1 <= len(repeated_shapes) <= 2):
        return False

    coords_list = [comp["coords"] for comp in comps]
    for i in range(len(coords_list)):
        for j in range(i + 1, len(coords_list)):
            a = coords_list[i]
            b = coords_list[j]
            min_cheb = min(
                max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2)))
                for r1, c1 in a
                for r2, c2 in b
            )
            if min_cheb < 3:
                return False

    group_shapes = list(groups)
    same_area_counterexample = False
    same_bbox_counterexample = False
    for i in range(len(group_shapes)):
        for j in range(i + 1, len(group_shapes)):
            s1 = group_shapes[i]
            s2 = group_shapes[j]
            if len(s1) == len(s2) and s1 != s2:
                same_area_counterexample = True
            if _bbox_dims(s1) == _bbox_dims(s2) and s1 != s2:
                same_bbox_counterexample = True
    if not (same_area_counterexample and same_bbox_counterexample):
        return False

    out = main(grid)
    expected = _expected_output(grid)
    if not np.array_equal(out, expected):
        return False

    unique_shape = unique_shapes[0]
    unique_coords = groups[unique_shape][0]["coords"]
    repeated_cells = {
        (int(r), int(c))
        for shape, members in groups.items()
        if shape != unique_shape
        for member in members
        for r, c in member["coords"]
    }
    unique_cells = {(int(r), int(c)) for r, c in unique_coords}

    if repeated_cells != set(map(tuple, np.argwhere(out == REPEATED_OUT))):
        return False

    expected_red = set(unique_cells)
    h, w = map(int, grid.shape)
    for r, c in unique_cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
                    expected_red.add((rr, cc))
    if expected_red != set(map(tuple, np.argwhere(out == UNIQUE_OUT))):
        return False
    if len(expected_red) <= len(unique_cells):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is None:
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
