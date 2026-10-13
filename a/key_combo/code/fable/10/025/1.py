from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OBJECT = 8
SHADOW = 2

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PAIR_COUNT_RANGE = (5, 7)
VECTOR_RANGE = range(-6, 7)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200


def _bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _mirror_h(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, c1 = _bbox(cells)
    return frozenset((r, c0 + c1 - c) for r, c in cells)


def _shift(cells: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def _in_bounds(cells: frozenset[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _padded(cells: frozenset[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _component_list(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            frozen = frozenset(cells)
            norm = _normalize(frozen)
            mirror_norm = _normalize(_mirror_h(frozen))
            r0, c0, r1, c1 = _bbox(frozen)
            components.append(
                {
                    "cells": frozen,
                    "norm": norm,
                    "mirror_norm": mirror_norm,
                    "bbox": (r0, c0, r1, c1),
                    "asymmetric": norm != mirror_norm,
                    "area": len(frozen),
                }
            )
    components.sort(key=lambda comp: (comp["bbox"], tuple(sorted(comp["cells"]))))
    return components


def _object_to_shadow(comp: dict, dr: int, dc: int, *, mirror: bool) -> frozenset[tuple[int, int]]:
    cells = comp["cells"]
    moved = _mirror_h(cells) if mirror else cells
    return _shift(moved, dr, dc)


def _cells_to_shadow(cells: frozenset[tuple[int, int]], dr: int, dc: int, *, mirror: bool) -> frozenset[tuple[int, int]]:
    moved = _mirror_h(cells) if mirror else cells
    return _shift(moved, dr, dc)


def _shadow_to_object(comp: dict, dr: int, dc: int, *, mirror: bool) -> frozenset[tuple[int, int]]:
    shifted = _shift(comp["cells"], -dr, -dc)
    return _mirror_h(shifted) if mirror else shifted


def _canonical_signature(cells: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    norm = tuple(sorted(_normalize(cells)))
    mirror_norm = tuple(sorted(_normalize(_mirror_h(cells))))
    return min(norm, mirror_norm)


def _all_valid_solutions(grid: np.ndarray, *, mirror: bool) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _component_list(g, OBJECT)
    shadows = _component_list(g, SHADOW)
    if len(objects) < 3 or len(objects) != len(shadows):
        return []

    fg_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(g != BG))
    shadow_lookup = {comp["cells"]: idx for idx, comp in enumerate(shadows)}
    candidate_vectors: set[tuple[int, int]] = set()
    for obj in objects:
        want = obj["mirror_norm"] if mirror else obj["norm"]
        for shadow in shadows:
            if want == shadow["norm"]:
                candidate_vectors.add((shadow["bbox"][0] - obj["bbox"][0], shadow["bbox"][1] - obj["bbox"][1]))

    solutions: list[dict] = []
    for dr, dc in sorted(candidate_vectors):
        matched: dict[int, int] = {}
        matched_shadows: set[int] = set()
        asymmetric_complete_pairs = 0
        for oi, obj in enumerate(objects):
            pred_shadow = _object_to_shadow(obj, dr, dc, mirror=mirror)
            sj = shadow_lookup.get(pred_shadow)
            if sj is not None:
                matched[oi] = sj
                matched_shadows.add(sj)
                if obj["asymmetric"]:
                    asymmetric_complete_pairs += 1

        if len(matched) != len(objects) - 1 or len(matched_shadows) != len(shadows) - 1:
            continue

        unmatched_objects = [idx for idx in range(len(objects)) if idx not in matched]
        unmatched_shadows = [idx for idx in range(len(shadows)) if idx not in matched_shadows]
        if len(unmatched_objects) != 1 or len(unmatched_shadows) != 1:
            continue

        missing_object_idx = unmatched_objects[0]
        missing_shadow_idx = unmatched_shadows[0]
        missing_shadow = _object_to_shadow(objects[missing_object_idx], dr, dc, mirror=mirror)
        missing_object = _shadow_to_object(shadows[missing_shadow_idx], dr, dc, mirror=mirror)
        if not _in_bounds(missing_shadow, (h, w)) or not _in_bounds(missing_object, (h, w)):
            continue
        if missing_shadow & fg_cells or missing_object & fg_cells:
            continue

        full_object_sets = [obj["cells"] for obj in objects] + [missing_object]
        full_shadow_sets = [shadow["cells"] for shadow in shadows] + [missing_shadow]
        expected_shadow_sets = {_cells_to_shadow(cells, dr, dc, mirror=mirror) for cells in full_object_sets}
        if expected_shadow_sets != set(full_shadow_sets):
            continue

        solutions.append(
            {
                "vector": (dr, dc),
                "matched": matched,
                "missing_object_idx": missing_object_idx,
                "missing_shadow_idx": missing_shadow_idx,
                "missing_object": missing_object,
                "missing_shadow": missing_shadow,
                "matched_count": len(matched),
                "asymmetric_complete_pairs": asymmetric_complete_pairs,
                "missing_shape_is_asymmetric": objects[missing_object_idx]["asymmetric"] or _normalize(missing_object) != _normalize(_mirror_h(missing_object)),
            }
        )
    return solutions


def _infer_solution(grid: np.ndarray) -> dict:
    solutions = _all_valid_solutions(grid, mirror=True)
    assert len(solutions) == 1, "expected a unique mirror-plus-translation solution"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the missing teal object from the lone red shadow and draw the missing red shadow for the lone teal object.

    Input structure:
    - A black background contains teal foreground objects and red shadow objects.
    - Under one global rule, each red shadow is the left-right mirror of a teal object, shifted by the same vector.
    - Most object-shadow pairs are already complete; exactly one red shadow lacks its teal object and exactly one teal object lacks its red shadow.

    Rule:
    - Infer the shared projection rule from the complete pairs: first mirror the teal object horizontally inside its own bbox, then translate it by the common vector.
    - For the lone red shadow, reverse that rule to restore the missing teal object.
    - For the lone teal object, apply the forward rule to draw the missing red shadow.
    - Keep every existing pixel unchanged.

    Key insight:
    - Symmetric complete pairs can hide the mirror step, so the reliable rule comes from combining all complete pairs and then completing both missing sides with the same transform.
    """
    g = np.asarray(grid, dtype=int)
    solution = _infer_solution(g)
    out = g.copy()
    for r, c in solution["missing_object"]:
        out[r, c] = OBJECT
    for r, c in solution["missing_shadow"]:
        out[r, c] = SHADOW
    return out


def _random_connected_shape(h: int, w: int, target_area: int) -> frozenset[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return _normalize(frozenset(cells))


def _make_shape(*, symmetric: bool, used_signatures: set[tuple[tuple[int, int], ...]]) -> frozenset[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        target_area = random.randint(3, min(h * w - 1, 8))
        base = _random_connected_shape(h, w, target_area)
        shape = _normalize(_mirror_h(base) | base) if symmetric else base
        shape = _normalize(shape)
        area = len(shape)
        if area < 3:
            continue
        bbox = _bbox(shape)
        hh = bbox[2] - bbox[0] + 1
        ww = bbox[3] - bbox[1] + 1
        if area == hh * ww:
            continue
        if symmetric != (_normalize(shape) == _normalize(_mirror_h(shape))):
            continue
        # Keep shapes visually meaningful and prevent direct-translation shortcuts via mirror-equivalent duplicates.
        if hh == 1 or ww == 1:
            continue
        signature = _canonical_signature(shape)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        return shape
    return None


def _vector_choices() -> list[tuple[int, int]]:
    out = []
    for dr in VECTOR_RANGE:
        for dc in VECTOR_RANGE:
            if dr == 0 and dc == 0:
                continue
            if abs(dr) + abs(dc) < 4:
                continue
            out.append((dr, dc))
    random.shuffle(out)
    return out


def _place_pairs(
    h: int,
    w: int,
    shapes: list[frozenset[tuple[int, int]]],
    dr: int,
    dc: int,
) -> list[dict] | None:
    order = sorted(range(len(shapes)), key=lambda idx: (-len(shapes[idx]), _bbox(shapes[idx])))
    placed: list[dict | None] = [None] * len(shapes)
    occupied_padded: set[tuple[int, int]] = set()
    grid_shape = (h, w)

    for idx in order:
        shape = shapes[idx]
        _r0, _c0, r1, c1 = _bbox(shape)
        sh = r1 + 1
        sw = c1 + 1
        row_lo = max(0, -dr)
        row_hi = h - sh - max(dr, 0)
        col_lo = max(0, -dc)
        col_hi = w - sw - max(dc, 0)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            obj = _shift(shape, top, left)
            shadow = _shift(_mirror_h(shape), top + dr, left + dc)
            if obj & shadow:
                continue
            if _padded(obj, grid_shape) & shadow or _padded(shadow, grid_shape) & obj:
                continue
            pair_cells = obj | shadow
            pair_padded = _padded(pair_cells, grid_shape)
            if pair_padded & occupied_padded:
                continue
            occupied_padded |= pair_padded
            placed[idx] = {
                "shape": shape,
                "object": obj,
                "shadow": shadow,
                "asymmetric": _normalize(shape) != _normalize(_mirror_h(shape)),
            }
            success = True
            break
        if not success:
            return None

    return [item for item in placed if item is not None]


def _render_scene(h: int, w: int, pairs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for pair in pairs:
        for r, c in pair["shadow"]:
            grid[r, c] = SHADOW
        for r, c in pair["object"]:
            grid[r, c] = OBJECT
    return grid


def _candidate_missing_indices(pairs: list[dict]) -> list[tuple[int, int]]:
    choices = []
    n = len(pairs)
    for miss_obj_idx in range(n):
        for miss_shadow_idx in range(n):
            if miss_obj_idx == miss_shadow_idx:
                continue
            complete = [pairs[i] for i in range(n) if i not in {miss_obj_idx, miss_shadow_idx}]
            if not any(pair["asymmetric"] for pair in complete):
                continue
            if not (pairs[miss_obj_idx]["asymmetric"] or pairs[miss_shadow_idx]["asymmetric"]):
                continue
            choices.append((miss_obj_idx, miss_shadow_idx))
    random.shuffle(choices)
    return choices


def _semantic_ok(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        produced = main(inp)
    except Exception:
        return False
    if not np.array_equal(produced, expected_out):
        return False
    mirror_solutions = _all_valid_solutions(inp, mirror=True)
    if len(mirror_solutions) != 1:
        return False
    sol = mirror_solutions[0]
    if sol["matched_count"] < 3:
        return False
    if sol["asymmetric_complete_pairs"] < 1:
        return False
    if _all_valid_solutions(inp, mirror=False):
        return False
    out_objects = _component_list(expected_out, OBJECT)
    out_shadows = _component_list(expected_out, SHADOW)
    if len(out_objects) != len(out_shadows):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    # Keep at least one symmetric decoy often, but never at the cost of losing an asymmetric complete pair.
    symmetric_target = random.randint(0, min(2, pair_count - 2))
    if random.random() < 0.7:
        symmetric_target = max(1, symmetric_target)

    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[frozenset[tuple[int, int]]] = []
    flags = [True] * symmetric_target + [False] * (pair_count - symmetric_target)
    random.shuffle(flags)
    if all(flags):
        flags[0] = False
    for symmetric in flags:
        shape = _make_shape(symmetric=symmetric, used_signatures=used_signatures)
        if shape is None:
            return None
        shapes.append(shape)

    for dr, dc in _vector_choices():
        pairs = _place_pairs(h, w, shapes, dr, dc)
        if pairs is None:
            continue
        full_scene = _render_scene(h, w, pairs)
        for miss_obj_idx, miss_shadow_idx in _candidate_missing_indices(pairs):
            inp = full_scene.copy()
            for r, c in pairs[miss_obj_idx]["object"]:
                inp[r, c] = BG
            for r, c in pairs[miss_shadow_idx]["shadow"]:
                inp[r, c] = BG
            if _semantic_ok(inp, full_scene):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lonely-shadow restoration puzzle")
