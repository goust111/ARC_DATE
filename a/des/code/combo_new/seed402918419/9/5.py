from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
SHADOW = 5
LIGHT = 7
BODY_COLORS = (0, 1, 2, 3, 4, 6, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
)

MAX_GENERATION_ATTEMPTS = 400


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _transform(shape: tuple[tuple[int, int], ...], turns: int, flip: bool) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for r, c in shape:
        rr, cc = int(r), int(c)
        if flip:
            cc = -cc
        for _ in range(turns):
            rr, cc = cc, -rr
        out.append((rr, cc))
    return _normalize(out)


def _build_shape_catalog() -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes = set()
    for shape in BASE_SHAPES:
        for turns in range(4):
            for flip in (False, True):
                shapes.add(_transform(shape, turns, flip))
    return tuple(sorted(shapes, key=lambda item: (len(item), item)))


SHAPE_CATALOG = _build_shape_catalog()


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shift(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r) + int(dr), int(c) + int(dc)) for r, c in cells}


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: set[tuple[int, int]] = set()
        while stack:
            cr, cc = stack.pop()
            comp.add((int(cr), int(cc)))
            for dr, dc in DIRS4:
                nr = int(cr) + int(dr)
                nc = int(cc) + int(dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (int(r) + int(dr), int(c) + int(dc))
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS8:
            nr = int(r) + int(dr)
            nc = int(c) + int(dc)
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
        out.add((int(r), int(c)))
    return out


def _body_components(grid: np.ndarray) -> list[dict[str, object]]:
    body_mask = (grid != BG) & (grid != SHADOW) & (grid != LIGHT)
    comps = _components(body_mask)
    out: list[dict[str, object]] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1
        out.append(
            {
                "color": int(next(iter(colors))),
                "cells": comp,
                "size": len(comp),
            }
        )
    out.sort(key=lambda item: (int(item["color"]), sorted(item["cells"])))
    return out


def _shadow_component_index(grid: np.ndarray) -> tuple[dict[int, set[tuple[int, int]]], np.ndarray]:
    comps = _components(grid == SHADOW)
    index = np.zeros(grid.shape, dtype=int)
    by_id: dict[int, set[tuple[int, int]]] = {}
    for comp_id, comp in enumerate(comps, start=1):
        by_id[int(comp_id)] = comp
        for r, c in comp:
            index[r, c] = int(comp_id)
    return by_id, index


def _shadow_sign(light: tuple[int, int], body_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    lr, lc = light
    r0, c0, r1, c1 = body_bbox
    dr_sign = 1 if lr < r0 else -1 if lr > r1 else 0
    dc_sign = 1 if lc < c0 else -1 if lc > c1 else 0
    assert dr_sign != 0 or dc_sign != 0
    return dr_sign, dc_sign


def _axis_values(sign: int, limit: int) -> tuple[int, ...]:
    if sign > 0:
        return tuple(range(1, limit))
    if sign < 0:
        return tuple(range(-1, -limit, -1))
    return (0,)


def _match_expected(
    expected: set[tuple[int, int]],
    comp_by_id: dict[int, set[tuple[int, int]]],
    comp_index: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = comp_index.shape
    for r, c in expected:
        if not (0 <= r < h and 0 <= c < w):
            return None
    hit_ids = [int(comp_index[r, c]) for r, c in expected if comp_index[r, c] > 0]
    if not hit_ids:
        return None
    comp_id = int(Counter(hit_ids).most_common(1)[0][0])
    actual = comp_by_id[comp_id]
    overlap = len(expected & actual)
    missing = len(expected - actual)
    extra = len(actual - expected)
    return comp_id, overlap, missing, extra


def _vector_score(
    bodies: list[dict[str, object]],
    vector: tuple[int, int],
    comp_by_id: dict[int, set[tuple[int, int]]],
    comp_index: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    close_count = 0
    exact_count = 0
    visible_count = 0
    overlap_sum = 0
    diff_sum = 0
    matched_ids: list[int] = []
    dr, dc = vector
    for body in bodies:
        expected = _shift(body["cells"], dr, dc)
        match = _match_expected(expected, comp_by_id, comp_index)
        if match is None:
            diff_sum += int(body["size"]) + 5
            continue
        comp_id, overlap, missing, extra = match
        visible_count += 1
        overlap_sum += int(overlap)
        diff = int(missing) + int(extra)
        diff_sum += diff
        matched_ids.append(int(comp_id))
        if diff <= 1:
            close_count += 1
        if diff == 0:
            exact_count += 1
    return (
        int(close_count),
        int(exact_count),
        len(set(matched_ids)),
        int(visible_count),
        int(overlap_sum),
        -int(diff_sum),
    )


def _infer_vector(grid: np.ndarray, bodies: list[dict[str, object]]) -> tuple[int, int]:
    light_pixels = np.argwhere(grid == LIGHT)
    assert len(light_pixels) == 1
    light = (int(light_pixels[0, 0]), int(light_pixels[0, 1]))

    body_cells = set().union(*(body["cells"] for body in bodies))
    sign = _shadow_sign(light, _bbox(body_cells))
    comp_by_id, comp_index = _shadow_component_index(grid)

    h, w = grid.shape
    best_vector: tuple[int, int] | None = None
    best_score: tuple[int, int, int, int, int, int] | None = None
    for dr in _axis_values(sign[0], h):
        for dc in _axis_values(sign[1], w):
            if dr == 0 and dc == 0:
                continue
            score = _vector_score(bodies, (dr, dc), comp_by_id, comp_index)
            if best_score is None or score > best_score:
                best_score = score
                best_vector = (int(dr), int(dc))

    assert best_vector is not None
    assert best_score is not None
    assert best_score[0] == len(bodies)
    assert best_score[1] >= len(bodies) - 2
    return best_vector


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair tampered grey shadows by inferring the shared projection vector from the light source and the intact pairs.

    Input structure:
    - The background is azure.
    - There are 3-5 small solid colored body shapes, each using a distinct non-grey, non-orange color.
    - Each body has one grey shadow component produced by translating the body by the same global offset.
    - One orange pixel is a light marker placed on the opposite side of every shadow direction.

    Rule:
    - Find the orange light marker and use its position relative to the colored bodies to determine the sign of the shadow direction.
    - Among all vectors consistent with that direction, choose the one for which the translated bodies best match the grey components; the intact majority identifies the true shared offset.
    - Erase the grey shadow pixels and redraw every shadow as the exact translation of its colored body by the inferred offset.
    - Leave the orange light marker and all colored body pixels unchanged.

    Key insight:
    - The orange pixel only reveals the coarse projection direction; the exact row/column offset must be calibrated from the majority of already-correct body-shadow pairs, and then the anomalous shadows can be repaired by pure translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bodies = _body_components(g)
    assert 3 <= len(bodies) <= 5

    vector = _infer_vector(g, bodies)
    out = g.copy()
    out[out == SHADOW] = BG
    for body in bodies:
        for r, c in _shift(body["cells"], vector[0], vector[1]):
            out[r, c] = SHADOW
    return out


def _non_overlapping_shapes(vector: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    valid: list[tuple[tuple[int, int], ...]] = []
    for shape in SHAPE_CATALOG:
        body = set(shape)
        shadow = _shift(body, vector[0], vector[1])
        if body.isdisjoint(shadow):
            valid.append(shape)
    return valid


def _instantiate_shape(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar) + int(r), int(ac) + int(c)) for r, c in shape}


def _place_objects(
    h: int,
    w: int,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
    vector: tuple[int, int],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    order = list(range(len(shapes)))
    order.sort(key=lambda idx: (-len(shapes[idx]), shapes[idx]))

    occupied_expanded: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(shapes)

    for idx in order:
        shape = shapes[idx]
        rel_body = set(shape)
        rel_shadow = _shift(rel_body, vector[0], vector[1])
        rel_pair = rel_body | rel_shadow
        min_r, min_c, max_r, max_c = _bbox(rel_pair)
        anchors = [
            (r, c)
            for r in range(-min_r, h - max_r)
            for c in range(-min_c, w - max_c)
        ]
        rng.shuffle(anchors)

        candidates: list[dict[str, object]] = []
        for anchor in anchors:
            body = _instantiate_shape(shape, anchor)
            shadow = _shift(body, vector[0], vector[1])
            pair = body | shadow
            if pair & occupied_expanded:
                continue
            expanded = _expand(pair, (h, w))
            candidates.append(
                {
                    "color": int(colors[idx]),
                    "shape": shape,
                    "body": body,
                    "shadow": shadow,
                    "expanded": expanded,
                }
            )
            if len(candidates) >= 40:
                break

        if not candidates:
            return None

        choice = rng.choice(candidates)
        placed[idx] = choice
        occupied_expanded |= choice["expanded"]

    return [item for item in placed if item is not None]


def _removable_variants(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    variants: list[set[tuple[int, int]]] = []
    for cell in sorted(cells):
        variant = set(cells)
        variant.remove(cell)
        if _is_connected(variant):
            variants.append(variant)
    return variants


def _addable_variants(
    cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[set[tuple[int, int]]]:
    h, w = shape
    variants: dict[frozenset[tuple[int, int]], set[tuple[int, int]]] = {}
    for r, c in cells:
        for dr, dc in DIRS4:
            nr = int(r) + int(dr)
            nc = int(c) + int(dc)
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in cells or nxt in blocked:
                continue
            variant = set(cells)
            variant.add(nxt)
            variants[frozenset(variant)] = variant
    return list(variants.values())


def _render_grid(
    h: int,
    w: int,
    bodies: list[dict[str, object]],
    shadows: list[set[tuple[int, int]]],
    light: tuple[int, int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for body in bodies:
        for r, c in body["body"]:
            grid[r, c] = int(body["color"])
    for shadow in shadows:
        for r, c in shadow:
            grid[r, c] = SHADOW
    grid[light[0], light[1]] = LIGHT
    return grid


def _choose_light(
    h: int,
    w: int,
    bodies: list[dict[str, object]],
    occupied_expanded: set[tuple[int, int]],
    vector: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    body_cells = set().union(*(body["body"] for body in bodies))
    r0, c0, r1, c1 = _bbox(body_cells)
    row_sign = 1 if vector[0] > 0 else -1 if vector[0] < 0 else 0
    col_sign = 1 if vector[1] > 0 else -1 if vector[1] < 0 else 0

    if row_sign > 0:
        rows = range(0, r0)
    elif row_sign < 0:
        rows = range(r1 + 1, h)
    else:
        rows = range(r0, r1 + 1)

    if col_sign > 0:
        cols = range(0, c0)
    elif col_sign < 0:
        cols = range(c1 + 1, w)
    else:
        cols = range(c0, c1 + 1)

    candidates = [
        (r, c)
        for r in rows
        for c in cols
        if (r, c) not in occupied_expanded
    ]
    if not candidates:
        return None
    return tuple(rng.choice(candidates))


def _sample_vector(rng: random.Random) -> tuple[int, int]:
    direction = rng.choice(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    )
    dr_sign, dc_sign = direction
    if dr_sign == 0 or dc_sign == 0:
        magnitude = rng.randint(4, 6)
        return dr_sign * magnitude, dc_sign * magnitude
    return dr_sign * rng.randint(2, 5), dc_sign * rng.randint(2, 5)


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    body_count = rng.randint(3, 5)
    vector = _sample_vector(rng)
    valid_shapes = _non_overlapping_shapes(vector)
    if len(valid_shapes) < body_count:
        return None

    min_side = 15 + 2 * body_count
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    colors = rng.sample(BODY_COLORS, body_count)
    shapes = rng.sample(valid_shapes, body_count)

    bodies = _place_objects(h, w, shapes, colors, vector, rng)
    if bodies is None:
        return None

    clean_shadows = [set(body["shadow"]) for body in bodies]
    actual_shadows = [set(shadow) for shadow in clean_shadows]
    actual_occupied = set().union(*(body["body"] for body in bodies), *actual_shadows)

    anomaly_count = rng.randint(1, 2)
    anomaly_indices = rng.sample(range(body_count), anomaly_count)
    anomaly_types = [rng.choice(("add", "remove")) for _ in anomaly_indices]

    for idx, kind in zip(anomaly_indices, anomaly_types):
        current = set(actual_shadows[idx])
        blocked = actual_occupied - current
        remove_options = _removable_variants(current)
        add_options = _addable_variants(current, blocked, (h, w))
        if kind == "add":
            if not add_options:
                return None
            chosen = rng.choice(add_options)
        else:
            if not remove_options:
                return None
            chosen = rng.choice(remove_options)
        actual_occupied -= current
        actual_shadows[idx] = set(chosen)
        actual_occupied |= actual_shadows[idx]

    occupied_expanded = set().union(*(body["expanded"] for body in bodies))
    light = _choose_light(h, w, bodies, occupied_expanded, vector, rng)
    if light is None:
        return None
    if light in actual_occupied:
        return None

    inp = _render_grid(h, w, bodies, actual_shadows, light)
    out = _render_grid(h, w, bodies, clean_shadows, light)

    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-repair ARC input")
