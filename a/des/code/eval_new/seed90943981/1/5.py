from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
SHAPE_COLORS = (1, 2, 3, 4, 6, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
OBJECT_RETRIES = 100
POLYOMINO_RETRIES = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _component_bounds(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_shape_components(
    g: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, ORANGE) or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                cell_to_comp[(rr, cc)] = len(comps)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append({"color": color, "cells": cells})

    return comps, cell_to_comp


def _extract_markers(
    g: np.ndarray, cell_to_comp: dict[tuple[int, int], int], n_comps: int
) -> dict[int, dict[str, object]]:
    info: dict[int, dict[str, object]] = {}
    for r, c in map(tuple, np.argwhere(g == ORANGE)):
        adj: list[tuple[int, tuple[int, int]]] = []
        for nr, nc in _neighbors4(int(r), int(c)):
            comp_id = cell_to_comp.get((nr, nc))
            if comp_id is not None:
                adj.append((comp_id, (int(r) - int(nr), int(c) - int(nc))))

        assert adj, "every orange marker must touch a shape"
        comp_ids = {comp_id for comp_id, _ in adj}
        dirs = {direction for _, direction in adj}
        assert len(comp_ids) == 1, "marker must be adjacent to exactly one shape"
        assert len(dirs) == 1, "marker direction must be unambiguous"

        comp_id = next(iter(comp_ids))
        assert comp_id not in info, "each shape must have exactly one orange marker"
        info[comp_id] = {"marker": (int(r), int(c)), "direction": next(iter(dirs))}

    assert len(info) == n_comps, "every shape must have one orange marker"
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored shape into a same-sized directional shadow and mark shadow overlaps specially.

    Input structure:
    - Background is black.
    - Several non-overlapping single-color filled shapes appear in the grid; each shape is a 4-connected polyomino.
    - Each shape has exactly one orange marker pixel just outside one of its sides, indicating a projection direction.

    Rule:
    - For each shape, infer the projection direction from the orange marker's position relative to the shape.
    - Translate the whole shape by its span in that direction: use the shape's width for left/right projection and height for up/down projection.
    - Draw translated cells as grey shadow pixels.
    - Any non-shape cell covered by two or more shadows becomes maroon instead of grey.
    - Original shape pixels keep their original colors if a shadow lands on them, and orange markers are removed from the output.

    Key insight:
    - The orange marker does not mark the shadow itself; it encodes a direction, while the translation distance comes from the shape's own directional span.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    comps, cell_to_comp = _find_shape_components(g)
    if not comps:
        out = g.copy()
        out[out == ORANGE] = BG
        return out

    marker_info = _extract_markers(g, cell_to_comp, len(comps))

    shadow_counts = np.zeros((h, w), dtype=int)
    shape_mask = (g != BG) & (g != ORANGE)

    for comp_id, comp in enumerate(comps):
        cells = comp["cells"]
        assert isinstance(cells, list)
        r0, c0, r1, c1 = _component_bounds(cells)
        dr, dc = marker_info[comp_id]["direction"]
        distance = (c1 - c0 + 1) if dc else (r1 - r0 + 1)
        for r, c in cells:
            rr = r + dr * distance
            cc = c + dc * distance
            if 0 <= rr < h and 0 <= cc < w:
                shadow_counts[rr, cc] += 1

    out = g.copy()
    out[out == ORANGE] = BG
    shadow_only = (shadow_counts > 0) & (~shape_mask)
    out[shadow_only] = GREY
    out[(shadow_counts > 1) & (~shape_mask)] = MAROON
    return out


def _random_polyomino(area: int) -> list[tuple[int, int]] | None:
    for _ in range(POLYOMINO_RETRIES):
        max_h = min(4, area)
        max_w = min(4, area)
        height = random.randint(1, max_h)
        width = random.randint(1, max_w)
        if height * width < area:
            continue

        start = (random.randint(0, height - 1), random.randint(0, width - 1))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) == area:
            return _normalize(cells)
    return None


def _marker_candidates(shape: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    shape_set = set(shape)
    candidates = {direction: [] for direction in DIRS4}
    for r, c in shape:
        for dr, dc in DIRS4:
            pos = (r + dr, c + dc)
            if pos in shape_set:
                continue
            if sum((nr, nc) in shape_set for nr, nc in _neighbors4(*pos)) == 1:
                candidates[(dr, dc)].append(pos)
    return candidates


def _make_object(direction: tuple[int, int]) -> dict[str, object] | None:
    for _ in range(OBJECT_RETRIES):
        area = random.randint(3, 8)
        shape = _random_polyomino(area)
        if shape is None:
            continue
        candidates = _marker_candidates(shape)
        if not candidates[direction]:
            continue

        marker = random.choice(candidates[direction])
        r0, c0, r1, c1 = _component_bounds(shape)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        dr, dc = direction
        distance = width if dc else height
        shadow = [(r + dr * distance, c + dc * distance) for r, c in shape]
        return {
            "shape_rel": shape,
            "marker_rel": marker,
            "shadow_rel": shadow,
            "direction": direction,
            "height": height,
            "width": width,
        }
    return None


def _dilate8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((r + dr, c + dc))
    return halo


def _place_object(
    obj: dict[str, object], color: int, h: int, w: int, blocked: set[tuple[int, int]]
) -> dict[str, object] | None:
    footprint = list(obj["shape_rel"]) + [obj["marker_rel"]] + list(obj["shadow_rel"])
    min_r = min(r for r, _ in footprint)
    max_r = max(r for r, _ in footprint)
    min_c = min(c for _, c in footprint)
    max_c = max(c for _, c in footprint)
    row_lo, row_hi = -min_r, h - 1 - max_r
    col_lo, col_hi = -min_c, w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(OBJECT_RETRIES):
        shift_r = random.randint(row_lo, row_hi)
        shift_c = random.randint(col_lo, col_hi)
        shape_abs = [(shift_r + r, shift_c + c) for r, c in obj["shape_rel"]]
        marker_abs = (shift_r + obj["marker_rel"][0], shift_c + obj["marker_rel"][1])
        source_cells = set(shape_abs) | {marker_abs}
        if any(cell in blocked for cell in source_cells):
            continue
        shadow_abs = [(shift_r + r, shift_c + c) for r, c in obj["shadow_rel"]]
        return {
            "color": color,
            "shape": shape_abs,
            "marker": marker_abs,
            "shadow": shadow_abs,
            "direction": obj["direction"],
            "height": obj["height"],
            "width": obj["width"],
        }
    return None


def _shadow_metrics(objects: list[dict[str, object]], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    shape_mask = np.zeros((h, w), dtype=bool)
    shadow_counts = np.zeros((h, w), dtype=int)
    for obj in objects:
        for r, c in obj["shape"]:
            shape_mask[r, c] = True
    for obj in objects:
        for r, c in obj["shadow"]:
            if 0 <= r < h and 0 <= c < w:
                shadow_counts[r, c] += 1
    return shape_mask, shadow_counts


def _build_case(mode: str) -> np.ndarray | None:
    if mode == "simple":
        h = random.randint(15, 24)
        w = random.randint(15, 24)
        n_shapes = random.randint(2, 3)
    elif mode == "shadow_overlap":
        h = random.randint(12, 18)
        w = random.randint(12, 18)
        n_shapes = random.randint(3, 4)
    else:
        h = random.randint(12, 17)
        w = random.randint(12, 17)
        n_shapes = random.randint(3, 4)

    directions = random.sample(DIRS4, k=min(4, n_shapes))
    while len(directions) < n_shapes:
        directions.append(random.choice(DIRS4))
    random.shuffle(directions)
    colors = random.sample(SHAPE_COLORS, n_shapes)

    blocked: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []
    for color, direction in zip(colors, directions):
        base = _make_object(direction)
        if base is None:
            return None
        placed = _place_object(base, color, h, w, blocked)
        if placed is None:
            return None
        objects.append(placed)
        blocked |= _dilate8(set(placed["shape"]) | {placed["marker"]})

    shape_mask, shadow_counts = _shadow_metrics(objects, h, w)
    visible_shadow = bool(np.any((shadow_counts > 0) & (~shape_mask)))
    shadow_overlap = bool(np.any((shadow_counts > 1) & (~shape_mask)))
    shape_priority = bool(np.any((shadow_counts > 0) & shape_mask))

    if not visible_shadow:
        return None
    if mode == "simple" and (shadow_overlap or shape_priority):
        return None
    if mode == "shadow_overlap" and (not shadow_overlap or shape_priority):
        return None
    if mode == "shape_priority" and not shape_priority:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj["shape"]:
            grid[r, c] = int(obj["color"])
    for obj in objects:
        mr, mc = obj["marker"]
        if grid[mr, mc] != BG:
            return None
        grid[mr, mc] = ORANGE

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if mode == "shadow_overlap" and not np.any(out == MAROON):
        return None
    if mode == "simple" and np.any(out == MAROON):
        return None
    return grid


def generate_input() -> np.ndarray:
    modes = ["simple", "shadow_overlap", "shape_priority"]
    for _ in range(GEN_RETRIES):
        grid = _build_case(random.choice(modes))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection puzzle instance")
