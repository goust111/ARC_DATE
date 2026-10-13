from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

SEED_STRIDE = 4
GEN_ATTEMPTS = 500
GEN_SIZE_RANGE = (15, 22)
GEN_OBSTACLE_COUNT_RANGE = (3, 5)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _extract_seed_offsets(grid: np.ndarray) -> list[tuple[int, int]]:
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2 or int(grid[0, 0]) != GREEN:
        raise AssertionError("expected a green seed touching the top-left corner")

    h, w = grid.shape
    q = deque([(0, 0)])
    seen = {(0, 0)}
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        if int(grid[r, c]) != GREEN:
            continue
        coords.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == GREEN:
                seen.add((nr, nc))
                q.append((nr, nc))
    return sorted(coords)


def _tile_origins(shape: tuple[int, int], seed_offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    max_r = max(r for r, _ in seed_offsets)
    max_c = max(c for _, c in seed_offsets)
    origins: list[tuple[int, int]] = []
    for r0 in range(0, h, SEED_STRIDE):
        for c0 in range(0, w, SEED_STRIDE):
            if r0 + max_r < h and c0 + max_c < w:
                origins.append((r0, c0))
    return origins


def _shifted_cells(
    origin: tuple[int, int], shift: int, seed_offsets: list[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = shape
    r0, c0 = origin
    cells: list[tuple[int, int]] = []
    for dr, dc in seed_offsets:
        rr = r0 + dr
        cc = c0 + dc + shift
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        cells.append((rr, cc))
    return cells


def _simulate_tiling(
    grid: np.ndarray, *, deflect_blocked: bool
) -> tuple[np.ndarray, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    seed_offsets = _extract_seed_offsets(g)
    seed_area = len(seed_offsets)

    obstacle_mask = g == GREY
    working = np.zeros_like(g)
    working[obstacle_mask] = GREY
    green_mask = np.zeros_like(obstacle_mask, dtype=bool)
    placements: list[dict] = []

    for origin in _tile_origins(g.shape, seed_offsets):
        natural_cells = _shifted_cells(origin, 0, seed_offsets, g.shape)
        assert natural_cells is not None
        obstructed_by_obstacle = any(obstacle_mask[r, c] for r, c in natural_cells)
        shift = 0
        placed = False
        dropped = False
        final_cells: list[tuple[int, int]] | None = None

        while True:
            cells = _shifted_cells(origin, shift, seed_offsets, g.shape)
            if cells is None:
                dropped = True
                break
            hits_obstacle = any(obstacle_mask[r, c] for r, c in cells)
            hits_green = any(green_mask[r, c] for r, c in cells)
            if not hits_obstacle and not hits_green:
                for r, c in cells:
                    green_mask[r, c] = True
                    working[r, c] = GREEN
                placed = True
                final_cells = cells
                break
            if not deflect_blocked:
                dropped = True
                break
            shift += 1

        placements.append(
            {
                "origin": origin,
                "obstructed_by_obstacle": obstructed_by_obstacle,
                "shift": shift if placed else None,
                "placed": placed,
                "dropped": dropped,
                "cells": final_cells,
            }
        )

    components = _components(green_mask)
    area_by_cell: dict[tuple[int, int], int] = {}
    out = working.copy()
    for comp in components:
        area = int(len(comp))
        color = BLUE if area == seed_area else RED
        for r, c in comp:
            out[r, c] = color
            area_by_cell[(int(r), int(c))] = area

    for placement in placements:
        cells = placement["cells"]
        placement["component_area"] = None if cells is None else area_by_cell[cells[0]]

    return out, placements, seed_area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the top-left seed regularly, deflect blocked copies rightward, then recolor by final 4-connectivity.

    Input structure:
    - The grid is mostly black background with one green seed component touching (0,0).
    - The seed is the shape to replicate on a row/column stride of 4.
    - Grey solid rectangles are fixed obstacles and remain unchanged.

    Rule:
    - Read the green component containing (0,0) as the seed shape.
    - For every valid tiling origin in left-to-right, top-to-bottom order, try to place one copy.
    - If the copy would overlap a grey obstacle or an already placed green copy, shift it right one column at a time until it fits or exits the grid; dropped copies vanish.
    - After all copies are placed, recolor each green 4-connected component: seed-sized components become blue, larger merged components become red.

    Key insight:
    - Obstacles do not merely block copies; they change where copies land, and those shifts can create new 4-connected mergers whose topology determines the final color.
    """
    out, _placements, _seed_area = _simulate_tiling(np.asarray(grid, dtype=int), deflect_blocked=True)
    return out


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return _interval_gap(a[0], a[2], b[0], b[2]) + _interval_gap(a[1], a[3], b[1], b[3])


def _rect_bbox(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, hh, ww = rect
    return (r0, c0, r0 + hh - 1, c0 + ww - 1)


def _expand_bbox(
    box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _can_place_rect(
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
    existing_rects: list[tuple[int, int, int, int]],
    forbidden_boxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = shape
    r0, c0, hh, ww = rect
    if r0 <= 0 or c0 <= 0 or r0 + hh >= h or c0 + ww >= w:
        return False
    box = _rect_bbox(rect)
    for other in existing_rects:
        if _bbox_distance(box, _rect_bbox(other)) < 2:
            return False
    for forbidden in forbidden_boxes:
        if _bbox_distance(box, forbidden) < 2:
            return False
    return True


def _build_input(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[0:2, 0:2] = GREEN
    for r0, c0, hh, ww in rects:
        grid[r0 : r0 + hh, c0 : c0 + ww] = GREY
    return grid


def _zone_for_merge(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = origin
    return _expand_bbox((r0, c0, r0 + 1, c0 + 5), shape)


def _zone_for_isolated(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = origin
    return _expand_bbox((r0, c0 - 1, r0 + 1, c0 + 2), shape)


def _zone_for_drop(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = origin
    return _expand_bbox((r0, c0, r0 + 1, c0 + 1), shape)


def _choose_interaction_origins(
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    h, w = shape
    rows = [r for r in range(0, h, SEED_STRIDE) if r + 1 < h]
    cols = [c for c in range(0, w, SEED_STRIDE) if c + 1 < w]
    if len(rows) < 3 or len(cols) < 4:
        raise AssertionError("grid too small for the required tiling interactions")

    seed_box = (0, 0, 1, 1)
    merge_candidates = [(r, c) for r in rows[:-1] for c in cols[:-1] if r <= h - 4]
    isolated_candidates = [(r, c) for r in rows[1:-1] for c in cols[1:] if r <= h - 3]
    random.shuffle(merge_candidates)
    random.shuffle(isolated_candidates)

    for merge_origin in merge_candidates:
        merge_zone = _zone_for_merge(merge_origin, shape)
        if _bbox_distance(merge_zone, seed_box) < 2:
            continue
        for isolated_origin in isolated_candidates:
            isolated_zone = _zone_for_isolated(isolated_origin, shape)
            if _bbox_distance(isolated_zone, seed_box) < 2:
                continue
            if _bbox_distance(merge_zone, isolated_zone) < 2:
                continue
            drop_candidates = [None]
            last_col = cols[-1]
            extra = [(r, last_col) for r in rows[:-1] if r <= h - 4]
            random.shuffle(extra)
            for candidate in extra:
                drop_zone = _zone_for_drop(candidate, shape)
                if (
                    _bbox_distance(drop_zone, seed_box) >= 2
                    and _bbox_distance(drop_zone, merge_zone) >= 2
                    and _bbox_distance(drop_zone, isolated_zone) >= 2
                ):
                    drop_candidates.append(candidate)
            use_drop = random.random() < 0.4 and len(drop_candidates) > 1
            if use_drop:
                return merge_origin, isolated_origin, random.choice(drop_candidates[1:])
            return merge_origin, isolated_origin, None
    raise RuntimeError("failed to choose interaction origins")


def _designed_rects(
    shape: tuple[int, int],
    merge_origin: tuple[int, int],
    isolated_origin: tuple[int, int],
    drop_origin: tuple[int, int] | None,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    h, _w = shape
    rects: list[tuple[int, int, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    merge_h = 3 if merge_origin[0] <= h - 5 and random.random() < 0.5 else 2
    merge_rect = (merge_origin[0] + 1, merge_origin[1], merge_h, 2)
    rects.append(merge_rect)
    reserved.append(_zone_for_merge(merge_origin, shape))

    isolated_h = 3 if isolated_origin[0] <= h - 4 and random.random() < 0.5 else 2
    isolated_rect = (isolated_origin[0], isolated_origin[1] - 1, isolated_h, 2)
    rects.append(isolated_rect)
    reserved.append(_zone_for_isolated(isolated_origin, shape))

    if drop_origin is not None:
        drop_h = 3 if drop_origin[0] <= h - 5 and random.random() < 0.5 else 2
        drop_rect = (drop_origin[0] + 1, drop_origin[1], drop_h, 2)
        rects.append(drop_rect)
        reserved.append(_zone_for_drop(drop_origin, shape))

    return rects, reserved


def _add_extra_rects(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    reserved: list[tuple[int, int, int, int]],
    target_count: int,
) -> bool:
    seed_box = (0, 0, 1, 1)
    forbidden = [seed_box, *reserved]
    attempts = 0
    while len(rects) < target_count and attempts < 300:
        attempts += 1
        hh = random.randint(2, 3)
        ww = random.randint(2, 3)
        r0 = random.randint(1, shape[0] - hh - 1)
        c0 = random.randint(1, shape[1] - ww - 1)
        rect = (r0, c0, hh, ww)
        if _can_place_rect(rect, shape, rects, forbidden):
            rects.append(rect)
    return len(rects) == target_count


def _shortcut_no_deflection(grid: np.ndarray) -> np.ndarray:
    out, _placements, _seed_area = _simulate_tiling(grid, deflect_blocked=False)
    return out


def _validate_generated(grid: np.ndarray, expect_drop: bool) -> bool:
    out, placements, seed_area = _simulate_tiling(grid, deflect_blocked=True)
    if seed_area != 4:
        return False
    if int(np.sum(grid == GREEN)) != 4:
        return False

    obstructed = [p for p in placements if p["obstructed_by_obstacle"]]
    if len(obstructed) < 2:
        return False

    shifted_merged = [
        p
        for p in obstructed
        if p["placed"] and p["shift"] is not None and p["shift"] > 0 and p["component_area"] > seed_area
    ]
    shifted_isolated = [
        p
        for p in obstructed
        if p["placed"] and p["shift"] is not None and p["shift"] > 0 and p["component_area"] == seed_area
    ]
    if not shifted_merged or not shifted_isolated:
        return False

    if expect_drop and not any(p["dropped"] for p in obstructed):
        return False

    colors = set(map(int, np.unique(out)))
    if not {BLUE, RED, GREY}.issubset(colors):
        return False
    if GREEN in colors:
        return False

    shortcut = _shortcut_no_deflection(grid)
    if np.array_equal(out, shortcut):
        return False

    red_components = _components(out == RED)
    blue_components = _components(out == BLUE)
    if not red_components or not blue_components:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_SIZE_RANGE),
        random.randint(*GEN_SIZE_RANGE),
    )
    try:
        merge_origin, isolated_origin, drop_origin = _choose_interaction_origins(shape)
    except RuntimeError:
        return None

    rects, reserved = _designed_rects(shape, merge_origin, isolated_origin, drop_origin)
    target_obstacles = random.randint(*GEN_OBSTACLE_COUNT_RANGE)
    if not _add_extra_rects(shape, rects, reserved, target_obstacles):
        return None

    grid = _build_input(shape, rects)
    expect_drop = drop_origin is not None
    if not _validate_generated(grid, expect_drop):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
