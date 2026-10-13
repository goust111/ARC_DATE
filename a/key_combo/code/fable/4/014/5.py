from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OBJECT_DIMS = tuple(
    (h, w)
    for h in range(1, 5)
    for w in range(1, 5)
    if 3 <= h * w <= 12
)
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _make_object(color: int, coords: tuple[tuple[int, int], ...]) -> dict:
    ordered = tuple(sorted((int(r), int(c)) for r, c in coords))
    return {
        "color": int(color),
        "coords": ordered,
        "cells": frozenset(ordered),
        "bbox": _bbox(ordered),
    }


def _rect_object(color: int, top: int, left: int, height: int, width: int) -> dict:
    coords = tuple((r, c) for r in range(top, top + height) for c in range(left, left + width))
    return _make_object(color, coords)


def _find_components(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    cell_to_component: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            comp = _make_object(color, tuple(coords))
            comp_idx = len(components)
            components.append(comp)
            for cell in comp["coords"]:
                cell_to_component[cell] = comp_idx

    return components, cell_to_component


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    components, cell_to_component = _find_components(g)

    seed_matches: list[tuple[int, int, tuple[int, int]]] = []
    for comp_idx, comp in enumerate(components):
        if comp["color"] != YELLOW or len(comp["coords"]) != 1:
            continue
        seed_cell = comp["coords"][0]
        adjacent_components = {
            cell_to_component[(nr, nc)]
            for nr, nc in _neighbors4(*seed_cell)
            if 0 <= nr < g.shape[0]
            and 0 <= nc < g.shape[1]
            and int(g[nr, nc]) != BLACK
            and (nr, nc) in cell_to_component
            and cell_to_component[(nr, nc)] != comp_idx
        }
        if len(adjacent_components) == 1:
            seed_matches.append((comp_idx, next(iter(adjacent_components)), seed_cell))

    assert len(seed_matches) == 1, "expected one yellow single-pixel seed marker"
    seed_comp_idx, seed_target_comp_idx, seed_cell = seed_matches[0]

    objects: list[dict] = []
    component_to_object: dict[int, int] = {}
    for comp_idx, comp in enumerate(components):
        if comp_idx == seed_comp_idx:
            continue
        component_to_object[comp_idx] = len(objects)
        objects.append(comp)

    assert seed_target_comp_idx in component_to_object, "seed must touch exactly one object"
    seed_object_idx = component_to_object[seed_target_comp_idx]
    return objects, seed_cell, seed_object_idx


def _draw_halo(out: np.ndarray, obj: dict, halo_color: int) -> set[tuple[int, int]]:
    h, w = out.shape
    r0, c0, r1, c1 = obj["bbox"]
    new_cells: set[tuple[int, int]] = set()

    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if (r, c) in obj["cells"]:
                continue
            if int(out[r, c]) != BLACK:
                continue
            out[r, c] = int(halo_color)
            new_cells.add((r, c))

    return new_cells


def _touches_cells(obj: dict, halo_cells: set[tuple[int, int]]) -> bool:
    if not halo_cells:
        return False
    for r, c in obj["coords"]:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in halo_cells:
                return True
    return False


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects, seed_cell, seed_object_idx = _parse_scene(g)
    out = g.copy()

    ignited: set[int] = set()
    order: list[int] = []
    queue: deque[tuple[int, int]] = deque([(seed_object_idx, YELLOW)])

    while queue:
        obj_idx, halo_color = queue.popleft()
        assert obj_idx not in ignited, "propagation chain must not revisit objects"
        ignited.add(obj_idx)
        order.append(obj_idx)

        new_halo = _draw_halo(out, objects[obj_idx], halo_color)
        candidates = [
            next_idx
            for next_idx, next_obj in enumerate(objects)
            if next_idx not in ignited and _touches_cells(next_obj, new_halo)
        ]
        assert len(candidates) <= 1, "propagation must ignite at most one new object per round"
        if candidates:
            queue.append((candidates[0], objects[obj_idx]["color"]))

    return out, {
        "objects": objects,
        "seed_cell": seed_cell,
        "seed_object_idx": seed_object_idx,
        "ignited_order": order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Start from the yellow seed marker and spread bbox halos along a unique ignition chain.

    Input structure:
    - The grid has a black background with several separated solid monochrome objects.
    - Exactly one yellow single pixel is orthogonally adjacent to one object and serves as the seed marker.
    - Objects keep their own colors, while some later outputs may add colored halo cells on background squares.

    Rule:
    - The seed-adjacent object ignites first and receives a halo whose color is yellow.
    - A halo is made by expanding that object's tight bounding box by one cell in every direction, clipping to the grid, and recoloring only background cells in that region.
    - Whenever an unignited object's pixels are orthogonally adjacent to the newly drawn halo, that object ignites next and its halo color becomes the igniter object's body color; continue until no new object is reached.

    Key insight:
    - The halos are not decoration but the transmission medium, and each halo color records which earlier object ignited the current one.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _can_place(obj: dict, occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in obj["coords"]:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _can_ignite(source: dict, target: dict, blocked: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = source["bbox"]
    blocked_cells = set(blocked) | set(target["coords"])
    halo_cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if (r, c) in source["cells"] or (r, c) in blocked_cells:
                continue
            halo_cells.add((r, c))
    return _touches_cells(target, halo_cells)


def _rand_margin_start(limit: int, size: int) -> int:
    if limit - size >= 2:
        return random.randint(1, limit - size - 1)
    return random.randint(0, limit - size)


def _place_first_object(shape: tuple[int, int], color: int, force_edge: bool) -> dict:
    h, w = shape
    height, width = random.choice(OBJECT_DIMS)

    if not force_edge:
        top = _rand_margin_start(h, height)
        left = _rand_margin_start(w, width)
        return _rect_object(color, top, left, height, width)

    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        top = 0
        left = _rand_margin_start(w, width)
    elif side == "bottom":
        top = h - height
        left = _rand_margin_start(w, width)
    elif side == "left":
        top = _rand_margin_start(h, height)
        left = 0
    else:
        top = _rand_margin_start(h, height)
        left = w - width
    return _rect_object(color, top, left, height, width)


def _candidate_offsets(prev: dict, height: int, width: int, shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = prev["bbox"]

    if direction == "right":
        left = c1 + 2
        if left + width > w:
            return []
        top_lo = max(0, r0 - height)
        top_hi = min(h - height, r1 + 1)
        return [(top, left) for top in range(top_lo, top_hi + 1)]
    if direction == "left":
        left = c0 - width - 1
        if left < 0:
            return []
        top_lo = max(0, r0 - height)
        top_hi = min(h - height, r1 + 1)
        return [(top, left) for top in range(top_lo, top_hi + 1)]
    if direction == "down":
        top = r1 + 2
        if top + height > h:
            return []
        left_lo = max(0, c0 - width)
        left_hi = min(w - width, c1 + 1)
        return [(top, left) for left in range(left_lo, left_hi + 1)]

    top = r0 - height - 1
    if top < 0:
        return []
    left_lo = max(0, c0 - width)
    left_hi = min(w - width, c1 + 1)
    return [(top, left) for left in range(left_lo, left_hi + 1)]


def _place_chain_object(
    prev: dict,
    earlier: list[dict],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    color: int,
) -> dict | None:
    dims = list(OBJECT_DIMS)
    random.shuffle(dims)
    for height, width in dims:
        directions = ["up", "down", "left", "right"]
        random.shuffle(directions)

        for direction in directions:
            candidates = _candidate_offsets(prev, height, width, shape, direction)
            random.shuffle(candidates)
            for top, left in candidates:
                obj = _rect_object(color, top, left, height, width)
                if not _can_place(obj, occupied, shape):
                    continue
                if not _can_ignite(prev, obj, occupied, shape):
                    continue
                if any(_can_ignite(other, obj, occupied, shape) for other in earlier[:-1]):
                    continue
                return obj
    return None


def _seed_candidates(first_obj: dict, objects: list[dict], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cell_to_object: dict[tuple[int, int], int] = {}
    for obj_idx, obj in enumerate(objects):
        for cell in obj["coords"]:
            cell_to_object[cell] = obj_idx

    candidates: list[tuple[int, int]] = []
    for r, c in first_obj["coords"]:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_to_object:
                continue
            adjacent = {
                cell_to_object[(rr, cc)]
                for rr, cc in _neighbors4(nr, nc)
                if (rr, cc) in cell_to_object
            }
            if adjacent == {0}:
                candidates.append((nr, nc))
    return sorted(set(candidates))


def _paint_scene(shape: tuple[int, int], objects: list[dict], seed_cell: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for obj in objects:
        for r, c in obj["coords"]:
            grid[r, c] = obj["color"]
    grid[seed_cell] = YELLOW
    return grid


def _touches_border(obj: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = obj["bbox"]
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _attempt_input() -> np.ndarray | None:
    chain_len = random.randint(2, 5)
    unreachable_len = random.randint(0, min(2, 6 - chain_len))
    total_objects = chain_len + unreachable_len
    height = random.randint(max(16, 11 + 3 * chain_len), 30)
    width = random.randint(max(16, 11 + 3 * chain_len), 30)
    shape = (height, width)
    want_clipped = random.random() < 0.4

    colors = random.sample(list(OBJECT_COLORS), total_objects)
    objects = [_place_first_object(shape, colors[0], want_clipped)]
    occupied = set(objects[0]["coords"])

    for color in colors[1:chain_len]:
        obj = _place_chain_object(objects[-1], objects, occupied, shape, color)
        if obj is None:
            return None
        objects.append(obj)
        occupied.update(obj["coords"])

    for color in colors[chain_len:]:
        placed = None
        for _ in range(60):
            height_i, width_i = random.choice(OBJECT_DIMS)
            top = random.randint(0, height - height_i)
            left = random.randint(0, width - width_i)
            candidate = _rect_object(color, top, left, height_i, width_i)
            if not _can_place(candidate, occupied, shape):
                continue
            if any(_can_ignite(src, candidate, occupied, shape) for src in objects[:chain_len]):
                continue
            placed = candidate
            break
        if placed is None:
            return None
        objects.append(placed)
        occupied.update(placed["coords"])

    seed_options = _seed_candidates(objects[0], objects, shape)
    if not seed_options:
        return None
    seed_cell = random.choice(seed_options)
    grid = _paint_scene(shape, objects, seed_cell)

    try:
        _out, trace = _simulate(grid)
    except AssertionError:
        return None

    ignited_colors = {trace["objects"][obj_idx]["color"] for obj_idx in trace["ignited_order"]}
    planned_chain_colors = set(colors[:chain_len])
    if len(trace["ignited_order"]) != chain_len or ignited_colors != planned_chain_colors:
        return None
    if trace["objects"][trace["seed_object_idx"]]["color"] != colors[0]:
        return None
    if want_clipped and not any(
        _touches_border(trace["objects"][obj_idx], shape) for obj_idx in trace["ignited_order"]
    ):
        return None
    if len(objects) > chain_len and len(trace["ignited_order"]) == len(objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid halo-chain input")
