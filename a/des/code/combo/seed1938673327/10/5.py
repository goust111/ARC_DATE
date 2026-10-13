from __future__ import annotations

from collections import deque

import numpy as np
from numpy.random import default_rng


BG = 0
RED = 2
YELLOW = 4
INPUT_COLORS = (1, 3, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 13
GRID_MAX = 22
HUB_SIZE_MIN = 4
HUB_SIZE_MAX = 7
BAR_LEN_MIN = 2
BAR_LEN_MAX = 4
PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray) -> dict[int, frozenset[tuple[int, int]]]:
    objects: dict[int, frozenset[tuple[int, int]]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        comps = _components(grid == color)
        assert len(comps) == 1, "each color must define exactly one 4-connected object"
        objects[color] = frozenset(comps[0])
    return objects


def _build_adjacency(
    grid: np.ndarray,
) -> tuple[dict[int, frozenset[tuple[int, int]]], dict[int, set[int]]]:
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    adj = {color: set() for color in objects}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG:
                continue
            if r + 1 < h:
                other = int(g[r + 1, c])
                if other not in (BG, color):
                    adj[color].add(other)
                    adj[other].add(color)
            if c + 1 < w:
                other = int(g[r, c + 1])
                if other not in (BG, color):
                    adj[color].add(other)
                    adj[other].add(color)
    return objects, adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the highest-degree object in the foreground adjacency graph.

    Objects are monochromatic 4-connected components on a black background. Two objects
    are adjacent when one has a pixel that is 4-neighboring a pixel of the other.
    The unique object with maximum graph degree becomes yellow, and all of its graph
    neighbors become red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects, adj = _build_adjacency(g)
    assert objects, "expected at least one foreground object"

    degrees = {color: len(neighbors) for color, neighbors in adj.items()}
    best_degree = max(degrees.values())
    winners = [color for color, degree in degrees.items() if degree == best_degree]
    assert len(winners) == 1, "highest-degree object must be unique"

    hub = winners[0]
    out = g.copy()
    out[g == hub] = YELLOW
    for neighbor in adj[hub]:
        out[g == neighbor] = RED
    return out


def _rectangle_cells(top: int, left: int, height: int, width: int) -> frozenset[tuple[int, int]]:
    return frozenset((top + r, left + c) for r in range(height) for c in range(width))


def _bar_cells(root_r: int, root_c: int, dr: int, dc: int, length: int) -> frozenset[tuple[int, int]]:
    return frozenset((root_r + i * dr, root_c + i * dc) for i in range(length))


def _bbox_size(cells: frozenset[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _translate(cells: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def _local_shape(rng: np.random.Generator) -> frozenset[tuple[int, int]]:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        length = int(rng.integers(BAR_LEN_MIN, BAR_LEN_MAX + 1))
        if rng.random() < 0.5:
            return frozenset((0, c) for c in range(length))
        return frozenset((r, 0) for r in range(length))
    if kind == 1:
        height = int(rng.integers(1, 4))
        width = int(rng.integers(1, 4))
        if height == 1 and width == 1:
            width = 2
        return _rectangle_cells(0, 0, height, width)

    height = int(rng.integers(2, 5))
    width = int(rng.integers(2, 5))
    cells = {(r, 0) for r in range(height)}
    cells.update((height - 1, c) for c in range(width))
    if rng.random() < 0.5:
        cells = {(r, width - 1 - c) for r, c in cells}
    if rng.random() < 0.5:
        cells = {(height - 1 - r, c) for r, c in cells}
    return frozenset(cells)


def _touching_ids(
    cells: frozenset[tuple[int, int]],
    occupancy: dict[tuple[int, int], int],
) -> set[int]:
    touched: set[int] = set()
    for r, c in cells:
        for dr, dc in DIRS4:
            obj_id = occupancy.get((r + dr, c + dc))
            if obj_id is not None:
                touched.add(obj_id)
    return touched


def _can_place(
    cells: frozenset[tuple[int, int]],
    height: int,
    width: int,
    occupancy: dict[tuple[int, int], int],
    allowed_touch: set[int],
    require_touch: bool,
) -> bool:
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if (r, c) in occupancy:
            return False
    touched = _touching_ids(cells, occupancy)
    if require_touch and not touched:
        return False
    return touched.issubset(allowed_touch)


def _commit_object(
    objects: list[dict[str, object]],
    occupancy: dict[tuple[int, int], int],
    color: int,
    cells: frozenset[tuple[int, int]],
) -> int:
    obj_id = len(objects)
    objects.append({"color": int(color), "cells": cells})
    for cell in cells:
        occupancy[cell] = obj_id
    return obj_id


def _place_isolated_shape(
    rng: np.random.Generator,
    height: int,
    width: int,
    occupancy: dict[tuple[int, int], int],
    objects: list[dict[str, object]],
    color: int,
) -> bool:
    for _ in range(PLACEMENT_ATTEMPTS):
        local = _local_shape(rng)
        shape_h, shape_w = _bbox_size(local)
        if shape_h > height or shape_w > width:
            continue
        top = int(rng.integers(0, height - shape_h + 1))
        left = int(rng.integers(0, width - shape_w + 1))
        cells = _translate(local, top, left)
        if _can_place(cells, height, width, occupancy, allowed_touch=set(), require_touch=False):
            _commit_object(objects, occupancy, color, cells)
            return True
    return False


def _place_attached_bar(
    rng: np.random.Generator,
    height: int,
    width: int,
    occupancy: dict[tuple[int, int], int],
    objects: list[dict[str, object]],
    base_id: int,
    color: int,
) -> bool:
    base_cells = tuple(objects[base_id]["cells"])
    candidates: list[tuple[int, int, int, int]] = []
    for r, c in base_cells:
        for dr, dc in DIRS4:
            root_r = r + dr
            root_c = c + dc
            if (root_r, root_c) not in objects[base_id]["cells"]:
                candidates.append((root_r, root_c, dr, dc))
    if not candidates:
        return False

    for idx in rng.permutation(len(candidates)):
        root_r, root_c, dr, dc = candidates[int(idx)]
        for length in rng.permutation(np.arange(BAR_LEN_MIN, BAR_LEN_MAX + 1)):
            cells = _bar_cells(root_r, root_c, dr, dc, int(length))
            if not _can_place(
                cells,
                height,
                width,
                occupancy,
                allowed_touch={base_id},
                require_touch=True,
            ):
                continue
            if _touching_ids(cells, occupancy) != {base_id}:
                continue
            _commit_object(objects, occupancy, color, cells)
            return True
    return False


def _hub_slots(top: int, left: int, height: int, width: int) -> list[tuple[int, int, int, int]]:
    slots: list[tuple[int, int, int, int]] = []
    for c in range(left, left + width, 2):
        slots.append((top, c, -1, 0))
        slots.append((top + height - 1, c, 1, 0))
    for r in range(top, top + height, 2):
        slots.append((r, left, 0, -1))
        slots.append((r, left + width - 1, 0, 1))
    return slots


def _render_grid(height: int, width: int, objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["cells"]:
            grid[r, c] = color
    return grid


def _pattern_sizes(rng: np.random.Generator, extras: int, hub_degree: int) -> list[int]:
    sizes: list[int] = []
    remaining = extras
    while remaining > 0:
        options = [1]
        if remaining >= 2:
            options.append(2)
        if remaining >= 3 and hub_degree >= 3:
            options.append(3)
        size = int(rng.choice(options))
        sizes.append(size)
        remaining -= size
    return sizes


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    max_hub_h = min(HUB_SIZE_MAX, height - 2 * BAR_LEN_MAX)
    max_hub_w = min(HUB_SIZE_MAX, width - 2 * BAR_LEN_MAX)
    if max_hub_h < HUB_SIZE_MIN or max_hub_w < HUB_SIZE_MIN:
        return None

    hub_h = int(rng.integers(HUB_SIZE_MIN, max_hub_h + 1))
    hub_w = int(rng.integers(HUB_SIZE_MIN, max_hub_w + 1))
    top = int(rng.integers(BAR_LEN_MAX, height - hub_h - BAR_LEN_MAX + 1))
    left = int(rng.integers(BAR_LEN_MAX, width - hub_w - BAR_LEN_MAX + 1))

    slots = _hub_slots(top, left, hub_h, hub_w)
    max_degree = min(len(slots), len(INPUT_COLORS) - 1, 6)
    if max_degree < 2:
        return None

    hub_degree = int(rng.integers(2, max_degree + 1))
    total_objects = int(rng.integers(hub_degree + 1, len(INPUT_COLORS) + 1))
    extras = total_objects - hub_degree - 1

    colors = [int(v) for v in rng.permutation(np.array(INPUT_COLORS))]
    occupancy: dict[tuple[int, int], int] = {}
    objects: list[dict[str, object]] = []

    hub_color = colors[0]
    hub_id = _commit_object(
        objects,
        occupancy,
        hub_color,
        _rectangle_cells(top, left, hub_h, hub_w),
    )

    slot_ids = rng.choice(len(slots), size=hub_degree, replace=False)
    for i, slot_idx in enumerate(slot_ids, start=1):
        edge_r, edge_c, dr, dc = slots[int(slot_idx)]
        root_r = edge_r + dr
        root_c = edge_c + dc
        length = int(rng.integers(BAR_LEN_MIN, BAR_LEN_MAX + 1))
        cells = _bar_cells(root_r, root_c, dr, dc, length)
        if not _can_place(
            cells,
            height,
            width,
            occupancy,
            allowed_touch={hub_id},
            require_touch=True,
        ):
            return None
        if _touching_ids(cells, occupancy) != {hub_id}:
            return None
        _commit_object(objects, occupancy, colors[i], cells)

    color_index = 1 + hub_degree
    for size in _pattern_sizes(rng, extras, hub_degree):
        if size == 1:
            if not _place_isolated_shape(
                rng,
                height,
                width,
                occupancy,
                objects,
                colors[color_index],
            ):
                return None
        elif size == 2:
            if not _place_isolated_shape(
                rng,
                height,
                width,
                occupancy,
                objects,
                colors[color_index],
            ):
                return None
            if not _place_attached_bar(
                rng,
                height,
                width,
                occupancy,
                objects,
                len(objects) - 1,
                colors[color_index + 1],
            ):
                return None
        else:
            if not _place_isolated_shape(
                rng,
                height,
                width,
                occupancy,
                objects,
                colors[color_index],
            ):
                return None
            if not _place_attached_bar(
                rng,
                height,
                width,
                occupancy,
                objects,
                len(objects) - 1,
                colors[color_index + 1],
            ):
                return None
            if not _place_attached_bar(
                rng,
                height,
                width,
                occupancy,
                objects,
                len(objects) - 1,
                colors[color_index + 2],
            ):
                return None
        color_index += size

    grid = _render_grid(height, width, objects)
    found_objects, adj = _build_adjacency(grid)
    if len(found_objects) != total_objects or hub_color not in adj:
        return None
    degrees = {color: len(neighbors) for color, neighbors in adj.items()}
    winners = [color for color, degree in degrees.items() if degree == max(degrees.values())]
    if winners != [hub_color]:
        return None
    if degrees[hub_color] != hub_degree:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
