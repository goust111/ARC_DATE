from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import itertools
import random

import numpy as np


BG = 8
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
CORE_DIMS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
)
MAX_RAY_LEN = 3
MAX_SHAPE_AREA = 15
MIN_OBJECTS = 6
MAX_OBJECTS = 10
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 600


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_cells(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not cells:
        return ()
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _bbox_from_norm(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _is_boundary_cell(cells: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    r, c = cell
    return any((nr, nc) not in cells for nr, nc in _neighbors4(r, c))


def _in_any_2x2(cells: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    r, c = cell
    for dr in (-1, 0):
        for dc in (-1, 0):
            block = {
                (r + dr, c + dc),
                (r + dr + 1, c + dc),
                (r + dr, c + dc + 1),
                (r + dr + 1, c + dc + 1),
            }
            if block <= cells:
                return True
    return False


def _is_removable(cells: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    if len(cells) <= 1:
        return False
    if not _is_boundary_cell(cells, cell):
        return False
    if _in_any_2x2(cells, cell):
        return False
    reduced = set(cells)
    reduced.remove(cell)
    return _is_connected(reduced)


@lru_cache(maxsize=None)
def _skeleton_signature(norm_cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = set(norm_cells)
    while True:
        removed = False
        for cell in sorted(cells):
            if _is_removable(cells, cell):
                cells.remove(cell)
                removed = True
                break
        if not removed:
            return _normalize_cells(cells)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            colors = {int(grid[rr, cc]) for rr, cc in coords}
            assert len(colors) == 1, "foreground objects must be monochrome"
            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "coords": arr,
                    "color": color,
                    "area": int(arr.shape[0]),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _component_signature(component: dict[str, object]) -> tuple[tuple[int, int], ...]:
    coords = component["coords"]
    cells = {(int(r), int(c)) for r, c in coords}
    return _skeleton_signature(_normalize_cells(cells))


def _group_components(components: list[dict[str, object]]) -> dict[tuple[tuple[int, int], ...], list[dict[str, object]]]:
    groups: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
    for component in components:
        groups[_component_signature(component)].append(component)
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group shapes by the skeleton left after pruning thin removable protrusions,
    then recolor each group with the color of its unique largest member.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    components = _find_components(g, bg)
    groups = _group_components(components)

    out = g.copy()
    for group in groups.values():
        max_area = max(int(component["area"]) for component in group)
        winners = [component for component in group if int(component["area"]) == max_area]
        assert len(winners) == 1, "each skeleton group must have a unique largest shape"
        winner_color = int(winners[0]["color"])
        for component in group:
            coords = component["coords"]
            out[coords[:, 0], coords[:, 1]] = winner_color
    return out


def _rect_cells(height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(height) for c in range(width)}


@lru_cache(maxsize=None)
def _core_variants(height: int, width: int) -> dict[int, tuple[tuple[tuple[int, int], ...], ...]]:
    core = _rect_cells(height, width)
    core_norm = _normalize_cells(core)
    core_sig = _skeleton_signature(core_norm)
    core_area = height * width
    variants: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    seen: set[tuple[tuple[int, int], ...]] = set()

    families = (
        (
            ((0, 0), (-1, 0)),
            ((0, width - 1), (0, 1)),
            ((height - 1, width - 1), (1, 0)),
            ((height - 1, 0), (0, -1)),
        ),
        (
            ((0, width - 1), (-1, 0)),
            ((height - 1, width - 1), (0, 1)),
            ((height - 1, 0), (1, 0)),
            ((0, 0), (0, -1)),
        ),
    )

    for family in families:
        for lengths in itertools.product(range(MAX_RAY_LEN + 1), repeat=4):
            cells = set(core)
            for ((sr, sc), (dr, dc)), ray_len in zip(family, lengths, strict=True):
                for step in range(1, ray_len + 1):
                    cells.add((sr + dr * step, sc + dc * step))
            area = len(cells)
            if area > MAX_SHAPE_AREA:
                continue
            norm = _normalize_cells(cells)
            if norm in seen:
                continue
            if _skeleton_signature(norm) != core_sig:
                continue
            seen.add(norm)
            variants[area - core_area].append(norm)

    return {extra: tuple(shapes) for extra, shapes in variants.items()}


def _sample_group_sizes(total_objects: int) -> list[int]:
    valid_groups = [
        group_count
        for group_count in range(2, 5)
        if 2 * group_count <= total_objects <= 4 * group_count
    ]
    group_count = random.choice(valid_groups)
    sizes = [2] * group_count
    for _ in range(total_objects - 2 * group_count):
        choices = [idx for idx, size in enumerate(sizes) if size < 4]
        sizes[random.choice(choices)] += 1
    random.shuffle(sizes)
    return sizes


def _shape_record(group_id: int, color: int, cells: tuple[tuple[int, int], ...]) -> dict[str, object]:
    height, width = _bbox_from_norm(cells)
    return {
        "group_id": group_id,
        "color": color,
        "cells": cells,
        "height": height,
        "width": width,
        "bbox_area": height * width,
    }


def _sample_object_colors(group_sizes: list[int]) -> list[list[int]]:
    if sum(group_sizes) <= len(PALETTE):
        colors = list(random.sample(PALETTE, sum(group_sizes)))
        out: list[list[int]] = []
        cursor = 0
        for size in group_sizes:
            out.append(colors[cursor : cursor + size])
            cursor += size
        return out

    winner_colors = list(random.sample(PALETTE, len(group_sizes)))
    out = []
    for winner_color, size in zip(winner_colors, group_sizes, strict=True):
        choices = [winner_color]
        others = [color for color in PALETTE if color != winner_color]
        choices.extend(random.sample(others, size - 1))
        random.shuffle(choices)
        out.append(choices)
    return out


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    total_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    group_sizes = _sample_group_sizes(total_objects)

    cores = random.sample(CORE_DIMS, len(group_sizes))
    color_groups = _sample_object_colors(group_sizes)

    objects: list[dict[str, object]] = []
    winner_colors: dict[int, int] = {}
    for group_id, ((height, width), group_size, colors) in enumerate(
        zip(cores, group_sizes, color_groups, strict=True)
    ):
        variants = _core_variants(height, width)
        extra_choices = sorted(variants)
        if len(extra_choices) < group_size:
            return None
        chosen_extras = sorted(random.sample(extra_choices, group_size))
        winner_extra = chosen_extras[-1]
        winner_color = random.choice(colors)
        winner_colors[group_id] = winner_color

        other_colors = [color for color in colors if color != winner_color]
        random.shuffle(other_colors)
        for extra, color in zip(chosen_extras[:-1], other_colors, strict=True):
            shape = random.choice(variants[extra])
            objects.append(_shape_record(group_id, color, shape))

        winner_shape = random.choice(variants[winner_extra])
        objects.append(_shape_record(group_id, winner_color, winner_shape))

    shuffled_objects = objects[:]
    random.shuffle(shuffled_objects)

    max_h = max(int(obj["height"]) for obj in shuffled_objects)
    max_w = max(int(obj["width"]) for obj in shuffled_objects)
    grid_h = random.randint(max(18, max_h + 4), 26)
    grid_w = random.randint(max(18, max_w + 4), 28)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = grid.copy()
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    placed_per_group: dict[int, int] = defaultdict(int)
    for obj in sorted(shuffled_objects, key=lambda item: int(item["bbox_area"]), reverse=True):
        shape_cells = obj["cells"]
        shape_h = int(obj["height"])
        shape_w = int(obj["width"])
        abs_cells: list[tuple[int, int]] | None = None

        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(0, grid_h - shape_h)
            left = random.randint(0, grid_w - shape_w)
            candidate = [(top + r, left + c) for r, c in shape_cells]
            if any(blocked[r, c] for r, c in candidate):
                continue
            abs_cells = candidate
            break

        if abs_cells is None:
            return None

        color = int(obj["color"])
        winner_color = int(winner_colors[int(obj["group_id"])])
        for r, c in abs_cells:
            grid[r, c] = color
            expected[r, c] = winner_color
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        blocked[nr, nc] = True
        placed_per_group[int(obj["group_id"])] += 1

    if any(placed_per_group[group_id] != size for group_id, size in enumerate(group_sizes)):
        return None

    if np.array_equal(grid, expected):
        return None

    components = _find_components(grid, BG)
    if len(components) != total_objects:
        return None
    groups = _group_components(components)
    if sorted(len(group) for group in groups.values()) != sorted(group_sizes):
        return None
    for group in groups.values():
        areas = [int(component["area"]) for component in group]
        if len(group) < 2 or areas.count(max(areas)) != 1:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _build_scene()
        if scene is None:
            continue
        grid, _expected = scene
        return grid
    raise RuntimeError("failed to generate a valid input")
