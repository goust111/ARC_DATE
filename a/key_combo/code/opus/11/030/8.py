from __future__ import annotations

import random
from collections import deque

import numpy as np


PALETTE = (1, 2, 3, 4, 6)
AREA_CHOICES = (3, 4, 5, 6, 7)
GRID_SIDE_RANGE = (11, 15)
COMPONENT_COUNT_CHOICES = (3, 4, 5)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 300
TARGET_COUNT_RETRIES = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == 0:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                assert int(grid[rr, cc]) == color, "foreground regions must be monochromatic"
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert scattered monochrome regions into an area-sorted color histogram.

    Input structure:
    - The grid has black background and several separated non-black foreground regions.
    - Each foreground region is a single 4-connected monochromatic component.
    - Region colors are distinct, while region shapes and positions are arbitrary.

    Rule:
    - Detect every 4-connected non-black component and measure its pixel area.
    - Sort the components by area from small to large; areas must be unique.
    - Create a black output grid whose width is the component count and whose height is the largest area.
    - In column order, draw one bottom-aligned vertical bar per component using the component color and its area as the bar height.

    Key insight:
    - The output ignores where a region was and what shape it had; only the component's color and measured area survive, and sorting by area determines the left-to-right order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    assert 3 <= len(components) <= 5, "expected 3 to 5 foreground regions"

    areas = [int(comp["area"]) for comp in components]
    assert len(set(areas)) == len(areas), "component areas must be unique"

    ordered = sorted(components, key=lambda comp: int(comp["area"]))
    out_h = max(int(comp["area"]) for comp in ordered)
    out_w = len(ordered)
    out = np.zeros((out_h, out_w), dtype=int)

    for col, comp in enumerate(ordered):
        height = int(comp["area"])
        color = int(comp["color"])
        out[out_h - height :, col] = color

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _is_rectangle(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_bbox(shape)
    return h * w == len(shape)


def _random_irregular_shape(area: int) -> tuple[tuple[int, int], ...]:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            counts: dict[tuple[int, int], int] = {}
            for pos in frontier:
                counts[pos] = counts.get(pos, 0) + 1
            choices = list(counts)
            weights = [2 + counts[pos] for pos in choices]
            cells.add(random.choices(choices, weights=weights, k=1)[0])

        shape = _normalize_shape(cells)
        if _is_rectangle(shape):
            continue
        h, w = _shape_bbox(shape)
        if h == 1 or w == 1:
            continue
        return shape
    raise RuntimeError(f"failed to build irregular shape of area {area}")


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = list(shape)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    transformed = cells

    for _ in range(random.randrange(4)):
        transformed = [(c, max_r - r) for r, c in transformed]
        max_r = max(r for r, _ in transformed)
        max_c = max(c for _, c in transformed)

    if random.choice((False, True)):
        transformed = [(r, max_c - c) for r, c in transformed]

    return _normalize_shape(set(transformed))


def _can_place(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    for r, c in shape:
        rr = top + r
        cc = left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_shapes(
    shape_specs: list[tuple[int, int, tuple[tuple[int, int], ...]]],
    grid_shape: tuple[int, int],
) -> np.ndarray | None:
    h, w = grid_shape
    order = sorted(range(len(shape_specs)), key=lambda idx: len(shape_specs[idx][2]), reverse=True)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        area, color, shape = shape_specs[order[pos]]
        del area
        sh, sw = _shape_bbox(shape)
        positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(positions)
        for top, left in positions[:PLACE_ATTEMPTS]:
            if not _can_place(occupied, shape, top, left):
                continue
            for dr, dc in shape:
                rr = top + dr
                cc = left + dc
                occupied[rr, cc] = True
                grid[rr, cc] = color
            if search(pos + 1):
                return True
            for dr, dc in shape:
                rr = top + dr
                cc = left + dc
                occupied[rr, cc] = False
                grid[rr, cc] = 0
        return False

    if search(0):
        return grid
    return None


def _input_is_nondegenerate(grid: np.ndarray) -> bool:
    comps = _find_components(grid)
    if not (3 <= len(comps) <= 5):
        return False

    areas = [int(comp["area"]) for comp in comps]
    if len(set(areas)) != len(areas):
        return False

    colors = [int(comp["color"]) for comp in comps]
    if len(set(colors)) != len(colors):
        return False

    by_area = sorted(comps, key=lambda comp: int(comp["area"]))
    left_order = sorted(comps, key=lambda comp: (sum(c for _, c in comp["cells"]) / int(comp["area"]), int(comp["area"])))
    if [int(comp["area"]) for comp in left_order] == [int(comp["area"]) for comp in by_area]:
        return False

    bbox_areas = []
    heights = []
    widths = []
    for comp in by_area:
        r0, c0, r1, c1 = comp["bbox"]
        heights.append(r1 - r0 + 1)
        widths.append(c1 - c0 + 1)
        bbox_areas.append((r1 - r0 + 1) * (c1 - c0 + 1))

    if all(bbox_areas[i] < bbox_areas[i + 1] for i in range(len(bbox_areas) - 1)):
        return False
    if all(heights[i] < heights[i + 1] for i in range(len(heights) - 1)):
        return False
    if all(widths[i] < widths[i + 1] for i in range(len(widths) - 1)):
        return False

    out = main(grid)
    if out.shape != (max(areas), len(areas)):
        return False
    for col, comp in enumerate(by_area):
        color = int(comp["color"])
        height = int(comp["area"])
        if not np.all(out[:-height, col] == 0 if height < out.shape[0] else True):
            return False
        if not np.all(out[out.shape[0] - height :, col] == color):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n = random.choice(COMPONENT_COUNT_CHOICES)
        for _ in range(TARGET_COUNT_RETRIES):
            areas = random.sample(list(AREA_CHOICES), n)
            colors = random.sample(list(PALETTE), n)

            shape_specs: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
            for area, color in zip(areas, colors):
                shape = _transform_shape(_random_irregular_shape(area))
                shape_specs.append((area, color, shape))

            side = random.randint(*GRID_SIDE_RANGE)
            grid = _place_shapes(shape_specs, (side, side))
            if grid is None:
                continue
            if _input_is_nondegenerate(grid):
                return grid

    raise RuntimeError("failed to generate a valid input")
