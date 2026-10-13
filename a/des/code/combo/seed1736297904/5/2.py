from __future__ import annotations

import math
from collections import deque

import numpy as np


BG = 0
FOREGROUND_COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIN_OBJECTS = 3
MAX_OBJECTS = 6
MIN_AREA = 2
MAX_AREA = 12
MIN_SIDE = 14
MAX_SIDE = 28
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            r_min = r_max = r
            c_min = c_max = c

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                r_min = min(r_min, cr)
                r_max = max(r_max, cr)
                c_min = min(c_min, cc)
                c_max = max(c_max, cc)
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "area": len(cells),
                    "cells": cells,
                    "bbox": (r_min, c_min, r_max, c_max),
                }
            )

    return components


def _recolor_by_reversed_area_order(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    components = _find_components(out, bg=BG)
    if len(components) < 2:
        return out

    ordered = sorted(components, key=lambda comp: int(comp["area"]))
    reversed_colors = [int(comp["color"]) for comp in reversed(ordered)]

    for comp, new_color in zip(ordered, reversed_colors):
        for r, c in comp["cells"]:
            out[r, c] = new_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color assignment after sorting monochrome 4-connected components by area.

    The black background stays fixed. Each foreground component keeps its shape
    and position, but the smallest component receives the original color of the
    largest one, the second smallest receives the second largest color, and so on.
    """
    return _recolor_by_reversed_area_order(grid)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _line_mask(area: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        return np.ones((1, area), dtype=bool)
    return np.ones((area, 1), dtype=bool)


def _rectangle_mask(area: int, rng: np.random.Generator) -> np.ndarray | None:
    factors = [(h, area // h) for h in range(2, area + 1) if area % h == 0 and area // h >= 2]
    if not factors:
        return None
    h, w = factors[int(rng.integers(len(factors)))]
    return np.ones((h, w), dtype=bool)


def _l_mask(area: int, rng: np.random.Generator) -> np.ndarray | None:
    options = [(arm_h, area - arm_h + 1) for arm_h in range(2, area) if area - arm_h + 1 >= 2]
    if not options:
        return None
    arm_h, arm_w = options[int(rng.integers(len(options)))]
    mask = np.zeros((arm_h, arm_w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _blob_mask(area: int, rng: np.random.Generator) -> np.ndarray:
    cells = {(0, 0)}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(0, 0)
    while len(cells) < area:
        choices = tuple(frontier)
        pick = choices[int(rng.integers(len(choices)))]
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(*pick)

    coords = np.array(list(cells), dtype=int)
    coords -= coords.min(axis=0)
    h, w = coords.max(axis=0) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _transform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if bool(rng.integers(2)):
        out = np.flipud(out)
    if bool(rng.integers(2)):
        out = np.fliplr(out)
    return _normalize_mask(out)


def _make_shape(area: int, rng: np.random.Generator) -> np.ndarray:
    choices: list[np.ndarray] = []
    choices.append(_line_mask(area, horizontal=True))
    choices.append(_line_mask(area, horizontal=False))

    rect = _rectangle_mask(area, rng)
    if rect is not None:
        choices.append(rect)

    l_shape = _l_mask(area, rng)
    if l_shape is not None:
        choices.append(l_shape)

    choices.append(_blob_mask(area, rng))
    return _transform_mask(choices[int(rng.integers(len(choices)))], rng)


def _choose_areas(count: int, rng: np.random.Generator) -> list[int]:
    while True:
        areas = sorted(int(x) for x in rng.choice(np.arange(MIN_AREA, MAX_AREA + 1), size=count, replace=False))
        if max(areas) >= 5 and len(set(areas)) == count:
            return areas


def _candidate_dimensions(areas: list[int], rng: np.random.Generator) -> tuple[int, int]:
    max_area = max(areas)
    total_area = sum(areas)
    min_side = max(MIN_SIDE, max_area + 4, int(math.ceil(math.sqrt(total_area * 5.0))))
    h = int(rng.integers(min_side, min(MAX_SIDE, min_side + 6) + 1))
    w = int(rng.integers(min_side, min(MAX_SIDE, min_side + 8) + 1))
    return h, w


def _can_place(reserved: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> bool:
    h, w = reserved.shape
    ph, pw = mask.shape
    r1 = r0 + ph
    c1 = c0 + pw
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(h, r1 + PLACEMENT_PADDING)
    cc1 = min(w, c1 + PLACEMENT_PADDING)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _place_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    ph, pw = mask.shape
    candidates = [
        (r0, c0)
        for r0 in range(h - ph + 1)
        for c0 in range(w - pw + 1)
        if _can_place(reserved, r0, c0, mask)
    ]
    if not candidates:
        return False

    r0, c0 = candidates[int(rng.integers(len(candidates)))]
    rr, cc = np.argwhere(mask).T
    grid[rr + r0, cc + c0] = color

    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(h, r0 + ph + PLACEMENT_PADDING)
    cc1 = min(w, c0 + pw + PLACEMENT_PADDING)
    reserved[rr0:rr1, cc0:cc1] = True
    return True


def _is_valid_scene(grid: np.ndarray, expected_count: int) -> bool:
    components = _find_components(grid, bg=BG)
    if len(components) != expected_count:
        return False

    colors = [int(comp["color"]) for comp in components]
    areas = [int(comp["area"]) for comp in components]
    if len(set(colors)) != expected_count or len(set(areas)) != expected_count:
        return False
    return all(color != BG for color in colors)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    areas = _choose_areas(count, rng)
    colors = [int(x) for x in rng.choice(np.array(FOREGROUND_COLORS), size=count, replace=False)]
    h, w = _candidate_dimensions(areas, rng)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    indices = list(int(x) for x in rng.permutation(count))

    for idx in indices:
        mask = _make_shape(areas[idx], rng)
        if not _place_shape(grid, reserved, mask, colors[idx], rng):
            return None

    if not _is_valid_scene(grid, count):
        return None

    solver_out = main(grid)
    expected = _recolor_by_reversed_area_order(grid)
    if not np.array_equal(solver_out, expected):
        return None

    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
