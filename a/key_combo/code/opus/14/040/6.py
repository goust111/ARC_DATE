from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7
WALL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert pts.shape[0] == 1, f"expected exactly one seed of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _trace_orange(grid: np.ndarray, seed: tuple[int, int]) -> tuple[set[tuple[int, int]], int]:
    cells = {seed}
    wall_hits = 0
    for dr, dc in DIRS4:
        r, c = seed
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                break
            if int(grid[nr, nc]) == WALL:
                wall_hits += 1
                break
            if int(grid[nr, nc]) == BLACK:
                cells.add((nr, nc))
            r, c = nr, nc
    return cells, wall_hits


def _trace_blue(
    orange_grid: np.ndarray, seed: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int]:
    blue_cells = {seed}
    red_cells: set[tuple[int, int]] = set()
    wall_hits = 0
    for dr, dc in DIRS4:
        r, c = seed
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(orange_grid.shape, nr, nc):
                break
            value = int(orange_grid[nr, nc])
            if value == WALL:
                wall_hits += 1
                break
            if value == ORANGE:
                red_cells.add((nr, nc))
                break
            if value == BLACK:
                blue_cells.add((nr, nc))
            r, c = nr, nc
    return blue_cells, red_cells, wall_hits


def _regions(ray_grid: np.ndarray) -> list[dict]:
    h, w = ray_grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(ray_grid[r0, c0]) != BLACK:
                continue
            q: deque[tuple[int, int]] = deque([(r0, c0)])
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            orange_boundary: set[tuple[int, int]] = set()
            blue_boundary: set[tuple[int, int]] = set()
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r in (0, h - 1) or c in (0, w - 1):
                    touches_border = True
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if not _in_bounds(ray_grid.shape, nr, nc):
                        continue
                    value = int(ray_grid[nr, nc])
                    if value == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif value == ORANGE:
                        orange_boundary.add((nr, nc))
                    elif value == BLUE:
                        blue_boundary.add((nr, nc))
            orange_count = len(orange_boundary)
            blue_count = len(blue_boundary)
            if orange_count > blue_count:
                fill = ORANGE
            elif orange_count < blue_count:
                fill = BLUE
            else:
                fill = RED
            regions.append(
                {
                    "cells": cells,
                    "touches_border": touches_border,
                    "orange_count": orange_count,
                    "blue_count": blue_count,
                    "fill": fill,
                }
            )
    return regions


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orange_seed = _single_coord(g, ORANGE)
    blue_seed = _single_coord(g, BLUE)
    assert orange_seed[0] != blue_seed[0] and orange_seed[1] != blue_seed[1], "seeds must differ in row and column"

    orange_cells, orange_wall_hits = _trace_orange(g, orange_seed)
    ray_grid = g.copy()
    for r, c in orange_cells:
        if int(ray_grid[r, c]) in (BLACK, ORANGE):
            ray_grid[r, c] = ORANGE

    blue_cells, red_cells, blue_wall_hits = _trace_blue(ray_grid, blue_seed)
    for r, c in blue_cells:
        if int(ray_grid[r, c]) in (BLACK, BLUE):
            ray_grid[r, c] = BLUE
    for r, c in red_cells:
        ray_grid[r, c] = RED

    regions = _regions(ray_grid)
    out = ray_grid.copy()
    fillable = [region for region in regions if not region["touches_border"]]
    for region in fillable:
        for r, c in region["cells"]:
            out[r, c] = region["fill"]

    meta = {
        "orange_seed": orange_seed,
        "blue_seed": blue_seed,
        "orange_cells": orange_cells,
        "blue_cells": blue_cells,
        "red_cells": red_cells,
        "orange_wall_hits": orange_wall_hits,
        "blue_wall_hits": blue_wall_hits,
        "ray_grid": ray_grid,
        "regions": regions,
        "fillable_regions": fillable,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend two blocked crosses, mark their intersections, then fill each enclosed black region by the dominant boundary ray color.

    Input structure:
    - Black cells form the background, with exactly one orange seed, one blue seed, and several teal wall segments.
    - The seeds are isolated single pixels and are not in the same row or column.
    - Walls are straight horizontal or vertical blockers that rays can stop against.

    Rule:
    - Extend orange rays left, right, up, and down from the orange seed, painting only black cells until a wall or the grid edge stops the ray.
    - Extend blue rays in the same four directions, but blue rays stop at walls, the grid edge, or the first orange ray cell they hit; that orange/blue crossing cell becomes red.
    - Treat walls and all ray pixels as barriers. For each remaining black 4-connected region that does not touch the grid border, count distinct orange and blue boundary pixels around it, ignoring red and teal. Fill the region orange if orange dominates, blue if blue dominates, or red on a tie. Border-touching regions stay black.

    Key insight:
    - The rays are not just decoration: once walls truncate them, the resulting barrier network partitions the background into regions whose fill color is determined by the balance of orange versus blue pixels on each region boundary.
    """
    out, _meta = _solve_with_meta(grid)
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _quadrant_specs(
    top: int, left: int, bottom: int, right: int, orange_seed: tuple[int, int]
) -> dict[str, tuple[int, int, int, int]]:
    ro, co = orange_seed
    return {
        "tl": (top + 1, left + 1, ro - 1, co - 1),
        "tr": (top + 1, co + 1, ro - 1, right - 1),
        "bl": (ro + 1, left + 1, bottom - 1, co - 1),
        "br": (ro + 1, co + 1, bottom - 1, right - 1),
    }


def _blue_candidates(
    top: int, left: int, bottom: int, right: int, orange_seed: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    ro, co = orange_seed
    choices: dict[str, list[tuple[int, int]]] = {}
    ranges = {
        "tl": (range(top + 2, ro - 1), range(left + 2, co - 1)),
        "tr": (range(top + 2, ro - 1), range(co + 2, right - 1)),
        "bl": (range(ro + 2, bottom - 1), range(left + 2, co - 1)),
        "br": (range(ro + 2, bottom - 1), range(co + 2, right - 1)),
    }
    for key, (rows, cols) in ranges.items():
        cells = [(r, c) for r in rows for c in cols]
        if cells:
            choices[key] = cells
    return choices


def _splitter_options(
    top: int,
    left: int,
    bottom: int,
    right: int,
    orange_seed: tuple[int, int],
    blue_quadrant: str,
) -> list[list[tuple[int, int]]]:
    ro, co = orange_seed
    options: list[list[tuple[int, int]]] = []
    bounds = _quadrant_specs(top, left, bottom, right, orange_seed)
    for quadrant, (r0, c0, r1, c1) in bounds.items():
        if quadrant == blue_quadrant:
            continue
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if quadrant in ("tl", "tr") and height >= 3:
            for c in range(c0 + 1, c1):
                options.append([(r, c) for r in range(r0, r1 + 1)])
        if quadrant in ("bl", "br") and height >= 3:
            for c in range(c0 + 1, c1):
                options.append([(r, c) for r in range(r0, r1 + 1)])
        if quadrant in ("tl", "bl") and width >= 3:
            for r in range(r0 + 1, r1):
                options.append([(r, c) for c in range(c0, c1 + 1)])
        if quadrant in ("tr", "br") and width >= 3:
            for r in range(r0 + 1, r1):
                options.append([(r, c) for c in range(c0, c1 + 1)])
    filtered = [cells for cells in options if 3 <= len(cells) <= 7]
    random.shuffle(filtered)
    return filtered


def _build_candidate() -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    h = random.randint(13, 19)
    w = random.randint(13, 19)
    grid = np.zeros((h, w), dtype=int)

    chamber_h = random.choice([6, 7])
    chamber_w = random.choice([6, 7])
    if h - chamber_h < 4 or w - chamber_w < 4:
        return None
    top = random.randint(2, h - chamber_h - 2)
    left = random.randint(2, w - chamber_w - 2)
    bottom = top + chamber_h - 1
    right = left + chamber_w - 1

    segments = [
        _segment_cells((top, left), (top, right)),
        _segment_cells((bottom, left), (bottom, right)),
        _segment_cells((top, left), (bottom, left)),
        _segment_cells((top, right), (bottom, right)),
    ]
    orange_rows = list(range(top + 2, bottom - 1))
    orange_cols = list(range(left + 2, right - 1))
    if not orange_rows or not orange_cols:
        return None
    orange_candidates = [(r, c) for r in orange_rows for c in orange_cols]
    random.shuffle(orange_candidates)

    for orange_seed in orange_candidates:
        blue_by_quadrant = _blue_candidates(top, left, bottom, right, orange_seed)
        if not blue_by_quadrant:
            continue
        blue_quadrant = random.choice(sorted(blue_by_quadrant))
        blue_seed = random.choice(blue_by_quadrant[blue_quadrant])

        splitter_cells: list[tuple[int, int]] = []
        if random.random() < 0.55:
            splitter_choices = _splitter_options(top, left, bottom, right, orange_seed, blue_quadrant)
            if splitter_choices:
                splitter_cells = random.choice(splitter_choices)

        for cells in segments:
            for r, c in cells:
                grid[r, c] = WALL
        for r, c in splitter_cells:
            grid[r, c] = WALL
        grid[orange_seed] = ORANGE
        grid[blue_seed] = BLUE
        return grid, splitter_cells

    return None


def _valid_candidate(grid: np.ndarray, splitter_cells: list[tuple[int, int]]) -> bool:
    try:
        out, meta = _solve_with_meta(grid)
    except AssertionError:
        return False

    fillable = meta["fillable_regions"]
    if meta["orange_wall_hits"] < 1 or meta["blue_wall_hits"] < 1:
        return False
    if len(meta["red_cells"]) != 2:
        return False
    if len(fillable) < 3:
        return False
    if not any(region["touches_border"] for region in meta["regions"]):
        return False

    comparisons = {(region["orange_count"] > region["blue_count"], region["orange_count"] < region["blue_count"]) for region in fillable}
    if not any(region["orange_count"] > region["blue_count"] for region in fillable):
        return False
    if not any(region["orange_count"] < region["blue_count"] for region in fillable):
        return False
    if not any(region["orange_count"] == region["blue_count"] for region in fillable):
        return False

    fills = {int(region["fill"]) for region in fillable}
    if not {ORANGE, BLUE, RED}.issubset(fills):
        return False
    if np.array_equal(grid, out):
        return False

    if splitter_cells:
        simpler = grid.copy()
        for r, c in splitter_cells:
            simpler[r, c] = BLACK
        if np.array_equal(main(simpler), out):
            return False

    orange_seed = meta["orange_seed"]
    blue_seed = meta["blue_seed"]
    if orange_seed[0] == blue_seed[0] or orange_seed[1] == blue_seed[1]:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, splitter_cells = built
        if _valid_candidate(grid, splitter_cells):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
