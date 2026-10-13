from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 14
GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict[str, object]] = []
    for comp in _components_mask(g != GREY):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        seed_cells = [(r, c) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one seed"
        area = sum(int(g[r, c]) == BLACK for r, c in comp)
        assert area > 0, "each enclosed region must contain at least one black cell"
        regions.append(
            {
                "cells": comp,
                "seed_color": int(g[seed_cells[0]]),
                "area": area,
                "anchor": min(comp),
            }
        )
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign enclosed-region seed colors by reversing the area ranking across the regions.

    Input structure:
    - Background is black.
    - One-pixel-thick grey closed boundaries separate several enclosed interior regions from the outside background.
    - Each enclosed region contains black cells plus exactly one non-black, non-grey seed pixel, and the seed colors are unique.

    Rule:
    - Treat each 4-connected non-grey component that does not touch the grid edge as one enclosed region interior.
    - Measure a region's area by counting its black cells only, excluding the seed pixel and the grey boundary.
    - Sort the seed colors by their own region areas from largest to smallest, then assign that color sequence to the regions ordered from smallest area to largest.
    - Fill every pixel inside each region, including the original seed location, with its reassigned color; keep the grey boundaries unchanged.

    Key insight:
    - The seed colors are not local fill labels; they first encode the descending area order, then get replayed onto the regions in the opposite area order.
    """
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)

    colors_by_large_area = [
        int(region["seed_color"])
        for region in sorted(regions, key=lambda region: (-int(region["area"]), region["anchor"]))
    ]
    regions_by_small_area = sorted(regions, key=lambda region: (int(region["area"]), region["anchor"]))

    out = g.copy()
    for region, color in zip(regions_by_small_area, colors_by_large_area):
        rows, cols = zip(*region["cells"])
        out[list(rows), list(cols)] = int(color)
    return out


def _normalize(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return {(r - min_r, c - min_c) for r, c in points}


def _has_hole(cells: set[tuple[int, int]]) -> bool:
    cells = _normalize(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_edge = True
                        q.append((nr, nc))
            if not touches_edge:
                return True
    return False


def _is_connected(points: set[tuple[int, int]], conn8: bool = False) -> bool:
    if not points:
        return False
    points = set(points)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    start = next(iter(points))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nxt = (r + dr, c + dc)
            if nxt in points and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(points)


def _boundary_from_interior(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in cells:
                boundary.add((nr, nc))
    return boundary


def _make_shape(rng: random.Random, cell_count: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(250):
        box_h = rng.randint(2, min(7, cell_count))
        box_w = rng.randint(2, min(7, cell_count))
        if box_h * box_w < cell_count:
            continue

        seed_r = rng.randint(0, box_h - 2)
        seed_c = rng.randint(0, box_w - 2)
        cells = {
            (seed_r, seed_c),
            (seed_r + 1, seed_c),
            (seed_r, seed_c + 1),
            (seed_r + 1, seed_c + 1),
        }

        while len(cells) < cell_count:
            frontier: list[tuple[int, int]] = []
            weights: list[int] = []
            seen_frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < box_h and 0 <= nc < box_w):
                        continue
                    if (nr, nc) in cells or (nr, nc) in seen_frontier:
                        continue
                    seen_frontier.add((nr, nc))
                    adj = sum((ar, ac) in cells for ar, ac in _neighbors4(nr, nc))
                    frontier.append((nr, nc))
                    weights.append(1 + adj * adj)
            if not frontier:
                break
            cells.add(rng.choices(frontier, weights=weights, k=1)[0])

        if len(cells) != cell_count:
            continue

        cells = _normalize(cells)
        if len({r for r, _ in cells}) < 2 or len({c for _, c in cells}) < 2:
            continue
        if _has_hole(cells):
            continue

        boundary = _boundary_from_interior(cells)
        footprint = cells | boundary
        min_r = min(r for r, _ in footprint)
        min_c = min(c for _, c in footprint)
        cells = {(r - min_r, c - min_c) for r, c in cells}
        boundary = {(r - min_r, c - min_c) for r, c in boundary}

        if not _is_connected(boundary, conn8=True):
            continue
        return cells, boundary
    return None


def _moat(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded.add((r + dr, c + dc))
    return padded


def _find_placement(
    rng: random.Random,
    grid_shape: tuple[int, int],
    footprint: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = grid_shape
    shape_h = max(r for r, _ in footprint) + 1
    shape_w = max(c for _, c in footprint) + 1
    if h - shape_h <= 1 or w - shape_w <= 1:
        return None

    candidates: list[tuple[int, int]] = []
    for top in range(1, h - shape_h):
        for left in range(1, w - shape_w):
            shifted = {(r + top, c + left) for r, c in footprint}
            if shifted & reserved:
                continue
            candidates.append((top, left))
    if not candidates:
        return None
    return rng.choice(candidates)


def _size_pool(region_count: int) -> list[int]:
    if region_count == 2:
        return list(range(5, 19))
    if region_count == 3:
        return list(range(5, 15))
    if region_count == 4:
        return list(range(5, 12))
    return list(range(5, 10))


def _region_count(rng: random.Random, h: int, w: int) -> int:
    side = min(h, w)
    if side >= 26:
        return rng.randint(3, 5)
    if side >= 20:
        return rng.randint(2, 4)
    return rng.randint(2, 3)


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    region_count = _region_count(rng, h, w)
    inner_sizes = rng.sample(_size_pool(region_count), region_count)
    region_colors = rng.sample(SEED_COLORS, region_count)
    specs = sorted(zip(inner_sizes, region_colors), key=lambda item: item[0], reverse=True)

    reserved: set[tuple[int, int]] = set()

    # Place large footprints first to reduce packing failures.
    for inner_size, seed_color in specs:
        placed = False
        for _ in range(200):
            shape = _make_shape(rng, inner_size)
            if shape is None:
                continue
            interior, boundary = shape
            footprint = interior | boundary
            placement = _find_placement(rng, grid.shape, footprint, reserved)
            if placement is None:
                continue

            top, left = placement
            shifted_interior = {(r + top, c + left) for r, c in interior}
            shifted_boundary = {(r + top, c + left) for r, c in boundary}
            shifted_footprint = shifted_interior | shifted_boundary

            for r, c in shifted_boundary:
                grid[r, c] = GREY

            seed_r, seed_c = rng.choice(tuple(shifted_interior))
            grid[seed_r, seed_c] = seed_color
            reserved |= _moat(shifted_footprint)
            placed = True
            break
        if not placed:
            return None

    try:
        regions = _extract_regions(grid)
    except AssertionError:
        return None

    if len(regions) != region_count:
        return None
    if len({int(region["seed_color"]) for region in regions}) != region_count:
        return None
    if len({int(region["area"]) for region in regions}) != region_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
