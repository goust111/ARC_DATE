from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict[str, object]] = []
    for cells in _components(g != GREY):
        if _touches_border(cells, g.shape):
            continue
        seeds = [(r, c) for r, c in cells if int(g[r, c]) != BLACK]
        assert len(seeds) == 1, "each enclosed region must contain exactly one colored seed"
        sr, sc = seeds[0]
        regions.append(
            {
                "cells": cells,
                "seed": (sr, sc),
                "color": int(g[sr, sc]),
                "area": int(len(cells)),
            }
        )
    assert regions, "expected at least one enclosed region"
    return regions


def _paint_cross(out: np.ndarray, seed: tuple[int, int], color: int, area: int) -> None:
    h, w = out.shape
    r, c = seed
    reach = area - 1
    for cc in range(max(0, c - reach), min(w, c + reach + 1)):
        out[r, cc] = color
    for rr in range(max(0, r - reach), min(h, r + reach + 1)):
        out[rr, c] = color


def _cross_cells(seed: tuple[int, int], area: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = seed
    reach = area - 1
    cells = {(r, cc) for cc in range(max(0, c - reach), min(w, c + reach + 1))}
    cells |= {(rr, c) for rr in range(max(0, r - reach), min(h, r + reach + 1))}
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each enclosed seed into a cross whose arm length equals its region area.

    Input structure:
    - Black background fills the canvas outside several grey one-pixel-thick closed boundaries.
    - Each grey boundary encloses one interior region made of black cells plus exactly one colored seed pixel.
    - Different enclosed regions have different interior areas, counted as all non-grey cells inside the boundary.

    Rule:
    - For every enclosed region, count its interior area S, including the seed cell itself.
    - Starting from the seed, paint a same-colored cross along its row and column, extending S cells in each of the four directions with the seed counted as length 1.
    - Cross growth ignores grey walls and other regions; any covered grey pixels are recolored too.
    - When crosses overlap, the one from the larger-area region is painted last and therefore remains visible.

    Key insight:
    - The seed color only identifies which cross to draw; the actual cross size is determined by the area of the seed's grey-enclosed container.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _extract_regions(g)
    regions.sort(key=lambda item: (int(item["area"]), tuple(item["seed"])))
    for region in regions:
        _paint_cross(out, tuple(region["seed"]), int(region["color"]), int(region["area"]))
    return out


TEMPLATES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    2: (
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
    ),
    3: (
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (-1, 0), (1, 0)),
        ((0, 0), (0, -1), (0, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (-1, 0), (1, 0), (0, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    5: (
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ),
    6: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)),
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (2, 0)),
    ),
    7: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (2, 0)),
    ),
    8: (
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)),
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (2, 0)),
    ),
}

GEN_RETRIES = 500
PLACEMENT_TRIES = 200
OVERLAP_PAIR_TRIES = 300
GRID_OVERLAP_RANGE = (18, 28)
GRID_SEPARATE_RANGE = (14, 24)


def _shell(shape: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    body = set(shape)
    ring: set[tuple[int, int]] = set()
    for r, c in body:
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr not in body:
                ring.add(nbr)
    return ring


def _profile(shape: tuple[tuple[int, int], ...]) -> dict[str, object]:
    body = set(shape)
    ring = _shell(shape)
    all_cells = body | ring
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    return {
        "shape": body,
        "ring": ring,
        "all": all_cells,
        "min_r": min(rows),
        "max_r": max(rows),
        "min_c": min(cols),
        "max_c": max(cols),
    }


def _translated(cells: set[tuple[int, int]], seed: tuple[int, int]) -> set[tuple[int, int]]:
    sr, sc = seed
    return {(sr + r, sc + c) for r, c in cells}


def _seed_bounds(profile: dict[str, object], shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    r0 = 1 - int(profile["min_r"])
    r1 = h - 2 - int(profile["max_r"])
    c0 = 1 - int(profile["min_c"])
    c1 = w - 2 - int(profile["max_c"])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, r1), (c0, c1)


def _buffer(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _cross_overlap(
    seeds: list[tuple[int, int]],
    areas: list[int],
    shape: tuple[int, int],
) -> bool:
    seen: set[tuple[int, int]] = set()
    for seed, area in zip(seeds, areas):
        cells = _cross_cells(seed, area, shape)
        if seen & cells:
            return True
        seen |= cells
    return False


def _build_region_cells(profile: dict[str, object], seed: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    body = _translated(profile["shape"], seed)
    ring = _translated(profile["ring"], seed)
    return body, ring


def _place_region(
    grid: np.ndarray,
    profile: dict[str, object],
    seed: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    body, ring = _build_region_cells(profile, seed)
    occupied = body | ring
    if occupied & blocked:
        return None
    for r, c in ring:
        grid[r, c] = GREY
    sr, sc = seed
    grid[sr, sc] = color
    return occupied


def _random_seed(
    profile: dict[str, object],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    used_crosses: set[tuple[int, int]],
    area: int,
) -> tuple[int, int] | None:
    bounds = _seed_bounds(profile, shape)
    if bounds is None:
        return None
    (r0, r1), (c0, c1) = bounds
    for _ in range(PLACEMENT_TRIES):
        seed = (random.randint(r0, r1), random.randint(c0, c1))
        occupied = _translated(profile["all"], seed)
        if occupied & blocked:
            continue
        cross = _cross_cells(seed, area, shape)
        if used_crosses & cross:
            continue
        return seed
    return None


def _paired_seeds(
    p1: dict[str, object],
    a1: int,
    p2: dict[str, object],
    a2: int,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    b1 = _seed_bounds(p1, shape)
    b2 = _seed_bounds(p2, shape)
    if b1 is None or b2 is None:
        return None
    (r10, r11), (c10, c11) = b1
    (r20, r21), (c20, c21) = b2
    reach1 = a1 - 1
    reach2 = a2 - 1
    patterns = ["hub", "same_row", "same_col"]
    random.shuffle(patterns)
    for pattern in patterns:
        for _ in range(OVERLAP_PAIR_TRIES):
            if pattern == "hub":
                seed1 = (random.randint(r10, r11), random.randint(c10, c11))
                seed2 = (random.randint(r20, r21), random.randint(c20, c21))
                dr = abs(seed1[0] - seed2[0])
                dc = abs(seed1[1] - seed2[1])
                if dr < 4 or dc < 4 or dr > reach2 or dc > reach1:
                    continue
            elif pattern == "same_row":
                lo = max(r10, r20)
                hi = min(r11, r21)
                if lo > hi or reach1 + reach2 < 6:
                    break
                row = random.randint(lo, hi)
                seed1 = (row, random.randint(c10, c11))
                seed2 = (row, random.randint(c20, c21))
                if abs(seed1[1] - seed2[1]) < 6 or abs(seed1[1] - seed2[1]) > reach1 + reach2:
                    continue
            else:
                lo = max(c10, c20)
                hi = min(c11, c21)
                if lo > hi or reach1 + reach2 < 6:
                    break
                col = random.randint(lo, hi)
                seed1 = (random.randint(r10, r11), col)
                seed2 = (random.randint(r20, r21), col)
                if abs(seed1[0] - seed2[0]) < 6 or abs(seed1[0] - seed2[0]) > reach1 + reach2:
                    continue

            occ1 = _translated(p1["all"], seed1)
            occ2 = _translated(p2["all"], seed2)
            if occ1 & _buffer(occ2, shape):
                continue
            if not (_cross_cells(seed1, a1, shape) & _cross_cells(seed2, a2, shape)):
                continue
            return seed1, seed2
    return None


def _pick_areas(region_count: int, want_overlap: bool) -> list[int]:
    if want_overlap:
        large = random.sample((5, 6, 7, 8), 2)
        rest_pool = [2, 3, 4]
        rest = random.sample(rest_pool, region_count - 2)
        areas = large + rest
    else:
        areas = random.sample((2, 3, 4, 5, 6, 7, 8), region_count)
    random.shuffle(areas)
    return areas


def _candidate_scene(want_overlap: bool) -> np.ndarray | None:
    h = random.randint(*(GRID_OVERLAP_RANGE if want_overlap else GRID_SEPARATE_RANGE))
    w = random.randint(*(GRID_OVERLAP_RANGE if want_overlap else GRID_SEPARATE_RANGE))
    region_count = random.randint(2, 4)
    areas = _pick_areas(region_count, want_overlap)
    colors = random.sample(SEED_COLORS, region_count)
    profiles = [_profile(random.choice(TEMPLATES[area])) for area in areas]

    grid = np.full((h, w), BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()
    used_crosses: set[tuple[int, int]] = set()
    placed_seeds: list[tuple[int, int]] = []

    start_index = 0
    if want_overlap:
        ordered = sorted(range(region_count), key=lambda idx: areas[idx], reverse=True)
        i0, i1 = ordered[:2]
        pair = _paired_seeds(profiles[i0], areas[i0], profiles[i1], areas[i1], (h, w))
        if pair is None:
            return None
        for idx, seed in zip((i0, i1), pair):
            occupied = _place_region(grid, profiles[idx], seed, colors[idx], blocked)
            if occupied is None:
                return None
            blocked |= _buffer(occupied, (h, w))
            used_crosses |= _cross_cells(seed, areas[idx], (h, w))
            placed_seeds.append(seed)
        start_index = -1

    for idx in range(region_count):
        if want_overlap and idx in ordered[:2]:
            continue
        seed = _random_seed(profiles[idx], (h, w), blocked, used_crosses, areas[idx])
        if seed is None:
            return None
        occupied = _place_region(grid, profiles[idx], seed, colors[idx], blocked)
        if occupied is None:
            return None
        blocked |= _buffer(occupied, (h, w))
        used_crosses |= _cross_cells(seed, areas[idx], (h, w))
        placed_seeds.append(seed)

    if start_index == 0 and _cross_overlap(placed_seeds, areas, (h, w)):
        return None

    try:
        regions = _extract_regions(grid)
    except AssertionError:
        return None

    if len(regions) != region_count:
        return None
    if len({int(region["area"]) for region in regions}) != region_count:
        return None
    overlap = _cross_overlap([tuple(region["seed"]) for region in regions], [int(region["area"]) for region in regions], (h, w))
    if overlap != want_overlap:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero((grid == GREY) & (out != GREY)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        want_overlap = random.random() < 0.5
        grid = _candidate_scene(want_overlap)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
