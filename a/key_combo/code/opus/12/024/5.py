from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
SEED_COLORS = (1, 3, 4, 5, 7)
DIR4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
INNER_H_RANGE = (7, 12)
INNER_W_RANGE = (7, 12)
MARGIN_RANGE = (2, 4)
TARGET_CAVITIES = (3, 5)
MIN_REGION_SIDE = 2
MIN_REGION_AREA = 4
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _outside_mask(grid: np.ndarray) -> np.ndarray:
    passable = grid != RED
    h, w = grid.shape
    out = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not out[r, c]:
                out[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not out[r, c]:
                out[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[dict[str, object]]]:
    outside = _outside_mask(grid)
    capacities: dict[int, int] = {}
    for color in map(int, np.unique(grid[outside])):
        if color in (BLACK, RED):
            continue
        capacities[color] = int(np.sum(outside & (grid == color)))

    cavity_mask = (grid != RED) & (~outside)
    cavities: list[dict[str, object]] = []
    for comp in _components(cavity_mask):
        colors = sorted({int(grid[r, c]) for r, c in comp if int(grid[r, c]) != BLACK})
        if not colors:
            continue
        if len(colors) != 1:
            raise AssertionError("each cavity must contain exactly one seed color")
        color = colors[0]
        seed_positions = [(r, c) for r, c in comp if int(grid[r, c]) == color]
        if len(seed_positions) != 1:
            raise AssertionError("each cavity must contain exactly one seed pixel")
        cavities.append(
            {
                "color": color,
                "seed": seed_positions[0],
                "cells": comp,
                "cell_set": set(comp),
                "size": len(comp),
            }
        )
    return outside, capacities, cavities


def _paint_cavity(out: np.ndarray, cavity: dict[str, object], capacity: int) -> None:
    color = int(cavity["color"])
    seed = tuple(cavity["seed"])
    cell_set = cavity["cell_set"]

    seen = {seed}
    frontier = [seed]
    painted = 1
    while frontier and painted < capacity:
        next_level: set[tuple[int, int]] = set()
        for r, c in sorted(frontier):
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cell_set and out[nr, nc] == BLACK and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    next_level.add((nr, nc))
        if not next_level:
            break
        frontier = sorted(next_level)
        for r, c in frontier:
            out[r, c] = color
            painted += 1
            if painted >= capacity:
                return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each sealed chamber from its seed, but stop when the matching outside bar runs out of pixels.

    Input structure:
    - Black background with a 4-connected red 1-pixel wall network that partitions the interior into 3-5 sealed cavities.
    - Each cavity contains exactly one uniquely colored seed pixel chosen from {1,3,4,5,7}; the rest of the cavity is black.
    - In the border-reachable outside region, each seed color also appears as one horizontal or vertical 1-pixel bar touching the exterior side of the red wall network.

    Rule:
    - Count the pixels of each outside colored bar; that count is the fill capacity for the cavity whose seed has the same color.
    - For every sealed cavity, start from its seed and expand only onto black cavity cells with 4-connected BFS.
    - Process BFS by increasing distance from the seed; within one distance layer, process cells in row-major order.
    - Recolor visited cells with the seed color and stop immediately when the total colored count, including the seed itself, reaches the bar length.
    - After all cavities are processed, erase every outside colored bar to black and keep the red wall network unchanged.

    Key insight:
    - The final filled shape depends on two independent clues that must be combined: chamber geometry determines BFS order, while the same-colored outside bar determines where that BFS is truncated.
    """
    g = np.asarray(grid, dtype=int)
    outside, capacities, cavities = _parse_scene(g)

    out = g.copy()
    out[outside & (out != BLACK) & (out != RED)] = BLACK

    for cavity in cavities:
        color = int(cavity["color"])
        capacity = int(capacities.get(color, 0))
        if capacity <= 0:
            raise AssertionError("every seed color must have a matching outside bar")
        _paint_cavity(out, cavity, capacity)

    return out


def _draw_outer_frame(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    grid[top, left : right + 1] = RED
    grid[bottom, left : right + 1] = RED
    grid[top : bottom + 1, left] = RED
    grid[top : bottom + 1, right] = RED


def _split_regions(
    inner_rect: tuple[int, int, int, int], target_count: int
) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]] | None:
    regions = [inner_rect]
    walls: list[tuple[str, int, int, int]] = []

    while len(regions) < target_count:
        splittable = []
        for idx, (r0, c0, r1, c1) in enumerate(regions):
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if width >= 2 * MIN_REGION_SIDE + 1:
                splittable.append((idx, "v"))
            if height >= 2 * MIN_REGION_SIDE + 1:
                splittable.append((idx, "h"))
        if not splittable:
            return None

        areas = []
        for idx, _ in splittable:
            r0, c0, r1, c1 = regions[idx]
            areas.append((r1 - r0 + 1) * (c1 - c0 + 1))
        pick = random.choices(splittable, weights=areas, k=1)[0]
        idx, orient = pick
        r0, c0, r1, c1 = regions.pop(idx)

        if orient == "v":
            choices = list(range(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE + 1))
            wall_c = random.choice(choices)
            left_region = (r0, c0, r1, wall_c - 1)
            right_region = (r0, wall_c + 1, r1, c1)
            walls.append(("v", wall_c, r0 - 1, r1 + 1))
            regions.extend([left_region, right_region])
        else:
            choices = list(range(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE + 1))
            wall_r = random.choice(choices)
            top_region = (r0, c0, wall_r - 1, c1)
            bottom_region = (wall_r + 1, c0, r1, c1)
            walls.append(("h", wall_r, c0 - 1, c1 + 1))
            regions.extend([top_region, bottom_region])

    if any((r1 - r0 + 1) * (c1 - c0 + 1) < MIN_REGION_AREA for r0, c0, r1, c1 in regions):
        return None
    return regions, walls


def _draw_partition(
    grid: np.ndarray,
    outer_frame: tuple[int, int, int, int],
    walls: list[tuple[str, int, int, int]],
) -> None:
    top, left, bottom, right = outer_frame
    _draw_outer_frame(grid, top, left, bottom, right)
    for orient, fixed, a0, a1 in walls:
        if orient == "v":
            grid[a0 : a1 + 1, fixed] = RED
        else:
            grid[fixed, a0 : a1 + 1] = RED


def _pick_seed_positions(
    regions: list[tuple[int, int, int, int]],
    colors: list[int],
) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for color, (r0, c0, r1, c1) in zip(colors, regions):
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        r, c = random.choice(cells)
        seeds.append((r, c, color))
    return seeds


def _choose_capacities(region_sizes: list[int]) -> list[int] | None:
    small = [i for i, size in enumerate(region_sizes) if size <= 8]
    partial_ok = [i for i, size in enumerate(region_sizes) if size >= 3]
    if not small or len(partial_ok) < 2:
        return None

    full_idx = random.choice(small)
    partial_candidates = [i for i in partial_ok if i != full_idx]
    if not partial_candidates:
        return None
    partial_idx = random.choice(partial_candidates)

    capacities: list[int] = []
    for idx, size in enumerate(region_sizes):
        full_feasible = size <= 8
        partial_feasible = size >= 3
        if idx == full_idx:
            mode = "full"
        elif idx == partial_idx:
            mode = "partial"
        else:
            modes = []
            if full_feasible:
                modes.extend(["full", "full"])
            if partial_feasible:
                modes.extend(["partial", "partial"])
            mode = random.choice(modes)

        if mode == "full":
            capacities.append(random.randint(max(2, size), 8))
        else:
            capacities.append(random.randint(2, min(8, size - 1)))
    return capacities


def _bar_candidates(
    outer_frame: tuple[int, int, int, int],
    length: int,
) -> list[list[tuple[int, int]]]:
    top, left, bottom, right = outer_frame
    candidates: list[list[tuple[int, int]]] = []
    for c0 in range(left, right - length + 2):
        candidates.append([(top - 1, c) for c in range(c0, c0 + length)])
        candidates.append([(bottom + 1, c) for c in range(c0, c0 + length)])
    for r0 in range(top, bottom - length + 2):
        candidates.append([(r, left - 1) for r in range(r0, r0 + length)])
        candidates.append([(r, right + 1) for r in range(r0, r0 + length)])
    random.shuffle(candidates)
    return candidates


def _can_place_bar(grid: np.ndarray, cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK or (r, c) in occupied:
            return False
        if not any(
            _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == RED
            for nr, nc in _neighbors4(r, c)
        ):
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_bars(
    grid: np.ndarray,
    outer_frame: tuple[int, int, int, int],
    colors: list[int],
    capacities: list[int],
) -> bool:
    occupied: set[tuple[int, int]] = set()
    order = list(range(len(colors)))
    random.shuffle(order)
    for idx in order:
        color = colors[idx]
        length = capacities[idx]
        placed = False
        for cells in _bar_candidates(outer_frame, length):
            if _can_place_bar(grid, cells, occupied):
                for r, c in cells:
                    grid[r, c] = color
                    occupied.add((r, c))
                placed = True
                break
        if not placed:
            return False
    return True


def _semantic_checks(grid: np.ndarray) -> None:
    if grid.ndim != 2 or max(grid.shape) > 30:
        raise AssertionError("grid shape must stay within ARC limits")

    red_mask = grid == RED
    red_comps = _components(red_mask)
    if len(red_comps) != 1:
        raise AssertionError("red wall drawing must be one 4-connected component")

    outside, capacities, cavities = _parse_scene(grid)
    if not (3 <= len(cavities) <= 5):
        raise AssertionError("must contain 3-5 cavities")

    cavity_colors = [int(cavity["color"]) for cavity in cavities]
    if len(set(cavity_colors)) != len(cavity_colors):
        raise AssertionError("seed colors must be unique")

    if set(cavity_colors) != set(capacities):
        raise AssertionError("outside bars must match cavity seed colors exactly")

    full_count = 0
    partial_count = 0
    for cavity in cavities:
        color = int(cavity["color"])
        capacity = int(capacities[color])
        if not (2 <= capacity <= 8):
            raise AssertionError("bar lengths must be 2..8")
        if capacity >= int(cavity["size"]):
            full_count += 1
        else:
            partial_count += 1

    if full_count < 1 or partial_count < 1:
        raise AssertionError("need both full-fill and partial-fill cavities")

    outside_colors = [int(v) for v in np.unique(grid[outside]) if int(v) not in (BLACK, RED)]
    for color in outside_colors:
        comps = _components(outside & (grid == color))
        if len(comps) != 1:
            raise AssertionError("each outside color must form exactly one bar")
        if len(comps[0]) != capacities[color]:
            raise AssertionError("bar size must equal capacity")

    out = main(grid)
    if np.any(out[outside] != np.where(grid[outside] == RED, RED, BLACK)):
        raise AssertionError("outside bars must be erased")

    residual_black = 0
    for cavity in cavities:
        cells = cavity["cells"]
        colors_after = {int(out[r, c]) for r, c in cells}
        if RED in colors_after:
            raise AssertionError("red should not appear inside cavities")
        if BLACK in colors_after:
            residual_black += 1
    if residual_black < 1:
        raise AssertionError("output should retain at least one partially filled cavity")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        mt = random.randint(*MARGIN_RANGE)
        mb = random.randint(*MARGIN_RANGE)
        ml = random.randint(*MARGIN_RANGE)
        mr = random.randint(*MARGIN_RANGE)
        h = inner_h + mt + mb + 2
        w = inner_w + ml + mr + 2
        if not (GRID_H_RANGE[0] <= h <= GRID_H_RANGE[1] and GRID_W_RANGE[0] <= w <= GRID_W_RANGE[1]):
            continue

        top, left = mt, ml
        bottom, right = top + inner_h + 1, left + inner_w + 1
        outer_frame = (top, left, bottom, right)
        inner_rect = (top + 1, left + 1, bottom - 1, right - 1)

        target = random.randint(*TARGET_CAVITIES)
        split_result = _split_regions(inner_rect, target)
        if split_result is None:
            continue
        regions, walls = split_result
        region_sizes = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in regions]
        capacities = _choose_capacities(region_sizes)
        if capacities is None:
            continue

        colors = random.sample(list(SEED_COLORS), len(regions))
        seeds = _pick_seed_positions(regions, colors)

        grid = np.zeros((h, w), dtype=int)
        _draw_partition(grid, outer_frame, walls)
        for r, c, color in seeds:
            grid[r, c] = color

        if not _place_bars(grid, outer_frame, colors, capacities):
            continue

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
