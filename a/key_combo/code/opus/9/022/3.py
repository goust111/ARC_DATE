from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


Coord = Tuple[int, int]
Seed = Tuple[int, int, int]

BLACK = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATIONS = [
    ((-1, 3), (0, -3), (3, 0)),
    ((-1, -3), (0, 3), (3, 0)),
    ((1, 3), (0, -3), (-3, 0)),
    ((1, -3), (0, 3), (-3, 0)),
]
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _extract_legend(grid: np.ndarray) -> Tuple[List[int], Dict[int, int]]:
    top = [int(v) for v in grid[0]]
    legend: List[int] = []
    for value in top:
        if value == BLACK:
            break
        legend.append(value)
    assert 3 <= len(legend) <= 4, "top-left legend must have length 3 or 4"
    assert all(v not in (BLACK, RED) for v in legend), "legend colors must avoid black/red"
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    assert all(v == BLACK for v in top[len(legend) :]), "legend must end before trailing background"
    return legend, {color: i + 1 for i, color in enumerate(legend)}


def _find_seeds(grid: np.ndarray) -> List[Seed]:
    coords = np.argwhere(grid[1:] != BLACK)
    seeds: List[Seed] = []
    for rr, c in coords:
        r = int(rr) + 1
        color = int(grid[r, c])
        assert color != RED, "red is reserved for intersections"
        seeds.append((r, int(c), color))
    return seeds


def _distance_ok(a: Coord, b: Coord) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 3


def _all_far_enough(points: Sequence[Coord]) -> bool:
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            if not _distance_ok(a, b):
                return False
    return True


def _compute_paint_info(grid: np.ndarray):
    grid = np.asarray(grid, dtype=int)
    legend, legend_map = _extract_legend(grid)
    legend_len = len(legend)
    seeds = _find_seeds(grid)
    obstacles = {(r, c) for r, c, _ in seeds}

    intents: Dict[Coord, Set[int]] = {}
    per_seed_paths: Dict[Coord, List[Coord]] = {}
    blocked_by_obstacle = False
    blocked_by_boundary = False

    for r, c, color in seeds:
        if color not in legend_map:
            continue
        reach = legend_map[color]
        path_cells: List[Coord] = []
        for dr, dc in DIRS4:
            for step in range(1, reach + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not _in_bounds(grid.shape, rr, cc):
                    blocked_by_boundary = True
                    break
                if rr != 0 and (rr, cc) in obstacles:
                    blocked_by_obstacle = True
                    break
                if rr == 0 and cc < legend_len and int(grid[0, cc]) != BLACK:
                    continue
                cell = (rr, cc)
                path_cells.append(cell)
                intents.setdefault(cell, set()).add(color)
        per_seed_paths[(r, c)] = path_cells

    red_cells = {
        cell
        for cell, colors in intents.items()
        if len(colors) >= 2 and not (cell[0] == 0 and cell[1] < legend_len)
    }
    return {
        "legend": legend,
        "legend_map": legend_map,
        "legend_len": legend_len,
        "seeds": seeds,
        "intents": intents,
        "paths": per_seed_paths,
        "red_cells": red_cells,
        "blocked_by_obstacle": blocked_by_obstacle,
        "blocked_by_boundary": blocked_by_boundary,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-edge legend positions as ray lengths and color ray overlaps red.

    Input structure:
    - The grid is black except for a contiguous legend string starting at the top-left corner and isolated non-top single pixels.
    - Legend colors are distinct and their left-to-right positions encode lengths 1, 2, 3, ...
    - Some isolated pixels use legend colors while others use colors absent from the legend.

    Rule:
    - Read the top legend and map each legend color to its 1-based position index.
    - For each non-top isolated pixel whose color appears in the legend, extend rays up, down, left, and right for exactly that many steps, truncating at the boundary and stopping before any original non-top foreground pixel.
    - Delete isolated pixels whose colors are absent from the legend, keep the legend unchanged, and recolor every cell claimed by different ray colors to red.

    Key insight:
    - The legend is not a template to copy: its spatial order is a color-to-range lookup table, so changing legend order changes which ray crossings can exist.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _compute_paint_info(g)
    out = g.copy()

    for r, c, color in info["seeds"]:
        if color not in info["legend_map"]:
            out[r, c] = BLACK

    for (r, c), colors in info["intents"].items():
        if r == 0 and c < info["legend_len"] and int(g[0, c]) != BLACK:
            continue
        out[r, c] = RED if len(colors) >= 2 else next(iter(colors))

    return out


def _pick_orientation_positions(height: int, width: int):
    options = list(ORIENTATIONS)
    random.shuffle(options)
    for b_off, u1_off, u2_off in options:
        for _ in range(120):
            r = random.randint(1, height - 1)
            c = random.randint(0, width - 1)
            pts = [
                (r, c),
                (r + b_off[0], c + b_off[1]),
                (r + u1_off[0], c + u1_off[1]),
                (r + u2_off[0], c + u2_off[1]),
            ]
            if not all(_in_bounds((height, width), rr, cc) and rr > 0 for rr, cc in pts):
                continue
            if _all_far_enough(pts):
                return pts
    return None


def _candidate_edge_positions(
    height: int,
    width: int,
    used: Sequence[Coord],
    *,
    prefer_unique_axes: bool,
) -> List[Coord]:
    candidates: List[Coord] = []
    for r in range(1, height):
        for c in (1, width - 2):
            pt = (r, c)
            if any(not _distance_ok(pt, other) for other in used):
                continue
            if prefer_unique_axes and any(r == ur or c == uc for ur, uc in used):
                continue
            candidates.append(pt)
    random.shuffle(candidates)
    return candidates


def _build_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 16)
    legend_len = random.choice([3, 4])

    legend = random.sample(PALETTE, legend_len)
    unmatched_pool = [c for c in PALETTE if c not in legend]
    if len(unmatched_pool) < 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, :legend_len] = np.array(legend, dtype=int)

    base = _pick_orientation_positions(height, width)
    if base is None:
        return None
    a_pos, b_pos, u1_pos, u2_pos = base

    seeds: List[Seed] = []
    color_b = legend[0]
    color_c = legend[1]
    color_a = legend[-1]
    unmatched_colors = random.sample(unmatched_pool, 2)

    seeds.append((a_pos[0], a_pos[1], color_a))
    seeds.append((b_pos[0], b_pos[1], color_b))
    seeds.append((u1_pos[0], u1_pos[1], unmatched_colors[0]))
    seeds.append((u2_pos[0], u2_pos[1], unmatched_colors[1]))

    used = [(r, c) for r, c, _ in seeds]
    edge_candidates = _candidate_edge_positions(
        height, width, used, prefer_unique_axes=True
    )
    if not edge_candidates:
        return None
    c_pos = edge_candidates.pop()
    seeds.append((c_pos[0], c_pos[1], color_c))
    used.append(c_pos)

    if legend_len == 4:
        color_d = legend[2]
        edge_candidates = _candidate_edge_positions(
            height, width, used, prefer_unique_axes=True
        )
        if not edge_candidates:
            return None
        d_pos = edge_candidates.pop()
        seeds.append((d_pos[0], d_pos[1], color_d))

    coords = [(r, c) for r, c, _ in seeds]
    if not _all_far_enough(coords):
        return None

    for r, c, color in seeds:
        grid[r, c] = color
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _compute_paint_info(grid)
        out = main(grid)
    except AssertionError:
        return False

    seeds = info["seeds"]
    if not (5 <= len(seeds) <= 8):
        return False

    matched = [(r, c, color) for r, c, color in seeds if color in info["legend_map"]]
    unmatched = [(r, c, color) for r, c, color in seeds if color not in info["legend_map"]]
    if len(matched) < 3 or len(unmatched) < 2:
        return False
    if len({color for _, _, color in seeds}) != len(seeds):
        return False
    if not _all_far_enough([(r, c) for r, c, _ in seeds]):
        return False

    if not info["blocked_by_obstacle"]:
        return False
    if not info["blocked_by_boundary"]:
        return False
    if not info["red_cells"]:
        return False

    if np.array_equal(out, grid):
        return False
    if any(out[r, c] != BLACK for r, c, _ in unmatched):
        return False
    if any(out[r, c] != color for r, c, color in matched):
        return False

    for r, c, _ in seeds:
        modified = grid.copy()
        modified[r, c] = BLACK
        if np.array_equal(main(modified), out):
            return False

    legend = info["legend"]
    if len(matched) >= 2:
        swap_i = 0
        swap_j = len(legend) - 1
        swapped = grid.copy()
        swapped[0, swap_i], swapped[0, swap_j] = swapped[0, swap_j], swapped[0, swap_i]
        if np.array_equal(main(swapped), out):
            return False

    signature_cells = set(info["red_cells"])
    for r, c, color in matched:
        signature_cells.update(info["paths"][(r, c)])
        if color == legend[-1] and len(info["paths"][(r, c)]) < 4:
            return False
    if len(signature_cells) < 8:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
