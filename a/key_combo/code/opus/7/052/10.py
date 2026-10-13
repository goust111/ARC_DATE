from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHO_DIRS:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIAG_DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray, conn8: bool) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    neighbor_fn = _neighbors8 if conn8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for nr, nc in neighbor_fn(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    border_cells = []
    for c in range(w):
        border_cells.append((0, c))
        border_cells.append((h - 1, c))
    for r in range(1, h - 1):
        border_cells.append((r, 0))
        border_cells.append((r, w - 1))

    seeds = []
    for r, c in border_cells:
        value = int(grid[r, c])
        if value not in (BLACK, GRAY):
            seeds.append((r, c, value))
    seeds.sort()
    return seeds


def _gray_distances(grid: np.ndarray, seed: tuple[int, int, int]) -> dict[tuple[int, int], int]:
    sr, sc, _color = seed
    starts = []
    seen_starts = set()
    for nr, nc in _neighbors8(sr, sc):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GRAY and (nr, nc) not in seen_starts:
            seen_starts.add((nr, nc))
            starts.append((nr, nc))

    dist: dict[tuple[int, int], int] = {}
    q = deque()
    for cell in starts:
        dist[cell] = 0
        q.append(cell)

    while q:
        r, c = q.popleft()
        base = dist[(r, c)]
        for nr, nc in _neighbors8(r, c):
            if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) != GRAY or (nr, nc) in dist:
                continue
            dist[(nr, nc)] = base + 1
            q.append((nr, nc))
    return dist


def _cavity_records(grid: np.ndarray) -> list[dict]:
    cavities = []
    for cells in _components(grid == BLACK, conn8=False):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in cells):
            continue
        boundary_gray = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GRAY:
                    boundary_gray.add((nr, nc))
        if boundary_gray:
            cavities.append(
                {
                    "cells": cells,
                    "area": len(cells),
                    "bbox": _bbox(cells),
                    "boundary_gray": boundary_gray,
                }
            )
    cavities.sort(key=lambda rec: rec["bbox"])
    return cavities


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    seeds = _find_seeds(g)
    seed_dists = [_gray_distances(g, seed) for seed in seeds]
    cavities = _cavity_records(g)
    gray_components = _components(g == GRAY, conn8=True)

    for cavity in cavities:
        reachable = []
        for idx, dist_map in enumerate(seed_dists):
            hits = [dist_map[cell] for cell in cavity["boundary_gray"] if cell in dist_map]
            if not hits:
                continue
            seed = seeds[idx]
            reachable.append(
                {
                    "seed_index": idx,
                    "seed": seed,
                    "distance": min(hits),
                }
            )
        reachable.sort(key=lambda item: (item["distance"], item["seed"][0], item["seed"][1]))
        cavity["reachable"] = reachable
        cavity["winner"] = None if not reachable else reachable[0]["seed_index"]
    return {
        "grid": g,
        "seeds": seeds,
        "cavities": cavities,
        "gray_components": gray_components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only seed-reachable enclosed cavities along the gray line drawing, then crop the largest filled chamber.

    Input structure:
    - The background is black, gray pixels form one or more thin line-drawing components, and 2-3 colored seed pixels sit on the outer border.
    - Some black 4-connected regions are fully enclosed by gray loops, while at least one other enclosed region can be disconnected from every seed.
    - Different seeds may touch the same gray network from different border entry points, so cavities can compete for ownership by gray-path distance.

    Rule:
    - Detect every enclosed black cavity whose 4-neighbors on the boundary are gray.
    - For each seed, walk only on gray pixels with 8-connectivity starting from the gray pixels adjacent to that seed.
    - A cavity is fillable by a seed if one of its gray boundary pixels is reachable from that seed; if multiple seeds can fill it, choose the seed with the shortest gray-path distance to the cavity boundary, breaking ties by smaller seed row then smaller seed column.
    - Fill every reachable cavity with its winning seed color, then select the uniquely largest filled cavity and return its bounding box expanded by one cell in every direction, keeping only the winner's fill and gray boundary pixels and setting everything else in the crop to black.

    Key insight:
    - The answer is not the geometrically largest cavity in the whole drawing: only cavities that are connected to a border seed through gray-line reachability enter the competition, and ownership is resolved by shortest distance along the gray topology.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    g = analysis["grid"]
    out = g.copy()

    filled = []
    for cavity in analysis["cavities"]:
        winner = cavity["winner"]
        if winner is None:
            continue
        color = analysis["seeds"][winner][2]
        for r, c in cavity["cells"]:
            out[r, c] = color
        filled.append(cavity)

    assert filled, "expected at least one seed-reachable cavity"
    winner = max(filled, key=lambda rec: rec["area"])
    r0, c0, r1, c1 = winner["bbox"]
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(g.shape[0] - 1, r1 + 1)
    c1 = min(g.shape[1] - 1, c1 + 1)

    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    gray_mask = g[r0 : r1 + 1, c0 : c1 + 1] == GRAY
    crop[gray_mask] = GRAY
    fill_color = analysis["seeds"][winner["winner"]][2]
    for r, c in winner["cells"]:
        crop[r - r0, c - c0] = fill_color
    return crop


def _rect_loop(top: int, left: int, inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    cells = set()
    for c in range(left, left + outer_w):
        cells.add((top, c))
        cells.add((top + outer_h - 1, c))
    for r in range(top, top + outer_h):
        cells.add((r, left))
        cells.add((r, left + outer_w - 1))
    return cells


def _add_hline(cells: set[tuple[int, int]], row: int, c0: int, c1: int) -> None:
    lo, hi = sorted((c0, c1))
    for c in range(lo, hi + 1):
        cells.add((row, c))


def _add_vline(cells: set[tuple[int, int]], col: int, r0: int, r1: int) -> None:
    lo, hi = sorted((r0, r1))
    for r in range(lo, hi + 1):
        cells.add((r, col))


def _render_grid(
    shape: tuple[int, int],
    gray_cells: set[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in gray_cells:
        grid[r, c] = GRAY
    for r, c, color in seeds:
        grid[r, c] = color
    return grid


def _valid_grid(grid: np.ndarray, seed_count: int) -> bool:
    analysis = _analyze(grid)
    cavities = analysis["cavities"]
    if len(analysis["seeds"]) != seed_count:
        return False
    if len({seed[2] for seed in analysis["seeds"]}) != seed_count:
        return False
    if len(cavities) != 4:
        return False
    if len({cavity["area"] for cavity in cavities}) != len(cavities):
        return False
    if len(analysis["gray_components"]) != 3:
        return False

    unfilled = [cavity for cavity in cavities if cavity["winner"] is None]
    if len(unfilled) < 1:
        return False

    contested = [cavity for cavity in cavities if len(cavity["reachable"]) >= 2]
    if len(contested) < 2:
        return False

    exclusive = [cavity for cavity in cavities if len(cavity["reachable"]) == 1]
    if len(exclusive) < 1:
        return False

    winner_indices = [cavity["winner"] for cavity in cavities if cavity["winner"] is not None]
    if len(set(winner_indices)) < 2:
        return False

    if seed_count == 3 and 2 not in winner_indices:
        return False

    filled_areas = [cavity["area"] for cavity in cavities if cavity["winner"] is not None]
    if not filled_areas:
        return False
    if max(cavity["area"] for cavity in unfilled) <= max(filled_areas):
        return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if np.count_nonzero(out == GRAY) == 0:
        return False
    if len(set(map(int, out.flatten())) - {BLACK, GRAY}) != 1:
        return False
    return True


def _sample_pairs(rng: random.Random) -> list[tuple[int, int]]:
    left_shared = rng.choice(((2, 2), (2, 3), (2, 4)))
    right_shared = rng.choice(((3, 3), (2, 5), (3, 4)))
    exclusive = rng.choice(((3, 5), (4, 4)))
    distractor = rng.choice(((4, 5), (5, 5)))
    return [left_shared, right_shared, exclusive, distractor]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 24)
    w = rng.randint(24, 28)
    use_three_seeds = rng.random() < 0.55
    seed_count = 3 if use_three_seeds else 2
    colors = rng.sample(SEED_PALETTE, seed_count)

    shared_a, shared_b, exclusive_pair, distractor_pair = _sample_pairs(rng)
    shared_pairs = [shared_a, shared_b]

    shared_outer_heights = [inner_h + 2 for inner_h, _ in shared_pairs]
    shared_row_min = max(shared_outer_heights) + 3
    shared_row_max = min(12, h - 9)
    if shared_row_min > shared_row_max:
        return None
    shared_row = rng.randint(shared_row_min, shared_row_max)

    gray_cells: set[tuple[int, int]] = set()

    exclusive_outer_h = exclusive_pair[0] + 2
    exclusive_outer_w = exclusive_pair[1] + 2
    distractor_outer_h = distractor_pair[0] + 2
    distractor_outer_w = distractor_pair[1] + 2

    left1 = 4
    left2 = w - (shared_pairs[1][1] + 2) - 4
    if left2 - (left1 + shared_pairs[0][1] + 2) < 4:
        return None
    shared_lefts = [left1, left2]

    _add_hline(gray_cells, shared_row, 1, w - 2)
    for (inner_h, inner_w), left in zip(shared_pairs, shared_lefts):
        top = shared_row - (inner_h + 2) - 2
        if top < 1:
            return None
        gray_cells.update(_rect_loop(top, left, inner_h, inner_w))
        attach_col = left + (inner_w + 2) // 2
        _add_vline(gray_cells, attach_col, top + inner_h + 1, shared_row)

    if use_three_seeds:
        inner_h, inner_w = exclusive_pair
        exclusive_left = w - exclusive_outer_w - 2
        top = h - exclusive_outer_h - 3
        gray_cells.update(_rect_loop(top, exclusive_left, inner_h, inner_w))
        attach_col = exclusive_left + (inner_w + 2) // 2
        _add_vline(gray_cells, attach_col, top + inner_h + 1, h - 2)
    else:
        inner_h, inner_w = exclusive_pair
        exclusive_left = 1
        top = h - exclusive_outer_h - 3
        gray_cells.update(_rect_loop(top, exclusive_left, inner_h, inner_w))
        attach_row = top + (inner_h + 2) // 2
        _add_vline(gray_cells, exclusive_left, shared_row + 2, attach_row)

    inner_h, inner_w = distractor_pair
    distractor_top = h - distractor_outer_h - 2
    distractor_left = 1 if use_three_seeds else w - distractor_outer_w - 2
    gray_cells.update(_rect_loop(distractor_top, distractor_left, inner_h, inner_w))

    if any(not _in_bounds((h, w), r, c) for r, c in gray_cells):
        return None

    if use_three_seeds:
        seed_c_col = exclusive_left + exclusive_outer_w // 2
        seeds = [
            (shared_row, 0, colors[0]),
            (shared_row, w - 1, colors[1]),
            (h - 1, seed_c_col, colors[2]),
        ]
    else:
        seeds = [
            (shared_row + 1, 0, colors[0]),
            (shared_row, w - 1, colors[1]),
        ]

    for r, c, _color in seeds:
        if not _in_bounds((h, w), r, c):
            return None
        if any((nr, nc) in gray_cells for nr, nc in _neighbors8(r, c)):
            continue
        return None

    grid = _render_grid((h, w), gray_cells, seeds)
    if _valid_grid(grid, seed_count):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
