from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

COLOR_BY_BENDS = {
    0: BLUE,
    1: GREEN,
    2: YELLOW,
}


def _bend_color(bends: int) -> int:
    return COLOR_BY_BENDS.get(int(bends), ORANGE)


def _start_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero(np.asarray(grid)[-1] == RED)]


def _simulate(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts = _start_columns(g)
    if not starts:
        return {
            "starts": [],
            "bends": [],
            "cells": [],
            "heads": [],
            "reached_top": [],
            "stop_reason": [],
            "occupied_blocked": [],
        }

    occupied = g == GREY
    bends = [0 for _ in starts]
    cells = [[(h - 1, c)] for c in starts]
    heads = [(h - 1, c) for c in starts]
    reached_top = [False for _ in starts]
    stop_reason = [None for _ in starts]
    occupied_blocked = [False for _ in starts]
    active = [True for _ in starts]

    for c in starts:
        occupied[h - 1, c] = True

    max_steps = h + w + 5
    for _ in range(max_steps):
        if not any(active):
            break
        for i in range(len(starts)):
            if not active[i]:
                continue
            r, c = heads[i]
            if r == 0:
                active[i] = False
                reached_top[i] = True
                stop_reason[i] = "top"
                continue

            up_r, up_c = r - 1, c
            if occupied[up_r, up_c]:
                right_c = c + 1
                if right_c >= w:
                    active[i] = False
                    stop_reason[i] = "boundary"
                    continue
                if occupied[r, right_c]:
                    active[i] = False
                    stop_reason[i] = "occupied" if g[r, right_c] != GREY else "obstacle"
                    occupied_blocked[i] = g[r, right_c] != GREY
                    continue
                bends[i] += 1
                heads[i] = (r, right_c)
                cells[i].append((r, right_c))
                occupied[r, right_c] = True
                continue

            heads[i] = (up_r, up_c)
            cells[i].append((up_r, up_c))
            occupied[up_r, up_c] = True
            if up_r == 0:
                active[i] = False
                reached_top[i] = True
                stop_reason[i] = "top"

    return {
        "starts": starts,
        "bends": bends,
        "cells": cells,
        "heads": heads,
        "reached_top": reached_top,
        "stop_reason": stop_reason,
        "occupied_blocked": occupied_blocked,
    }


def _simulate_independent(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts = _start_columns(g)

    bends: list[int] = []
    cells: list[list[tuple[int, int]]] = []
    reached_top: list[bool] = []
    stop_reason: list[str | None] = []

    for c in starts:
        single = np.zeros((h, w), dtype=int)
        single[g == GREY] = GREY
        single[h - 1, c] = RED
        result = _simulate(single)
        bends.append(int(result["bends"][0]))
        cells.append(list(result["cells"][0]))
        reached_top.append(bool(result["reached_top"][0]))
        stop_reason.append(result["stop_reason"][0])

    return {
        "starts": starts,
        "bends": bends,
        "cells": cells,
        "reached_top": reached_top,
        "stop_reason": stop_reason,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow all bottom starts upward together, then recolor each finished path by its bend count.

    Input structure:
    - The grid is a black background with several red start pixels on the bottom row.
    - Gray obstacle pixels appear only in interior rows and are pairwise non-adjacent.
    - Every non-black, non-gray path cell created during growth belongs to exactly one start.

    Rule:
    - Treat all starts as simultaneously active heads, resolved left-to-right within each step to encode the
      "leftmost start wins" competition rule for contested cells.
    - Each active head first tries to grow one cell upward; if that target cell is already non-black or gray,
      it instead tries to grow one cell to the right and counts one bend.
    - If the right cell is also blocked or outside the grid, that path stops; a path also stops once it reaches
      the top row.
    - After all paths finish, recolor every cell of each path by its total bend count: 0->blue, 1->green,
      2->yellow, 3 or more->orange. Preserve gray obstacles and black background.

    Key insight:
    - The final color is determined by simulating interacting upward/rightward growth, because other paths occupy
      space and can change when a bend succeeds or fails.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    out[g == GREY] = GREY

    sim = _simulate(g)
    for bends, path_cells in zip(sim["bends"], sim["cells"]):
        color = _bend_color(int(bends))
        for r, c in path_cells:
            out[int(r), int(c)] = color
    return out


def _non_adjacent(existing: set[tuple[int, int]], r: int, c: int) -> bool:
    for rr, cc in existing:
        if max(abs(int(rr) - int(r)), abs(int(cc) - int(c))) <= 1:
            return False
    return True


def _competition_obstacles(h: int, start_c: int, gap: int) -> list[tuple[int, int]]:
    return [(h - 2 - 2 * j, start_c + j) for j in range(gap)]


def _success_obstacles(h: int, start_c: int, count: int, row_offset: int) -> list[tuple[int, int]]:
    return [(h - 2 - row_offset - 2 * j, start_c + j) for j in range(count)]


def _valid_obstacle_positions(
    h: int,
    w: int,
    starts: list[int],
    obstacles: list[tuple[int, int]],
) -> bool:
    seen: set[tuple[int, int]] = set()
    for r, c in obstacles:
        if not (0 < r < h - 1 and 0 <= c < w):
            return False
        if c in starts and r == h - 1:
            return False
        if not _non_adjacent(seen, r, c):
            return False
        seen.add((int(r), int(c)))
    return True


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    path_count = rng.randint(3, 5)
    gaps = [rng.choice((2, 3)) for _ in range(path_count - 1)]
    left_margin = rng.randint(0, 2)
    starts = [left_margin]
    for gap in gaps:
        starts.append(starts[-1] + gap)

    comp_i = rng.randrange(path_count - 1)
    comp_gap = starts[comp_i + 1] - starts[comp_i]

    outside = [i for i in range(path_count) if i not in {comp_i, comp_i + 1}]
    success_i = max(outside) if rng.random() < 0.7 else rng.choice(outside)

    success_cap = 3
    if success_i < path_count - 1:
        success_cap = min(success_cap, starts[success_i + 1] - starts[success_i] - 1)
    success_bends = rng.randint(1, max(1, success_cap))

    right_margin = rng.randint(3, 5)
    if success_i == path_count - 1:
        right_margin = max(right_margin, success_bends + 2)
    w = starts[-1] + right_margin + 1
    if w > 16:
        return None

    max_bend_count = max(comp_gap, success_bends)
    h = rng.randint(max(11, 2 * max_bend_count + 6), 15)

    obstacles = _competition_obstacles(h, starts[comp_i], comp_gap)
    row_offset = rng.choice((0, 1))
    obstacles.extend(_success_obstacles(h, starts[success_i], success_bends, row_offset))

    if not _valid_obstacle_positions(h, w, starts, obstacles):
        return None

    obstacle_set = {(int(r), int(c)) for r, c in obstacles}

    max_extra = min(8 - len(obstacle_set), 2)
    extra_count = rng.randint(0, max_extra)
    extra_candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(w)
        if (r, c) not in obstacle_set and _non_adjacent(obstacle_set, r, c)
    ]
    rng.shuffle(extra_candidates)
    for r, c in extra_candidates:
        if len(obstacle_set) >= len(obstacles) + extra_count:
            break
        if _non_adjacent(obstacle_set, r, c):
            obstacle_set.add((int(r), int(c)))

    if not (3 <= len(obstacle_set) <= 8):
        return None

    grid = np.zeros((h, w), dtype=int)
    for c in starts:
        grid[h - 1, c] = RED
    for r, c in obstacle_set:
        grid[r, c] = GREY
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    starts = _start_columns(g)
    if not (3 <= len(starts) <= 5):
        return False
    if np.count_nonzero(g[:-1] == RED) != 0:
        return False
    if any(starts[i + 1] - starts[i] < 2 for i in range(len(starts) - 1)):
        return False

    obstacle_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    if not (3 <= len(obstacle_coords) <= 8):
        return False
    for r, c in obstacle_coords:
        if not (0 < r < h - 1):
            return False
    for i, (r0, c0) in enumerate(obstacle_coords):
        for r1, c1 in obstacle_coords[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False

    sim = _simulate(g)
    independent = _simulate_independent(g)

    if len(set(sim["bends"])) == 1:
        return False
    if len(set(_bend_color(b) for b in sim["bends"])) < 2:
        return False

    for bends, top in zip(sim["bends"], sim["reached_top"]):
        if not top and bends < 1:
            return False

    if not any(int(a) != int(b) for a, b in zip(sim["bends"], independent["bends"])):
        return False
    if not any(sim["occupied_blocked"]):
        return False
    if not any(b > 0 for b in sim["bends"]):
        return False
    if not any(b >= 2 for b in sim["bends"]):
        return False

    out = main(g)
    if out.shape != g.shape:
        return False
    if np.any(out[g == GREY] != GREY):
        return False
    if np.any(out[(g == BLACK) & (out == RED)]):
        return False

    unique_path_colors = {int(out[r, c]) for path in sim["cells"] for r, c in path}
    if len(unique_path_colors) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        grid = _build_candidate(rng)
        if grid is not None and _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid competitive-growth instance")
