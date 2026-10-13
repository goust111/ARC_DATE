from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_TO_DIRS = {
    (-1, -1): [(-1, 0), (0, -1)],
    (-1, 1): [(-1, 0), (0, 1)],
    (1, -1): [(1, 0), (0, -1)],
    (1, 1): [(1, 0), (0, 1)],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _degree_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            val = 0
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    val += 1
            deg[r, c] = val
    return deg


def _seed_connected_mask(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    raw_open = (grid != GREY) & (grid != ORANGE)
    seed_coords = [
        (int(r), int(c))
        for r, c in np.argwhere(raw_open & (grid != BLACK))
    ]
    if not seed_coords:
        return np.zeros_like(raw_open), []

    connected = np.zeros_like(raw_open, dtype=bool)
    q = deque(seed_coords)
    for r, c in seed_coords:
        connected[r, c] = True

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if _in_bounds(grid.shape, rr, cc) and raw_open[rr, cc] and not connected[rr, cc]:
                connected[rr, cc] = True
                q.append((rr, cc))

    return connected, seed_coords


def _detect_blocked_dirs(passable: np.ndarray, orange_mask: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    h, w = passable.shape
    deg = _degree_map(passable)
    blocked: dict[tuple[int, int], set[tuple[int, int]]] = {}

    for or_r, or_c in np.argwhere(orange_mask):
        or_r = int(or_r)
        or_c = int(or_c)
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for diag, pair_dirs in DIAG_TO_DIRS.items():
            ir = or_r - diag[0]
            ic = or_c - diag[1]
            if not _in_bounds(passable.shape, ir, ic):
                continue
            if not passable[ir, ic] or deg[ir, ic] < 3:
                continue
            active_dirs = []
            for dr, dc in pair_dirs:
                rr, cc = ir + dr, ic + dc
                if _in_bounds(passable.shape, rr, cc) and passable[rr, cc]:
                    active_dirs.append((dr, dc))
            if len(active_dirs) == 1:
                candidates.append(((ir, ic), active_dirs[0]))

        unique_candidates = sorted(set(candidates))
        if len(unique_candidates) != 1:
            continue
        intersection, blocked_dir = unique_candidates[0]
        blocked.setdefault(intersection, set()).add(blocked_dir)

    return blocked


def _bfs_dist(
    passable: np.ndarray,
    blocked: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
) -> np.ndarray:
    h, w = passable.shape
    dist = -np.ones((h, w), dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = deque([start])

    while q:
        r, c = q.popleft()
        local_blocked = blocked.get((r, c), set())
        for dr, dc in DIRS4:
            if (dr, dc) in local_blocked:
                continue
            rr, cc = r + dr, c + dc
            if not _in_bounds(passable.shape, rr, cc):
                continue
            if not passable[rr, cc] or dist[rr, cc] != -1:
                continue
            dist[rr, cc] = dist[r, c] + 1
            q.append((rr, cc))
    return dist


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    passable, seeds = _seed_connected_mask(g)
    blocked = _detect_blocked_dirs(passable, g == ORANGE)
    seed_colors = [int(g[r, c]) for r, c in seeds]
    dists = [_bfs_dist(passable, blocked, seed) for seed in seeds]

    out = g.copy()
    tie_cells: list[tuple[int, int]] = []
    unreachable: list[tuple[int, int]] = []
    color_sets: dict[tuple[int, int], set[int]] = {}

    for r, c in np.argwhere(passable):
        r = int(r)
        c = int(c)
        best = None
        winners: list[int] = []
        for idx, dist in enumerate(dists):
            val = int(dist[r, c])
            if val < 0:
                continue
            if best is None or val < best:
                best = val
                winners = [idx]
            elif val == best:
                winners.append(idx)
        if best is None:
            unreachable.append((r, c))
            continue

        winner_colors = {seed_colors[idx] for idx in winners}
        color_sets[(r, c)] = set(winner_colors)
        if len(winner_colors) == 1:
            out[r, c] = next(iter(winner_colors))
            continue

        tie_cells.append((r, c))
        chosen = min(winners, key=lambda idx: seeds[idx])
        out[r, c] = seed_colors[chosen]

    details = {
        "passable": passable,
        "seeds": list(seeds),
        "blocked": blocked,
        "tie_cells": tie_cells,
        "unreachable": unreachable,
        "colors": color_sets,
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill seed colors through maze corridors, with orange corner markers
    blocking one outgoing branch at a nearby intersection.
    """
    out, _details = _solve_details(grid)
    return out


def _carve_maze(cell_h: int, cell_w: int, rng: random.Random) -> np.ndarray:
    h = 2 * cell_h + 1
    w = 2 * cell_w + 1
    grid = np.full((h, w), GREY, dtype=int)

    start = (rng.randrange(cell_h), rng.randrange(cell_w))
    visited = {start}
    stack = [start]
    grid[2 * start[0] + 1, 2 * start[1] + 1] = BLACK

    while stack:
        cr, cc = stack[-1]
        neighbors = []
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and (nr, nc) not in visited:
                neighbors.append((nr, nc, dr, dc))
        if not neighbors:
            stack.pop()
            continue

        nr, nc, dr, dc = rng.choice(neighbors)
        visited.add((nr, nc))
        stack.append((nr, nc))

        gr, gc = 2 * cr + 1, 2 * cc + 1
        wr, wc = gr + dr, gc + dc
        tr, tc = 2 * nr + 1, 2 * nc + 1
        grid[wr, wc] = BLACK
        grid[tr, tc] = BLACK

    return grid


def _component_from_edge(mask: np.ndarray, start: tuple[int, int], banned: tuple[int, int]) -> set[tuple[int, int]]:
    comp = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if not _in_bounds(mask.shape, rr, cc):
                continue
            if not mask[rr, cc] or (rr, cc) == banned or (rr, cc) in comp:
                continue
            comp.add((rr, cc))
            q.append((rr, cc))
    return comp


def _gate_options(passable: np.ndarray) -> list[dict]:
    deg = _degree_map(passable)
    options: list[dict] = []
    for ir, ic in np.argwhere(passable & (deg == 3)):
        ir = int(ir)
        ic = int(ic)
        for diag, pair_dirs in DIAG_TO_DIRS.items():
            active_dirs = []
            for dr, dc in pair_dirs:
                rr, cc = ir + dr, ic + dc
                if _in_bounds(passable.shape, rr, cc) and passable[rr, cc]:
                    active_dirs.append((dr, dc))
            if len(active_dirs) != 1:
                continue
            orange_pos = (ir + diag[0], ic + diag[1])
            if not _in_bounds(passable.shape, orange_pos[0], orange_pos[1]):
                continue
            if passable[orange_pos]:
                continue
            blocked_dir = active_dirs[0]
            branch_start = (ir + blocked_dir[0], ic + blocked_dir[1])
            branch_cells = _component_from_edge(passable, branch_start, (ir, ic))
            options.append(
                {
                    "intersection": (ir, ic),
                    "orange": orange_pos,
                    "blocked_dir": blocked_dir,
                    "branch": branch_cells,
                }
            )
    return options


def _pick_seeds(dead_ends: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    max_seed_count = min(len(dead_ends), len(SEED_COLORS), 4)
    if max_seed_count < 2:
        return []
    target = rng.choice([2, 2, 3, 3, 4])
    target = min(target, max_seed_count)
    return rng.sample(dead_ends, target)


def _pick_gates(
    gate_options: list[dict],
    seeds: list[tuple[int, int]],
    intersections: list[tuple[int, int]],
    rng: random.Random,
) -> list[dict]:
    if not gate_options:
        return []

    valid_by_intersection: dict[tuple[int, int], list[dict]] = {}
    seed_set = set(seeds)
    for option in gate_options:
        branch_has_seed = any(cell in seed_set for cell in option["branch"])
        branch_not_all = len(seed_set - option["branch"]) > 0
        if branch_has_seed and branch_not_all:
            valid_by_intersection.setdefault(option["intersection"], []).append(option)

    available = list(valid_by_intersection.keys())
    rng.shuffle(available)
    if not available:
        return []

    max_count = min(3, len(available))
    target = rng.randint(1, max_count)
    chosen: list[dict] = []
    for intersection in available:
        if len(chosen) >= target:
            break
        chosen.append(rng.choice(valid_by_intersection[intersection]))

    if len(intersections) <= len(chosen):
        return []
    return chosen


def _maze_stats(passable: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    deg = _degree_map(passable)
    dead_ends = [
        (int(r), int(c))
        for r, c in np.argwhere(passable & (deg == 1))
    ]
    intersections = [
        (int(r), int(c))
        for r, c in np.argwhere(passable & (deg >= 3))
    ]
    return dead_ends, intersections


def _build_input_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(4, 8)
    cell_w = rng.randint(4, 8)
    grid = _carve_maze(cell_h, cell_w, rng)
    passable = grid == BLACK

    dead_ends, intersections = _maze_stats(passable)
    if len(dead_ends) < 2 or len(intersections) < 2:
        return None

    seeds = _pick_seeds(dead_ends, rng)
    if len(seeds) < 2:
        return None

    gate_options = _gate_options(passable)
    gates = _pick_gates(gate_options, seeds, intersections, rng)
    if not gates:
        return None

    colored = grid.copy()
    colors = rng.sample(SEED_COLORS, len(seeds))
    for (r, c), color in zip(seeds, colors):
        colored[r, c] = int(color)
    for gate in gates:
        or_r, or_c = gate["orange"]
        colored[or_r, or_c] = ORANGE

    solved, details = _solve_details(colored)
    if details["tie_cells"] or details["unreachable"]:
        return None

    passable_cells = {tuple(map(int, rc)) for rc in np.argwhere(details["passable"])}
    if passable_cells != {tuple(map(int, rc)) for rc in np.argwhere(passable)}:
        return None

    gated_intersections = set(details["blocked"].keys())
    if len(gated_intersections) != len(gates):
        return None
    if len(intersections) <= len(gated_intersections):
        return None

    no_gate_solved, _no_gate_details = _solve_details(np.where(colored == ORANGE, GREY, colored))
    gate_changes_fill = any(
        int(solved[r, c]) != int(no_gate_solved[r, c])
        for r, c in np.argwhere(passable)
    )
    if not gate_changes_fill:
        return None

    non_seed_changes = 0
    seed_set = set(seeds)
    for r, c in np.argwhere(passable):
        r = int(r)
        c = int(c)
        if (r, c) not in seed_set and int(solved[r, c]) != BLACK:
            non_seed_changes += 1
    if non_seed_changes < max(6, len(passable_cells) // 4):
        return None

    return colored


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _build_input_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze flood-fill input")
