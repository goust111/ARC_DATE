from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SIDES = ("top", "bottom", "left", "right")
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator bounds.
MIN_CELL_H = 5
MAX_CELL_H = 14
MIN_CELL_W = 5
MAX_CELL_W = 14
MIN_PATH_LEN = 7
MAX_BRANCHES = 4
MAX_BRANCH_LEN = 5
MIN_TURNS = 2
GEN_ATTEMPTS = 300

COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _pixel_neighbors(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _cell_in_bounds(cell: tuple[int, int], cell_h: int, cell_w: int) -> bool:
    r, c = cell
    return 0 <= r < cell_h and 0 <= c < cell_w


def _cell_neighbors(cell: tuple[int, int], cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < cell_h and 0 <= nc < cell_w:
            out.append((nr, nc))
    return out


def _boundary_sides(cell: tuple[int, int], cell_h: int, cell_w: int) -> list[str]:
    r, c = cell
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == cell_h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == cell_w - 1:
        sides.append("right")
    return sides


def _is_boundary(cell: tuple[int, int], cell_h: int, cell_w: int) -> bool:
    return bool(_boundary_sides(cell, cell_h, cell_w))


def _is_noncorner_boundary(cell: tuple[int, int], cell_h: int, cell_w: int) -> bool:
    return len(_boundary_sides(cell, cell_h, cell_w)) == 1


def _random_side_cell(side: str, cell_h: int, cell_w: int) -> tuple[int, int]:
    if side == "top":
        return (0, random.randint(1, cell_w - 2))
    if side == "bottom":
        return (cell_h - 1, random.randint(1, cell_w - 2))
    if side == "left":
        return (random.randint(1, cell_h - 2), 0)
    return (random.randint(1, cell_h - 2), cell_w - 1)


def _cell_to_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    return (2 * cell[0] + 1, 2 * cell[1] + 1)


def _opening_pixel(cell: tuple[int, int], side: str, height: int, width: int) -> tuple[int, int]:
    pr, pc = _cell_to_pixel(cell)
    if side == "top":
        return (0, pc)
    if side == "bottom":
        return (height - 1, pc)
    if side == "left":
        return (pr, 0)
    return (pr, width - 1)


def _trace_guided_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    walkable = grid == BLACK
    walkable[start] = True
    walkable[goal] = True

    colored = (grid != BLACK) & (grid != GREY)
    colored[start] = False
    colored[goal] = False

    path = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        if cur == goal:
            return path

        options: list[tuple[int, int]] = []
        for nxt in _pixel_neighbors(grid.shape, cur):
            if not walkable[nxt]:
                continue
            if prev is not None and nxt == prev:
                continue
            options.append(nxt)

        if not options:
            return None

        if len(options) == 1:
            chosen = options[0]
        else:
            signs = [nxt for nxt in _pixel_neighbors(grid.shape, cur) if colored[nxt]]
            if len(signs) != 1:
                return None

            sr, sc = signs[0]
            dr = sr - cur[0]
            dc = sc - cur[1]
            chosen = (cur[0] - dr, cur[1] - dc)
            if chosen not in options:
                return None

        if chosen in visited:
            return None

        path.append(chosen)
        visited.add(chosen)
        prev, cur = cur, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the sign-guided route through the maze and recolor only that entrance-to-exit path.

    Input structure:
    - Black cells are passable maze corridors and grey cells are one-pixel walls.
    - Exactly two non-black, non-grey border cells mark the entrance and the exit.
    - Interior non-black, non-grey cells are wall-mounted signposts beside branch points.

    Rule:
    - Treat black cells together with the two colored border markers as the walkable network.
    - Starting from the true entrance, move through corridor cells; when only one forward option exists, continue straight along it.
    - At a branch, inspect the unique adjacent signpost and choose the corridor in the opposite direction from that signpost.
    - Stop at the other border marker and recolor every visited corridor cell, including both endpoints, to the entrance color; keep walls, signposts, and unused corridors unchanged.

    Key insight:
    - The signposts encode direction by position rather than by color: a sign on one side of a branch means the path continues out through the opposite side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colored = np.argwhere((g != BLACK) & (g != GREY))
    endpoints = [
        (int(r), int(c))
        for r, c in colored
        if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1)
    ]
    assert len(endpoints) == 2, "expected exactly two colored border endpoints"

    traces: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    a, b = endpoints
    for start, goal in ((a, b), (b, a)):
        path = _trace_guided_path(g, start, goal)
        if path is not None:
            traces.append((start, path))

    assert len(traces) == 1, "maze must determine a unique entrance direction"
    start, path = traces[0]
    fill_color = int(g[start])

    out = g.copy()
    for r, c in path:
        out[r, c] = fill_color
    return out


def _future_degree(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    cell_h: int,
    cell_w: int,
) -> int:
    return sum(
        1
        for nn in _cell_neighbors(nxt, cell_h, cell_w)
        if nn != cur and nn not in used
    )


def _choose_next_step(
    cur: tuple[int, int],
    options: list[tuple[int, int]],
    prev_dir: tuple[int, int] | None,
    used: set[tuple[int, int]],
    cell_h: int,
    cell_w: int,
    turns: int,
) -> tuple[int, int]:
    weights: list[float] = []
    for nxt in options:
        step_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
        future = _future_degree(nxt, cur, used, cell_h, cell_w)
        weight = 1.0 + 0.45 * future
        if prev_dir is not None and step_dir != prev_dir:
            weight *= 1.8 if turns < MIN_TURNS + 1 else 1.25
        elif prev_dir is not None:
            weight *= 1.1
        if _is_boundary(nxt, cell_h, cell_w):
            weight *= 0.45
        weights.append(weight)
    return random.choices(options, weights=weights, k=1)[0]


def _generate_main_path(cell_h: int, cell_w: int) -> tuple[list[tuple[int, int]], str, str] | None:
    start_side = random.choice(SIDES)
    start = _random_side_cell(start_side, cell_h, cell_w)
    min_len = random.randint(MIN_PATH_LEN, max(MIN_PATH_LEN, min(18, cell_h * cell_w // 3)))
    max_len = min(cell_h * cell_w, min_len + random.randint(4, 10))

    path = [start]
    used = {start}
    prev_dir: tuple[int, int] | None = None
    turns = 0

    while len(path) <= max_len:
        cur = path[-1]
        if (
            cur != start
            and len(path) >= min_len
            and turns >= MIN_TURNS
            and _is_noncorner_boundary(cur, cell_h, cell_w)
        ):
            end_side = _boundary_sides(cur, cell_h, cell_w)[0]
            return path, start_side, end_side

        options: list[tuple[int, int]] = []
        for nxt in _cell_neighbors(cur, cell_h, cell_w):
            if nxt in used:
                continue
            if len(path) + 1 < min_len and _is_boundary(nxt, cell_h, cell_w):
                continue

            can_finish = (
                nxt != start
                and len(path) + 1 >= min_len
                and turns >= MIN_TURNS - 1
                and _is_noncorner_boundary(nxt, cell_h, cell_w)
            )
            if _future_degree(nxt, cur, used, cell_h, cell_w) == 0 and not can_finish:
                continue
            options.append(nxt)

        if not options:
            return None

        nxt = _choose_next_step(cur, options, prev_dir, used, cell_h, cell_w, turns)
        step_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
        if prev_dir is not None and step_dir != prev_dir:
            turns += 1
        path.append(nxt)
        used.add(nxt)
        prev_dir = step_dir

    return None


def _sign_would_touch_border(
    cell: tuple[int, int], sign_dir: tuple[int, int], cell_h: int, cell_w: int
) -> bool:
    r, c = cell
    dr, dc = sign_dir
    return (
        (dr == -1 and r == 0)
        or (dr == 1 and r == cell_h - 1)
        or (dc == -1 and c == 0)
        or (dc == 1 and c == cell_w - 1)
    )


def _branch_candidates(
    path: list[tuple[int, int]], cell_h: int, cell_w: int
) -> list[dict[str, object]]:
    occupied = set(path)
    candidates: list[dict[str, object]] = []
    for idx in range(1, len(path) - 1):
        prev_cell = path[idx - 1]
        cur = path[idx]
        next_cell = path[idx + 1]

        back_dir = (prev_cell[0] - cur[0], prev_cell[1] - cur[1])
        out_dir = (next_cell[0] - cur[0], next_cell[1] - cur[1])
        if back_dir[0] == out_dir[0] or back_dir[1] == out_dir[1]:
            continue

        branch_dir = (-back_dir[0], -back_dir[1])
        sign_dir = (-out_dir[0], -out_dir[1])
        branch_start = (cur[0] + branch_dir[0], cur[1] + branch_dir[1])
        sign_neighbor = (cur[0] + sign_dir[0], cur[1] + sign_dir[1])

        if not _cell_in_bounds(branch_start, cell_h, cell_w):
            continue
        if branch_start in occupied:
            continue
        if _cell_in_bounds(sign_neighbor, cell_h, cell_w) and sign_neighbor in occupied:
            continue
        if _sign_would_touch_border(cur, sign_dir, cell_h, cell_w):
            continue

        max_len = 0
        probe = branch_start
        while _cell_in_bounds(probe, cell_h, cell_w) and probe not in occupied:
            max_len += 1
            probe = (probe[0] + branch_dir[0], probe[1] + branch_dir[1])

        if max_len >= 1:
            candidates.append(
                {
                    "index": idx,
                    "cell": cur,
                    "branch_dir": branch_dir,
                    "sign_dir": sign_dir,
                    "max_len": max_len,
                }
            )
    return candidates


def _select_branches(
    path: list[tuple[int, int]], candidates: list[dict[str, object]], cell_h: int, cell_w: int
) -> list[dict[str, object]]:
    occupied = set(path)
    chosen: list[dict[str, object]] = []
    target = random.randint(1, min(MAX_BRANCHES, len(candidates)))
    pool = candidates[:]
    random.shuffle(pool)

    for cand in pool:
        cur = cand["cell"]
        branch_dir = cand["branch_dir"]
        sign_dir = cand["sign_dir"]
        assert isinstance(cur, tuple)
        assert isinstance(branch_dir, tuple)
        assert isinstance(sign_dir, tuple)

        sign_neighbor = (cur[0] + sign_dir[0], cur[1] + sign_dir[1])
        if _cell_in_bounds(sign_neighbor, cell_h, cell_w) and sign_neighbor in occupied:
            continue

        available: list[tuple[int, int]] = []
        probe = (cur[0] + branch_dir[0], cur[1] + branch_dir[1])
        while _cell_in_bounds(probe, cell_h, cell_w) and probe not in occupied:
            available.append(probe)
            probe = (probe[0] + branch_dir[0], probe[1] + branch_dir[1])

        if not available:
            continue

        length = random.randint(1, min(MAX_BRANCH_LEN, len(available)))
        branch_cells = tuple(available[:length])
        for cell in branch_cells:
            occupied.add(cell)
        chosen.append({**cand, "branch_cells": branch_cells})

        if len(chosen) >= target:
            break

    return chosen


def _carve_chain(grid: np.ndarray, chain: list[tuple[int, int]]) -> None:
    for idx, cell in enumerate(chain):
        pr, pc = _cell_to_pixel(cell)
        grid[pr, pc] = BLACK
        if idx == 0:
            continue
        prev_r, prev_c = _cell_to_pixel(chain[idx - 1])
        grid[(pr + prev_r) // 2, (pc + prev_c) // 2] = BLACK


def _build_grid(
    cell_h: int,
    cell_w: int,
    path: list[tuple[int, int]],
    start_side: str,
    end_side: str,
    branches: list[dict[str, object]],
) -> np.ndarray:
    height = 2 * cell_h + 1
    width = 2 * cell_w + 1
    grid = np.full((height, width), GREY, dtype=int)

    _carve_chain(grid, path)
    for branch in branches:
        chain = [branch["cell"], *branch["branch_cells"]]
        _carve_chain(grid, chain)

    used_colors = random.sample(COLOR_POOL, 2 + len(branches))
    entry_color = used_colors[0]
    exit_color = used_colors[1]
    sign_colors = used_colors[2:]

    start_open = _opening_pixel(path[0], start_side, height, width)
    end_open = _opening_pixel(path[-1], end_side, height, width)
    grid[start_open] = entry_color
    grid[end_open] = exit_color

    for branch, color in zip(branches, sign_colors):
        pr, pc = _cell_to_pixel(branch["cell"])
        dr, dc = branch["sign_dir"]
        grid[pr + dr, pc + dc] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(MIN_CELL_H, MAX_CELL_H)
    cell_w = random.randint(MIN_CELL_W, MAX_CELL_W)

    path_result = _generate_main_path(cell_h, cell_w)
    if path_result is None:
        return None
    path, start_side, end_side = path_result

    candidates = _branch_candidates(path, cell_h, cell_w)
    if not candidates:
        return None

    branches = _select_branches(path, candidates, cell_h, cell_w)
    if not branches:
        return None

    grid = _build_grid(cell_h, cell_w, path, start_side, end_side, branches)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guided-maze input")
