from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DIR_TO_VEC = {
    RED: (-1, 0),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (0, 1),
}
ORDER_TO_INDEX = {
    ORANGE: 0,
    MAGENTA: 1,
    MAROON: 2,
    AZURE: 3,
}
ORDER_COLORS = [ORANGE, MAGENTA, MAROON, AZURE]

GRID_H_MIN = 10
GRID_H_MAX = 18
GRID_W_MIN = 10
GRID_W_MAX = 18
MOVE_COUNT_CHOICES = (2, 3, 4)
MAX_SIGN_DISTANCE = 5
MAX_ATTEMPTS = 300
START_SEARCH_LIMIT = 96
PLACEMENT_BRANCH_LIMIT = 160


def _mode_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _nearest_edge_distance(r: int, c: int, h: int, w: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _steps_available_from(pos: tuple[int, int], h: int, w: int, dir_color: int) -> int:
    r, c = pos
    if dir_color == RED:
        return r
    if dir_color == BLUE:
        return h - 1 - r
    if dir_color == GREEN:
        return c
    return w - 1 - c


def _buffer_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _find_signposts(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    posts: list[dict] = []
    for r0, c0 in np.argwhere(g == GREY):
        r = int(r0)
        c = int(c0)
        neighbors = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
                neighbors.append((nr, nc))
        if len(neighbors) != 2:
            continue
        (r1, c1), (r2, c2) = neighbors
        horizontal = r1 == r2 == r and {c1, c2} == {c - 1, c + 1}
        vertical = c1 == c2 == c and {r1, r2} == {r - 1, r + 1}
        if not (horizontal or vertical):
            continue

        v1 = int(g[r1, c1])
        v2 = int(g[r2, c2])
        if v1 == v2:
            continue
        if v1 in DIR_TO_VEC and v2 in ORDER_TO_INDEX:
            dir_end = (r1, c1)
            dir_color = v1
            order_color = v2
        elif v2 in DIR_TO_VEC and v1 in ORDER_TO_INDEX:
            dir_end = (r2, c2)
            dir_color = v2
            order_color = v1
        else:
            continue

        posts.append(
            {
                "center": (r, c),
                "dir_end": dir_end,
                "dir_color": dir_color,
                "order": ORDER_TO_INDEX[order_color],
                "pixels": [
                    (r1, c1, v1),
                    (r, c, GREY),
                    (r2, c2, v2),
                ],
            }
        )

    posts.sort(key=lambda item: (item["order"], item["center"][0], item["center"][1]))
    unique: list[dict] = []
    seen_orders: set[int] = set()
    seen_centers: set[tuple[int, int]] = set()
    for post in posts:
        if post["order"] in seen_orders or post["center"] in seen_centers:
            continue
        unique.append(post)
        seen_orders.add(post["order"])
        seen_centers.add(post["center"])
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)
    out = g.copy()

    signposts = _find_signposts(g, bg)
    signpost_cells = {(r, c) for post in signposts for r, c, _ in post["pixels"]}

    start_candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(g == RED)
        if (int(r), int(c)) not in signpost_cells
    ]
    if not start_candidates:
        return out
    start = min(start_candidates)

    pos = start
    for post in signposts:
        steps = _nearest_edge_distance(post["dir_end"][0], post["dir_end"][1], h, w)
        dr, dc = DIR_TO_VEC[post["dir_color"]]
        for _ in range(steps):
            nr = pos[0] + dr
            nc = pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            pos = (nr, nc)
            out[nr, nc] = post["dir_color"]

    for post in signposts:
        for r, c, value in post["pixels"]:
            out[r, c] = value
    return out


def _signpost_candidates(h: int, w: int) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for r in range(h):
        for c in range(w):
            for dr, dc in ((0, 1), (1, 0)):
                r1, c1 = r - dr, c - dc
                r2, c2 = r + dr, c + dc
                if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
                    continue
                cells = [(r1, c1), (r, c), (r2, c2)]
                for dir_slot in (0, 2):
                    dir_end = cells[dir_slot]
                    steps = _nearest_edge_distance(dir_end[0], dir_end[1], h, w)
                    if steps < 1:
                        continue
                    groups.setdefault(steps, []).append(
                        {
                            "cells": cells,
                            "dir_slot": dir_slot,
                        }
                    )
    return groups


def _walk_dfs(
    rng: random.Random,
    h: int,
    w: int,
    pos: tuple[int, int],
    target_len: int,
    step_values: list[int],
    occupied: set[tuple[int, int]],
    moves: list[dict],
    prev_dir: int | None,
):
    if len(moves) == target_len:
        dir_colors = [move["dir_color"] for move in moves]
        if target_len > 1 and len(set(dir_colors)) == 1:
            return None
        if max(move["steps"] for move in moves) < 2:
            return None
        return [dict(move) for move in moves]

    dir_items = list(DIR_TO_VEC.items())
    rng.shuffle(dir_items)
    shuffled_steps = step_values[:]
    rng.shuffle(shuffled_steps)

    candidates = []
    for dir_color, (dr, dc) in dir_items:
        max_steps = _steps_available_from(pos, h, w, dir_color)
        for steps in shuffled_steps:
            if steps > max_steps:
                continue
            r, c = pos
            segment = []
            ok = True
            for _ in range(steps):
                r += dr
                c += dc
                cell = (r, c)
                if cell in occupied:
                    ok = False
                    break
                segment.append(cell)
            if ok:
                candidates.append((dir_color, steps, segment))

    rng.shuffle(candidates)
    candidates.sort(key=lambda item: (prev_dir is not None and item[0] == prev_dir, item[1] == 1))

    for dir_color, steps, segment in candidates:
        moves.append({"dir_color": dir_color, "steps": steps, "segment": segment})
        for cell in segment:
            occupied.add(cell)
        result = _walk_dfs(
            rng,
            h,
            w,
            segment[-1],
            target_len,
            step_values,
            occupied,
            moves,
            dir_color,
        )
        if result is not None:
            return result
        for cell in segment:
            occupied.remove(cell)
        moves.pop()
    return None


def _build_walk(rng: random.Random, h: int, w: int, move_count: int, step_values: list[int]):
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(starts)
    for start in starts[:START_SEARCH_LIMIT]:
        occupied = {start}
        result = _walk_dfs(rng, h, w, start, move_count, step_values, occupied, [], None)
        if result is not None:
            return start, result
    return None


def _trace_path(start: tuple[int, int], moves: list[dict]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    pos = start
    for move in moves:
        dr, dc = DIR_TO_VEC[move["dir_color"]]
        for _ in range(move["steps"]):
            pos = (pos[0] + dr, pos[1] + dc)
            cells.append(pos)
    return cells


def _place_signposts(
    rng: random.Random,
    h: int,
    w: int,
    start: tuple[int, int],
    path_cells: list[tuple[int, int]],
    moves: list[dict],
    candidates_by_step: dict[int, list[dict]],
):
    path_blocked = set(path_cells)
    start_reserved = _buffer_cells([start], h, w, pad=1)

    def dfs(index: int, reserved: set[tuple[int, int]], placed: list[dict]):
        if index == len(moves):
            return list(placed)

        pool = list(candidates_by_step[moves[index]["steps"]])
        rng.shuffle(pool)
        tried = 0
        for cand in pool:
            cells = cand["cells"]
            if any(cell in path_blocked for cell in cells):
                continue
            if any(cell in reserved for cell in cells):
                continue
            placed.append(cand)
            next_reserved = reserved | _buffer_cells(cells, h, w, pad=1)
            result = dfs(index + 1, next_reserved, placed)
            if result is not None:
                return result
            placed.pop()
            tried += 1
            if tried >= PLACEMENT_BRANCH_LIMIT:
                break
        return None

    return dfs(0, start_reserved, [])


def _render_input(h: int, w: int, start: tuple[int, int], moves: list[dict], placements: list[dict]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((h, w), BG, dtype=int)
    grid[start] = RED
    signposts: list[dict] = []
    for order, (move, placement) in enumerate(zip(moves, placements)):
        order_color = ORDER_COLORS[order]
        pixels = []
        for idx, (r, c) in enumerate(placement["cells"]):
            if idx == 1:
                value = GREY
            elif idx == placement["dir_slot"]:
                value = move["dir_color"]
            else:
                value = order_color
            grid[r, c] = value
            pixels.append((r, c, value))
        signposts.append({"pixels": pixels})
    return grid, signposts


def _expected_output(grid: np.ndarray, start: tuple[int, int], moves: list[dict], signposts: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    pos = start
    for move in moves:
        dr, dc = DIR_TO_VEC[move["dir_color"]]
        for _ in range(move["steps"]):
            pos = (pos[0] + dr, pos[1] + dc)
            out[pos] = move["dir_color"]
    for signpost in signposts:
        for r, c, value in signpost["pixels"]:
            out[r, c] = value
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_MIN, GRID_H_MAX)
    w = rng.randint(GRID_W_MIN, GRID_W_MAX)

    candidates_by_step = _signpost_candidates(h, w)
    max_step = min(MAX_SIGN_DISTANCE, (min(h, w) - 1) // 2)
    step_values = [step for step in sorted(candidates_by_step) if 1 <= step <= max_step]
    if len(step_values) < 2:
        return None

    move_count = rng.choice(MOVE_COUNT_CHOICES)
    walk = _build_walk(rng, h, w, move_count, step_values)
    if walk is None:
        return None
    start, moves = walk

    path_cells = _trace_path(start, moves)
    placements = _place_signposts(rng, h, w, start, path_cells, moves, candidates_by_step)
    if placements is None:
        return None

    grid, signposts = _render_input(h, w, start, moves, placements)
    if len(_find_signposts(grid, BG)) != move_count:
        return None

    expected = _expected_output(grid, start, moves, signposts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
