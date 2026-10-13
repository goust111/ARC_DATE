from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == BLUE)
    assert coords.shape == (1, 2), "expected exactly one blue seed"
    return int(coords[0, 0]), int(coords[0, 1])


def _wall_segment(grid: np.ndarray, r: int, c: int) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    left = c - 1 >= 0 and int(grid[r, c - 1]) == GREY
    right = c + 1 < w and int(grid[r, c + 1]) == GREY
    up = r - 1 >= 0 and int(grid[r - 1, c]) == GREY
    down = r + 1 < h and int(grid[r + 1, c]) == GREY

    horizontal = left or right
    vertical = up or down
    assert horizontal != vertical, "wall segments must be straight and isolated"

    if horizontal:
        lo = hi = c
        while lo - 1 >= 0 and int(grid[r, lo - 1]) == GREY:
            lo -= 1
        while hi + 1 < w and int(grid[r, hi + 1]) == GREY:
            hi += 1
        cells = [(r, cc) for cc in range(lo, hi + 1)]
        openings = [(r, lo - 1), (r, hi + 1)]
        axis = (0, 1)
        return cells, axis, openings

    lo = hi = r
    while lo - 1 >= 0 and int(grid[lo - 1, c]) == GREY:
        lo -= 1
    while hi + 1 < h and int(grid[hi + 1, c]) == GREY:
        hi += 1
    cells = [(rr, c) for rr in range(lo, hi + 1)]
    openings = [(lo - 1, c), (hi + 1, c)]
    axis = (1, 0)
    return cells, axis, openings


@dataclass
class _SpiralState:
    r: int
    c: int
    dir_idx: int
    leg_len: int
    color: int
    can_split: bool
    kind: str


def _simulate(grid: np.ndarray, *, collect_metrics: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    seed = _find_seed(g)

    metrics = {
        "parent_first_wall_arm": None,
        "child_count": 0,
        "red_hits": 0,
        "blue_new": 0,
        "green_new": 0,
        "parent_wall_hits": 0,
        "child_wall_hits": 0,
        "painted_bg_cells": [],
    }
    spawn_keys: set[tuple[int, int, int, int, int]] = set()
    queue = [_SpiralState(seed[0], seed[1], 0, 2, BLUE, True, "parent")]
    q_index = 0

    while q_index < len(queue):
        state = queue[q_index]
        q_index += 1

        r, c = state.r, state.c
        dir_idx = state.dir_idx
        leg_len = state.leg_len
        pair_phase = 0

        if _in_bounds((h, w), r, c) and int(out[r, c]) == BG:
            out[r, c] = state.color
            if state.color == BLUE:
                metrics["blue_new"] += 1
            else:
                metrics["green_new"] += 1
            metrics["painted_bg_cells"].append((r, c, state.color, state.kind))

        arm_index = 0
        terminated = False
        while not terminated:
            dr, dc = DIRS[dir_idx]
            arm_painted = False

            for _ in range(leg_len):
                nr, nc = r + dr, c + dc
                if not _in_bounds((h, w), nr, nc):
                    r, c = nr, nc
                    continue

                cell = int(out[nr, nc])
                if cell == RED:
                    metrics["red_hits"] += 1
                    terminated = True
                    break

                if cell == GREY:
                    segment_cells, _axis, openings = _wall_segment(g, nr, nc)
                    for wr, wc in segment_cells:
                        sr, sc = wr - dr, wc - dc
                        if _in_bounds((h, w), sr, sc) and int(out[sr, sc]) == BG:
                            out[sr, sc] = state.color
                            arm_painted = True
                            if state.color == BLUE:
                                metrics["blue_new"] += 1
                            else:
                                metrics["green_new"] += 1
                            metrics["painted_bg_cells"].append((sr, sc, state.color, state.kind))

                    if state.kind == "parent":
                        metrics["parent_wall_hits"] += 1
                        if metrics["parent_first_wall_arm"] is None:
                            metrics["parent_first_wall_arm"] = arm_index
                    else:
                        metrics["child_wall_hits"] += 1

                    if state.can_split:
                        for orow, ocol in openings:
                            if not _in_bounds((h, w), orow, ocol):
                                continue
                            if int(g[orow, ocol]) != BG:
                                continue
                            key = (orow, ocol, dir_idx, leg_len, GREEN)
                            if key in spawn_keys:
                                continue
                            spawn_keys.add(key)
                            queue.append(_SpiralState(orow, ocol, dir_idx, leg_len, GREEN, False, "child"))
                            metrics["child_count"] += 1

                    break

                if cell == BG:
                    out[nr, nc] = state.color
                    arm_painted = True
                    if state.color == BLUE:
                        metrics["blue_new"] += 1
                    else:
                        metrics["green_new"] += 1
                    metrics["painted_bg_cells"].append((nr, nc, state.color, state.kind))

                r, c = nr, nc

            if terminated:
                break
            if not arm_painted:
                break

            dir_idx = (dir_idx + 1) % 4
            pair_phase ^= 1
            if pair_phase == 0:
                leg_len += 2
            arm_index += 1

            if arm_index > 2 * (h + w + 6):
                break

    if collect_metrics:
        return out, metrics
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a blue spiral that branches into green child spirals when it grazes grey wall segments.

    Input structure:
    - A black background contains exactly one blue seed pixel, several straight grey wall segments, and optional red obstacle pixels.
    - Wall segments are horizontal or vertical, length at least 3, and remain fixed throughout the transformation.
    - The seed is isolated from the walls, so the first spiral starts in open space.

    Rule:
    - Starting from the blue seed, trace a clipped square spiral with direction order up, right, down, left and leg lengths 2, 2, 4, 4, 6, 6, ...
    - In-bounds background cells on the spiral path become blue; out-of-bounds steps are skipped but still advance the spiral, and a red obstacle stops that spiral immediately.
    - If a spiral step would enter a grey wall, leave the wall unchanged, paint the wall's near side in the spiral color, and for the parent spiral only, start green child spirals from the background openings just beyond the wall segment endpoints. Child spirals start with the collision direction and inherit the parent's current leg length.

    Key insight:
    - The grey segments act as branching triggers rather than filled obstacles: the parent spiral turns a wall hit into a side-trace plus new child spirals whose later sizes depend on the inherited leg-length state.
    """
    return _simulate(np.asarray(grid, dtype=int), collect_metrics=False)


def _arm_trace(start: tuple[int, int], dir_idx: int, leg_len: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = DIRS[dir_idx]
    cells = []
    for _ in range(leg_len):
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _segment_cells(orientation: str, fixed: int, start: int, length: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(fixed, cc) for cc in range(start, start + length)]
    return [(rr, fixed) for rr in range(start, start + length)]


def _can_place_wall(grid: np.ndarray, cells: list[tuple[int, int]], seed: tuple[int, int]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds((h, w), r, c):
            return False
        if int(grid[r, c]) != BG:
            return False
        if _manhattan((r, c), seed) <= 1:
            return False
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == GREY and (nr, nc) not in cell_set:
                return False
    return True


def _wall_openings(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) >= 2 and cells[0][0] == cells[1][0]:
        row = cells[0][0]
        cols = sorted(c for _, c in cells)
        return [(row, cols[0] - 1), (row, cols[-1] + 1)]
    col = cells[0][1]
    rows = sorted(r for r, _ in cells)
    return [(rows[0] - 1, col), (rows[-1] + 1, col)]


def _place_wall(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = GREY


def _sample_primary_wall(grid: np.ndarray, seed: tuple[int, int]) -> bool:
    h, w = grid.shape
    arms = [(0, 2), (1, 2), (2, 4)]
    arm_choices = []
    for dir_idx, leg_len in arms:
        trace = _arm_trace(seed, dir_idx, leg_len)
        for step_idx, (r, c) in enumerate(trace, start=1):
            if not _in_bounds((h, w), r, c):
                continue
            if dir_idx == 0 and step_idx == 1:
                continue
            arm_choices.append((dir_idx, leg_len, step_idx, r, c))
    random.shuffle(arm_choices)

    for dir_idx, leg_len, _step_idx, hit_r, hit_c in arm_choices:
        orientation_choices = ["h" if dir_idx in (0, 2) else "v"]
        if random.random() < 0.25:
            orientation_choices.append("v" if orientation_choices[0] == "h" else "h")
        for orientation in orientation_choices:
            length = random.randint(3, 6)
            for offset in random.sample(range(length), k=length):
                if orientation == "h":
                    start_c = hit_c - offset
                    cells = _segment_cells("h", hit_r, start_c, length)
                else:
                    start_r = hit_r - offset
                    cells = _segment_cells("v", hit_c, start_r, length)

                openings = _wall_openings(cells)
                if not _can_place_wall(grid, cells, seed):
                    continue
                if any(not _in_bounds((h, w), r, c) for r, c in openings):
                    continue
                if any(abs(r - seed[0]) < 3 and abs(c - seed[1]) < 3 for r, c in openings):
                    continue
                if any(int(grid[r, c]) != BG for r, c in openings):
                    continue
                _place_wall(grid, cells)
                return True
    return False


def _random_wall_cells(grid: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    orientation = random.choice(["h", "v"])
    length = random.randint(3, 6)
    candidates = []
    if orientation == "h":
        for row in range(1, h - 1):
            for start_c in range(1, w - length):
                cells = _segment_cells("h", row, start_c, length)
                openings = _wall_openings(cells)
                if _can_place_wall(grid, cells, seed) and all(_in_bounds((h, w), r, c) for r, c in openings):
                    candidates.append(cells)
    else:
        for col in range(1, w - 1):
            for start_r in range(1, h - length):
                cells = _segment_cells("v", col, start_r, length)
                openings = _wall_openings(cells)
                if _can_place_wall(grid, cells, seed) and all(_in_bounds((h, w), r, c) for r, c in openings):
                    candidates.append(cells)
    return random.choice(candidates) if candidates else None


def _place_targeted_wall(grid: np.ndarray, seed: tuple[int, int], painted: list[tuple[int, int, int, str]]) -> bool:
    h, w = grid.shape
    candidates = [(r, c, color) for r, c, color, _kind in painted if color in (BLUE, GREEN) and _manhattan((r, c), seed) >= 4]
    random.shuffle(candidates)
    for hit_r, hit_c, color in candidates:
        preferred = "h" if color == GREEN and random.random() < 0.5 else random.choice(["h", "v"])
        for orientation in [preferred, "v" if preferred == "h" else "h"]:
            length = random.randint(3, 6)
            for offset in random.sample(range(length), k=length):
                if orientation == "h":
                    start_c = hit_c - offset
                    cells = _segment_cells("h", hit_r, start_c, length)
                else:
                    start_r = hit_r - offset
                    cells = _segment_cells("v", hit_c, start_r, length)
                openings = _wall_openings(cells)
                if not _can_place_wall(grid, cells, seed):
                    continue
                if any(not _in_bounds((h, w), r, c) for r, c in openings):
                    continue
                if any(int(grid[r, c]) != BG for r, c in openings):
                    continue
                _place_wall(grid, cells)
                return True
    return False


def _path_red_candidates(grid: np.ndarray, painted: list[tuple[int, int, int, str]], seed: tuple[int, int]) -> list[tuple[int, int]]:
    preferred = []
    fallback = []
    seen = set()
    for r, c, color, kind in painted:
        if (r, c) == seed or int(grid[r, c]) != BG:
            continue
        if _manhattan((r, c), seed) < 2:
            continue
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if color == GREEN or kind == "child":
            preferred.append((r, c))
        else:
            fallback.append((r, c))
    return preferred + fallback


def _place_reds(grid: np.ndarray, seed: tuple[int, int]) -> bool:
    _, metrics = _simulate(grid, collect_metrics=True)
    candidates = _path_red_candidates(grid, metrics["painted_bg_cells"], seed)
    if not candidates:
        return False

    primary = random.choice(candidates[: max(1, min(8, len(candidates)))])
    grid[primary] = RED

    if random.random() < 0.35:
        extras = [
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if int(grid[r, c]) == BG and _manhattan((r, c), seed) >= 3
        ]
        if extras:
            grid[random.choice(extras)] = RED

    _, after = _simulate(grid, collect_metrics=True)
    return after["red_hits"] >= 1


def _count_wall_segments(grid: np.ndarray) -> int:
    h, w = grid.shape
    seen = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY or (r, c) in seen:
                continue
            cells, _axis, _openings = _wall_segment(grid, r, c)
            for cell in cells:
                seen.add(cell)
            count += 1
    return count


def _accept(grid: np.ndarray) -> bool:
    out, metrics = _simulate(grid, collect_metrics=True)
    if np.array_equal(grid, out):
        return False
    if metrics["parent_first_wall_arm"] is None or metrics["parent_first_wall_arm"] > 2:
        return False
    if metrics["child_count"] < 1:
        return False
    if metrics["green_new"] < 4:
        return False
    if metrics["blue_new"] < 3:
        return False
    if _count_wall_segments(grid) < 1:
        return False
    density = np.count_nonzero(grid) / grid.size
    if density > 0.18:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    seed = (random.randint(max(4, h // 2 - 2), min(h - 5, h // 2 + 2)), random.randint(max(4, w // 2 - 2), min(w - 5, w // 2 + 2)))
    grid[seed] = BLUE

    if not _sample_primary_wall(grid, seed):
        return None

    for _ in range(random.randint(0, 1)):
        cells = _random_wall_cells(grid, seed)
        if cells is not None and random.random() < 0.7:
            _place_wall(grid, cells)

    _, base_metrics = _simulate(grid, collect_metrics=True)
    if base_metrics["parent_first_wall_arm"] is None or base_metrics["child_count"] < 1:
        return None

    if random.random() < 0.55:
        _place_targeted_wall(grid, seed, base_metrics["painted_bg_cells"])

    if random.random() < 0.25:
        cells = _random_wall_cells(grid, seed)
        if cells is not None:
            _place_wall(grid, cells)

    if random.random() < 0.7 and not _place_reds(grid, seed):
        return None

    if not _accept(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
