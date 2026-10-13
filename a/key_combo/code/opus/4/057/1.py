from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_057_1",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
in_bounds = _ARC_UTILS.in_bounds


BACKGROUND = Color.BLACK
FILL = Color.BLUE
PATH = Color.RED
BLOCKER = Color.GREY

HEIGHT_RANGE = (14, 18)
WIDTH_RANGE = (18, 25)
CONTROL_COUNT_CHOICES = (3, 4)
OBSTACLE_COUNT_RANGE = (4, 6)
MAX_ATTEMPTS = 400

PATTERNS_BY_CONTROL_COUNT = {
    3: (
        (0, 1, 1),
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
        (1, 1, 0),
    ),
    4: (
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 2, 0),
        (0, 2, 1, 0),
        (1, 0, 1, 1),
    ),
}


def _control_columns(rng: random.Random, width: int, count: int) -> list[int] | None:
    gaps = [rng.randint(4, 6) for _ in range(count - 1)]
    span = sum(gaps)
    min_left = 1
    max_left = width - 3 - span
    if max_left < min_left:
        return None
    left = rng.randint(min_left, max_left)
    cols = [left]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    if cols[-1] > width - 3:
        return None
    return cols


def _sample_trigger_rows(
    rng: random.Random,
    height: int,
    deflection_count: int,
    *,
    stop_here: bool,
) -> list[int] | None:
    if deflection_count == 0:
        return []

    top_limit = max(2, height // 2)
    if stop_here:
        top_limit = min(top_limit, max(2, height // 2 - 1))
    candidates = list(range(2, top_limit + 1))
    if len(candidates) < deflection_count:
        return None

    picked = []
    for row in sorted(rng.sample(candidates, deflection_count), reverse=True):
        if picked and picked[-1] - row < 2:
            return None
        picked.append(row)
    return picked


def _grow_paths(
    grid: np.ndarray,
    *,
    ignore_blockers: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape
    controls = [int(c) for c in np.flatnonzero(out[-1] == PATH)]
    infos: list[dict] = []

    for start_col in controls:
        r = height - 1
        c = start_col
        cells = [(r, c)]
        turns = 0
        hits = 0
        stopped = False
        stop_reason = "top"

        while r > 0:
            above = int(out[r - 1, c])
            if ignore_blockers and above == BLOCKER:
                above = BACKGROUND

            if above == BACKGROUND:
                r -= 1
                out[r, c] = PATH
                cells.append((r, c))
                continue

            if above == BLOCKER and not ignore_blockers:
                hits += 1
                nc = c + 1
                if not in_bounds(out.shape, r, nc) or int(out[r, nc]) != BACKGROUND:
                    stopped = True
                    stop_reason = "blocked_right"
                    break
                c = nc
                turns += 1
                out[r, c] = PATH
                cells.append((r, c))
                continue

            stopped = True
            stop_reason = "non_background_above"
            break

        infos.append(
            {
                "start_col": start_col,
                "cells": cells,
                "turns": turns,
                "hits": hits,
                "stopped": stopped,
                "stop_reason": stop_reason,
                "top_row": min(r0 for r0, _ in cells),
            }
        )

    return out, infos


def _fill_between_paths(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r in range(out.shape[0]):
        cols = [int(c) for c in np.flatnonzero(out[r] == PATH)]
        for left, right in zip(cols, cols[1:]):
            if right - left <= 1:
                continue
            span = out[r, left + 1 : right]
            span[span == BACKGROUND] = FILL
    return out


def _solve(grid: np.ndarray, *, ignore_blockers: bool = False) -> tuple[np.ndarray, list[dict]]:
    grown, infos = _grow_paths(grid, ignore_blockers=ignore_blockers)
    return _fill_between_paths(grown), infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow bottom-origin red paths upward and use them as scanline boundaries for blue filling.

    Input structure:
    - A black background grid contains 4 to 6 isolated grey blocker pixels in the upper half.
    - The bottom row contains 3 or 4 separated red control pixels.
    - No other foreground colors are present in the input.

    Rule:
    - Starting from each bottom-row red control pixel, grow a red path upward one step at a time.
    - If the cell directly above the current path tip is black, extend upward into it; if that above cell is grey, move one cell right on the current row instead and continue from there; if the right cell is unavailable or the above cell is another non-background color, that path stops.
    - After all paths are grown, scan each row from left to right and fill every black span strictly between consecutive red pixels with blue, while keeping all red path pixels and grey blockers unchanged.

    Key insight:
    - The red paths are not the final target by themselves: blocker-driven right deflections reshape the red row endpoints, and those row endpoints are exactly what the scanline fill uses to create the blue regions.
    """
    output, _infos = _solve(np.asarray(grid, dtype=int))
    return output


def _path_cells_from_infos(infos: list[dict]) -> set[tuple[int, int]]:
    return {tuple(cell) for info in infos for cell in info["cells"]}


def _obstacle_isolated(pos: tuple[int, int], obstacles: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in obstacles:
        if abs(rr - r) + abs(cc - c) == 1:
            return False
    return True


def _blue_profile(grid: np.ndarray) -> list[int]:
    return [int(np.count_nonzero(grid[r] == FILL)) for r in range(grid.shape[0])]


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    control_count = rng.choice(CONTROL_COUNT_CHOICES)
    controls = _control_columns(rng, width, control_count)
    if controls is None:
        return None

    deflection_pattern = list(rng.choice(PATTERNS_BY_CONTROL_COUNT[control_count]))
    if max(deflection_pattern) == 0 or min(deflection_pattern) != 0:
        return None

    stop_candidates = [idx for idx, count in enumerate(deflection_pattern) if count > 0]
    stop_index = rng.choice(stop_candidates) if stop_candidates and rng.random() < 0.7 else None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[-1, controls] = PATH
    obstacles: set[tuple[int, int]] = set()

    for path_index, (start_col, deflection_count) in enumerate(zip(controls, deflection_pattern)):
        trigger_rows = _sample_trigger_rows(
            rng,
            height,
            deflection_count,
            stop_here=path_index == stop_index,
        )
        if trigger_rows is None:
            return None

        current_col = start_col
        for deflection_id, trigger_row in enumerate(trigger_rows):
            blocker = (trigger_row, current_col)
            if blocker in obstacles or not _obstacle_isolated(blocker, obstacles):
                return None
            obstacles.add(blocker)
            grid[blocker] = BLOCKER

            is_stop_turn = path_index == stop_index and deflection_id == len(trigger_rows) - 1
            if is_stop_turn:
                right_blocker = (trigger_row + 1, current_col + 1)
                if not in_bounds(grid.shape, *right_blocker):
                    return None
                if right_blocker in obstacles or not _obstacle_isolated(right_blocker, obstacles):
                    return None
                if right_blocker[0] > height // 2:
                    return None
                obstacles.add(right_blocker)
                grid[right_blocker] = BLOCKER
                break

            current_col += 1

    grown_only, infos = _grow_paths(grid)
    path_cells = _path_cells_from_infos(infos)
    sensitive_above = {(r - 1, c) for r, c in path_cells if r > 0}

    target_obstacle_count = rng.randint(*OBSTACLE_COUNT_RANGE)
    while len(obstacles) < target_obstacle_count:
        candidate = (rng.randint(1, height // 2), rng.randint(0, width - 1))
        if candidate in obstacles or not _obstacle_isolated(candidate, obstacles):
            continue
        if candidate in path_cells or candidate in sensitive_above:
            continue
        if candidate[0] == height - 1:
            continue
        obstacles.add(candidate)
        grid[candidate] = BLOCKER

    if not _semantic_checks(grid):
        return None
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    height, width = grid.shape
    if height > 30 or width > 30:
        return False

    controls = [int(c) for c in np.flatnonzero(grid[-1] == PATH)]
    if len(controls) not in CONTROL_COUNT_CHOICES:
        return False
    if any(b - a < 3 for a, b in zip(controls, controls[1:])):
        return False

    obstacle_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLOCKER)]
    if not (OBSTACLE_COUNT_RANGE[0] <= len(obstacle_coords) <= OBSTACLE_COUNT_RANGE[1]):
        return False
    for r, _c in obstacle_coords:
        if r > height // 2:
            return False
    for i, (r1, c1) in enumerate(obstacle_coords):
        for r2, c2 in obstacle_coords[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                return False

    grown, infos = _grow_paths(grid)
    solved = _fill_between_paths(grown)
    if any(info["top_row"] > height - 6 for info in infos):
        return False
    if not any(info["turns"] == 0 for info in infos):
        return False
    if not any(info["turns"] > 0 for info in infos):
        return False
    if not any(info["hits"] > 0 for info in infos):
        return False
    if np.count_nonzero(solved == FILL) == 0:
        return False

    profile = [count for count in _blue_profile(solved) if count > 0]
    if len(set(profile)) < 2:
        return False

    naive_without_blockers, _ = _solve(grid, ignore_blockers=True)
    if np.array_equal(naive_without_blockers, solved):
        return False
    if np.array_equal(grown, solved):
        return False

    blocker_sensitive = False
    for r, c in obstacle_coords:
        modified = np.array(grid, copy=True)
        modified[r, c] = BACKGROUND
        if not np.array_equal(main(modified), solved):
            blocker_sensitive = True
            break
    if not blocker_sensitive:
        return False

    return True


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-deflection fill input")
