from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREEN = 3
YELLOW = 4
GREY = 5
SEED_COLORS = (1, 2, 4, 6, 7, 8, 9)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HEIGHT_RANGE = (13, 18)
CONTROL_COUNT_CHOICES = (3, 4)
CONTROL_GAP_RANGE = (4, 6)
LEFT_MARGIN_RANGE = (1, 3)
RIGHT_MARGIN_RANGE = (2, 4)
WIDTH_SLACK_RANGE = (0, 4)
EXTRA_OBSTACLE_RANGE = (10, 15)
PATH_SHIFT_MAX = 4
PATH_EVENT_MAX = 3
PATH_RUN_MAX = 2
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _passable_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _simulate_paths(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts = sorted(int(c) for c in np.flatnonzero(g[h - 1] == GREY))
    out = g.copy()
    infos: list[dict] = []

    for sc in starts:
        r = h - 1
        c = sc
        cells = [(r, c)]
        blockers: list[tuple[int, int]] = []
        row_profile = {r: c}
        turns = 0
        reached_top = False

        while r > 0:
            above = int(g[r - 1, c])
            if above == BG:
                r -= 1
                out[r, c] = GREY
                cells.append((r, c))
                row_profile[r] = c
                if r == 0:
                    reached_top = True
            elif above == GREEN:
                blockers.append((r - 1, c))
                if c + 1 >= w or int(g[r, c + 1]) != BG:
                    break
                c += 1
                turns += 1
                out[r, c] = GREY
                cells.append((r, c))
                row_profile[r] = c
            else:
                break

        infos.append(
            {
                "start_col": sc,
                "cells": cells,
                "cell_set": set(cells),
                "blockers": blockers,
                "turns": turns,
                "reached_top": reached_top,
                "row_profile": row_profile,
            }
        )

    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow grey control paths upward around green blockers, then fill each seed-bearing pocket with the seed color.

    Input structure:
    - The background is black.
    - Grey control pixels sit on the bottom row, green pixels are blockers, and a few
      colored singleton seeds lie between neighboring controls on the bottom edge.
    - The rest of the grid is black.

    Rule:
    - From each grey control pixel, repeatedly try to move upward. If the cell above
      is black, extend the grey path upward; if the cell above is green, step one
      cell right on the current row and continue from there, but stop if that right
      cell is unavailable or non-black.
    - After all paths are grown, treat grey path cells and green blockers as walls.
    - Find every 4-connected non-wall region. If a region contains exactly one seed
      color from the input, flood-fill all black cells in that region with that seed
      color. Seeds themselves are overwritten by the same fill color.

    Key insight:
    - The seed colors do not decide their target areas by simple column gaps: the
      grey walls must be simulated first, because obstacle-driven right shifts change
      which cells are reachable from each seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out, _infos = _simulate_paths(g)
    h, w = out.shape

    seed_positions = {
        (int(r), int(c)): int(g[int(r), int(c)])
        for r, c in np.argwhere(~np.isin(g, [BG, GREEN, GREY]))
    }
    passable = (out != GREY) & (out != GREEN)

    for comp in _passable_components(passable):
        seeds = [seed_positions[pos] for pos in comp if pos in seed_positions]
        if len(seeds) != 1:
            continue
        fill_color = int(seeds[0])
        for r, c in comp:
            if int(out[r, c]) == BG:
                out[r, c] = fill_color

    return out


def _choose_control_columns(count: int) -> tuple[int, list[int]]:
    gaps = [random.randint(*CONTROL_GAP_RANGE) for _ in range(count - 1)]
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    right_margin = random.randint(*RIGHT_MARGIN_RANGE)
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    min_width = cols[-1] + right_margin + 1
    width = min_width + random.randint(*WIDTH_SLACK_RANGE)
    return width, cols


def _split_shift(total_shift: int) -> list[int]:
    parts = [1] * random.randint(1, min(PATH_EVENT_MAX, total_shift))
    remaining = total_shift - len(parts)
    while remaining > 0:
        candidates = [i for i, value in enumerate(parts) if value < PATH_RUN_MAX]
        if not candidates:
            parts.append(1)
            remaining -= 1
            continue
        idx = random.choice(candidates)
        parts[idx] += 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def _build_path_plan(height: int, start_col: int, total_shift: int) -> dict:
    runs = _split_shift(total_shift)
    event_rows = sorted(random.sample(range(2, height - 1), len(runs)), reverse=True)
    row_runs = {row: run for row, run in zip(event_rows, runs)}

    r = height - 1
    c = start_col
    cells = [(r, c)]
    cell_set = {(r, c)}
    blockers: list[tuple[int, int]] = []
    blocker_set: set[tuple[int, int]] = set()
    clear_above: set[tuple[int, int]] = set()
    row_profile = {r: c}

    for row in range(height - 1, 0, -1):
        run = row_runs.get(row, 0)
        for _ in range(run):
            blocker = (row - 1, c)
            blockers.append(blocker)
            blocker_set.add(blocker)
            c += 1
            cells.append((row, c))
            cell_set.add((row, c))
        row_profile[row] = c
        clear_above.add((row - 1, c))
        cells.append((row - 1, c))
        cell_set.add((row - 1, c))
        row_profile[row - 1] = c

    return {
        "start_col": start_col,
        "cells": cells,
        "cell_set": cell_set,
        "blockers": blockers,
        "blocker_set": blocker_set,
        "clear_above": clear_above,
        "turns": total_shift,
        "row_profile": row_profile,
    }


def _seed_columns(control_cols: list[int]) -> list[int]:
    seeds: list[int] = []
    for left, right in zip(control_cols, control_cols[1:]):
        candidates = list(range(left + 2, right - 1))
        seeds.append(random.choice(candidates))
    return seeds


def _candidate_extra_obstacles(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    forbidden_clear: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(w):
            pos = (r, c)
            if pos in reserved or pos in forbidden_clear:
                continue
            candidates.append(pos)
    random.shuffle(candidates)
    return candidates


def _shape_varies(path_infos: list[dict]) -> bool:
    for left, right in zip(path_infos, path_infos[1:]):
        diffs = {
            int(right["row_profile"][row]) - int(left["row_profile"][row])
            for row in left["row_profile"]
        }
        if len(diffs) > 1:
            return True
    return False


def _validate_example(grid: np.ndarray) -> bool:
    h, w = grid.shape
    control_cols = sorted(int(c) for c in np.flatnonzero(grid[h - 1] == GREY))
    if len(control_cols) not in CONTROL_COUNT_CHOICES:
        return False
    if np.count_nonzero(grid == GREEN) < 10 or np.count_nonzero(grid == GREEN) > 15:
        return False
    if np.any(grid[0] == GREEN) or np.any(grid[h - 1] == GREEN):
        return False

    seeds = [(int(r), int(c), int(grid[int(r), int(c)])) for r, c in np.argwhere(~np.isin(grid, [BG, GREEN, GREY]))]
    if len(seeds) != len(control_cols) - 1:
        return False
    if len({color for _, _, color in seeds}) != len(seeds):
        return False
    if any(r != h - 1 for r, _, _ in seeds):
        return False

    for (left, right), (_, c, _) in zip(zip(control_cols, control_cols[1:]), sorted(seeds, key=lambda item: item[1])):
        if not (left + 1 < c < right - 1):
            return False

    grown, infos = _simulate_paths(grid)
    if len(infos) != len(control_cols):
        return False
    if not all(info["reached_top"] and info["turns"] >= 1 for info in infos):
        return False

    used: set[tuple[int, int]] = set()
    for info in infos:
        if used & info["cell_set"]:
            return False
        used |= info["cell_set"]

    if not _shape_varies(infos):
        return False

    seed_map = {(r, c): color for r, c, color in seeds}
    passable = (grown != GREY) & (grown != GREEN)
    comps = _passable_components(passable)
    seed_components = 0
    total_fillable = 0
    for comp in comps:
        seed_hits = [seed_map[pos] for pos in comp if pos in seed_map]
        if len(seed_hits) == 1:
            seed_components += 1
            black_count = sum(int(grid[r, c]) == BG for r, c in comp)
            if black_count < 3:
                return False
            total_fillable += black_count
        elif len(seed_hits) > 1:
            return False
    if seed_components != len(seeds):
        return False
    if total_fillable < 10:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[grown == GREY] == GREY):
        return False
    if not np.all(out[grid == GREEN] == GREEN):
        return False
    if len(set(np.unique(out)) - {BG, GREEN, GREY}) != len(seeds):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    count = random.choice(CONTROL_COUNT_CHOICES)
    width, control_cols = _choose_control_columns(count)
    if width > 30:
        return None

    max_shifts: list[int] = []
    for i, start_col in enumerate(control_cols):
        if i + 1 < len(control_cols):
            limit = control_cols[i + 1] - start_col - 2
        else:
            limit = width - start_col - 2
        limit = min(limit, PATH_SHIFT_MAX)
        if limit < 1:
            return None
        max_shifts.append(limit)

    path_infos: list[dict] = []
    reserved: set[tuple[int, int]] = set()
    all_blockers: set[tuple[int, int]] = set()
    all_clear_above: set[tuple[int, int]] = set()
    for start_col, limit in zip(control_cols, max_shifts):
        total_shift = random.randint(1, limit)
        info = _build_path_plan(height, start_col, total_shift)
        if info["cell_set"] & reserved:
            return None
        if info["blocker_set"] & reserved:
            return None
        path_infos.append(info)
        reserved |= info["cell_set"]
        all_blockers |= info["blocker_set"]
        all_clear_above |= info["clear_above"]

    if not _shape_varies(path_infos):
        return None

    grid = np.zeros((height, width), dtype=int)
    for col in control_cols:
        grid[height - 1, col] = GREY
    for blocker in all_blockers:
        grid[blocker] = GREEN

    seed_cols = _seed_columns(control_cols)
    seed_colors = random.sample(SEED_COLORS, len(seed_cols))
    for col, color in zip(seed_cols, seed_colors):
        grid[height - 1, col] = color
        reserved.add((height - 1, col))

    reserved |= all_blockers
    extra_target = random.randint(*EXTRA_OBSTACLE_RANGE) - len(all_blockers)
    if extra_target < 0:
        return None

    candidates = _candidate_extra_obstacles(grid.shape, reserved, all_clear_above)
    extras_added = 0
    for r, c in candidates:
        if extras_added >= extra_target:
            break
        grid[r, c] = GREEN
        extras_added += 1
        reserved.add((r, c))
    if extras_added != extra_target:
        return None

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-deflected fenced-fill example")
