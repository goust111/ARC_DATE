from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SEED_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))

GEN_ATTEMPTS = 600
GEN_HEIGHT_RANGE = (13, 19)
GEN_WIDTH_RANGE = (13, 19)
GEN_SEED_COUNT_OPTIONS = (2, 3)
GEN_BLOCKER_COUNT_RANGE = (3, 6)  # upper bound is exclusive for rng.integers
MIN_SEED_MANHATTAN = 6
MIN_SAFE_MARGIN = 3
MIN_FIRST_BLOCK_STEPS = 12
MIN_SINGLE_REGION_AREA = 4
MIN_MULTI_REGION_AREA = 4
MIN_TOTAL_FILLED_AREA = 16

GEN_TEMPLATES = (
    {
        "seeds": ((4, 7), (10, 4), (10, 10)),
        "blocker_sets": (
            ((7, 5), (9, 14), (13, 4), (14, 12)),
            ((3, 0), (10, 14), (13, 9), (14, 8)),
            ((4, 5), (6, 0), (6, 12), (14, 2)),
        ),
        "extras": ((0, 10), (1, 2), (2, 13), (11, 1), (14, 6)),
    },
    {
        "seeds": ((4, 4), (4, 10), (10, 7)),
        "blocker_sets": (
            ((0, 1), (0, 6), (6, 8), (8, 4)),
            ((0, 7), (5, 0), (6, 2), (7, 14)),
            ((0, 7), (4, 14), (10, 9), (14, 14)),
        ),
        "extras": ((2, 12), (8, 0), (10, 1), (12, 12), (14, 3)),
    },
    {
        "seeds": ((4, 6), (10, 4), (9, 10)),
        "blocker_sets": (
            ((0, 3), (10, 1), (11, 0), (14, 6)),
            ((6, 0), (10, 14), (11, 11), (14, 0)),
            ((3, 9), (7, 0), (11, 11), (14, 3)),
        ),
        "extras": ((0, 12), (1, 1), (5, 13), (8, 14), (13, 8)),
    },
    {
        "seeds": ((4, 5), (10, 5)),
        "blocker_sets": (
            ((4, 7), (7, 0), (11, 0), (13, 10)),
            ((2, 4), (2, 14), (4, 1), (9, 0)),
            ((4, 7), (8, 13), (11, 0), (13, 10)),
        ),
        "extras": ((0, 11), (1, 1), (6, 12), (12, 14), (14, 2)),
    },
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _turn(state: dict[str, int]) -> None:
    state["dir_idx"] = (state["dir_idx"] + 1) % 4
    state["leg_progress"] = 0
    if state["dir_idx"] in (0, 2):
        state["leg_len"] += 1


def _blocked(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    r: int,
    c: int,
) -> bool:
    return (not _in_bounds(shape, r, c)) or (r, c) in blockers or (r, c) in occupied


def _seed_info(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    counts = Counter(int(v) for v in g.ravel())
    blockers = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        r, c = int(r), int(c)
        color = int(g[r, c])
        assert counts[color] == 1, "every non-grey foreground color must be a single-pixel seed"
        seeds.append((r, c, color))
    seeds.sort()
    assert 2 <= len(seeds) <= 3, "expected 2 or 3 seeds"
    assert len({color for _r, _c, color in seeds}) == len(seeds), "seed colors must be distinct"
    return seeds, blockers


def _simulate_spirals(
    seeds: list[tuple[int, int, int]],
    blockers: set[tuple[int, int]],
    shape: tuple[int, int],
    *,
    shared_walls: bool,
) -> dict[str, object]:
    states: list[dict[str, object]] = []
    owner_sets: list[set[tuple[int, int]]] = []
    all_occupied: dict[tuple[int, int], int] = {}
    for r, c, color in seeds:
        state = {
            "r": int(r),
            "c": int(c),
            "color": int(color),
            "dir_idx": 0,
            "leg_len": 1,
            "leg_progress": 0,
            "active": True,
            "new_pixels": 0,
            "first_block_after": None,
            "terminated_after": None,
        }
        states.append(state)
        own = {(int(r), int(c))}
        owner_sets.append(own)
        all_occupied[(int(r), int(c))] = int(color)

    had_collision = False
    safety_limit = shape[0] * shape[1] * 12
    for _tick in range(safety_limit):
        if not any(bool(state["active"]) for state in states):
            break

        proposals: list[tuple[int, int, int]] = []
        current_occupied = set(all_occupied.keys()) if shared_walls else set()
        for idx, own in enumerate(owner_sets):
            current_occupied |= own

        for idx, state in enumerate(states):
            if not bool(state["active"]):
                continue
            occupied = set(all_occupied.keys()) if shared_walls else owner_sets[idx]
            r, c = int(state["r"]), int(state["c"])
            dr, dc = DIRS4[int(state["dir_idx"])]
            nr, nc = r + dr, c + dc
            if _blocked(shape, occupied, blockers, nr, nc):
                if state["first_block_after"] is None:
                    state["first_block_after"] = int(state["new_pixels"])
                _turn(state)
                dr2, dc2 = DIRS4[int(state["dir_idx"])]
                tr, tc = r + dr2, c + dc2
                if _blocked(shape, occupied, blockers, tr, tc):
                    state["active"] = False
                    state["terminated_after"] = int(state["new_pixels"])
                continue
            proposals.append((idx, nr, nc))

        by_target: dict[tuple[int, int], list[int]] = {}
        for idx, nr, nc in proposals:
            by_target.setdefault((nr, nc), []).append(idx)

        for (nr, nc), claimers in by_target.items():
            if len(claimers) > 1:
                had_collision = True
                for idx in claimers:
                    state = states[idx]
                    if state["first_block_after"] is None:
                        state["first_block_after"] = int(state["new_pixels"])
                    state["active"] = False
                    state["terminated_after"] = int(state["new_pixels"])
                continue
            idx = claimers[0]
            state = states[idx]
            color = int(state["color"])
            owner_sets[idx].add((nr, nc))
            if shared_walls:
                all_occupied[(nr, nc)] = color
            else:
                all_occupied[(nr, nc)] = color
            state["r"], state["c"] = int(nr), int(nc)
            state["new_pixels"] = int(state["new_pixels"]) + 1
            state["leg_progress"] = int(state["leg_progress"]) + 1
            if int(state["leg_progress"]) == int(state["leg_len"]):
                _turn(state)
    else:
        raise AssertionError("spiral simulation did not terminate")

    path_grid = np.zeros(shape, dtype=int)
    for idx, own in enumerate(owner_sets):
        color = int(states[idx]["color"])
        for r, c in own:
            path_grid[r, c] = color
    for r, c in blockers:
        path_grid[r, c] = GREY

    return {
        "path_grid": path_grid,
        "states": states,
        "owner_sets": owner_sets,
        "had_collision": had_collision,
    }


def _collect_regions(path_grid: np.ndarray) -> list[dict[str, object]]:
    h, w = path_grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(path_grid[r, c]) != BLACK:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            adjacent_colors: set[int] = set()
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(path_grid.shape, nr, nc):
                        continue
                    value = int(path_grid[nr, nc])
                    if value == BLACK:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                    elif value != GREY:
                        adjacent_colors.add(value)
            if len(adjacent_colors) == 0:
                fill_color = BLACK
            elif len(adjacent_colors) == 1:
                fill_color = next(iter(adjacent_colors))
            else:
                fill_color = YELLOW
            regions.append(
                {
                    "cells": cells,
                    "adjacent_colors": adjacent_colors,
                    "fill_color": fill_color,
                    "area": len(cells),
                }
            )
    return regions


def _paint_regions(path_grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.array(path_grid, copy=True)
    regions = _collect_regions(path_grid)
    for region in regions:
        fill_color = int(region["fill_color"])
        if fill_color == BLACK:
            continue
        for r, c in region["cells"]:
            out[r, c] = fill_color
    return out, regions


def _run_core(grid: np.ndarray, *, shared_walls: bool) -> tuple[np.ndarray, dict[str, object]]:
    seeds, blockers = _seed_info(grid)
    sim = _simulate_spirals(seeds, blockers, grid.shape, shared_walls=shared_walls)
    out, regions = _paint_regions(sim["path_grid"])
    sim["regions"] = regions
    return out, sim


def _simulate_without_blockers(grid: np.ndarray) -> np.ndarray:
    clean = np.array(grid, copy=True)
    clean[clean == GREY] = BLACK
    out, _ = _run_core(clean, shared_walls=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the seed pixels into mutually blocking synchronous spirals, then color each reachable background partition by the spiral walls around it.

    Input structure:
    - The grid is a black background with 2 or 3 distinct single-pixel colored seeds and several grey single-pixel blockers.
    - Each non-grey foreground color appears exactly once and marks one spiral's seed.
    - Grey blockers are static walls and all remaining black pixels are candidate flood-fill regions.

    Rule:
    - All seeds advance simultaneously on clockwise square spirals using directions up, right, down, left and leg lengths 1,1,2,2,3,3,...
    - On a step, if the next cell is out of bounds, grey, or already occupied by any spiral path, that spiral abandons the rest of the current leg, turns once, and terminates if the turned direction is immediately blocked too.
    - After all spirals stop, split the remaining black cells into 4-connected regions; a region adjacent to exactly one spiral color inherits that color, a region adjacent to multiple spiral colors becomes yellow, and a region adjacent to none stays black.
    - Seed cells, spiral path cells, and grey blockers remain unchanged.

    Key insight:
    - The final colors are not determined by distance to a seed; they depend on the synchronous order in which the spirals claim corridor cells and thereby create different enclosing walls before the flood-fill partitioning step.
    """
    out, _sim = _run_core(np.asarray(grid, dtype=int), shared_walls=True)
    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _sample_seed_positions(
    rng: np.random.Generator, h: int, w: int, count: int
) -> list[tuple[int, int]] | None:
    rows = range(MIN_SAFE_MARGIN, h - MIN_SAFE_MARGIN)
    cols = range(MIN_SAFE_MARGIN, w - MIN_SAFE_MARGIN)
    candidates = [(r, c) for r in rows for c in cols]
    if len(candidates) < count:
        return None
    for _ in range(80):
        rng.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if all(_manhattan((r, c), other) >= MIN_SEED_MANHATTAN for other in chosen):
                chosen.append((int(r), int(c)))
                if len(chosen) == count:
                    return chosen
    return None


def _trace_single_seed_path(
    seed: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    state: dict[str, int | bool] = {
        "r": int(seed[0]),
        "c": int(seed[1]),
        "dir_idx": 0,
        "leg_len": 1,
        "leg_progress": 0,
        "active": True,
    }
    own = {(int(seed[0]), int(seed[1]))}
    path: list[tuple[int, int]] = []
    safety_limit = shape[0] * shape[1] * 8
    for _ in range(safety_limit):
        if not bool(state["active"]):
            break
        r, c = int(state["r"]), int(state["c"])
        dr, dc = DIRS4[int(state["dir_idx"])]
        nr, nc = r + dr, c + dc
        if _blocked(shape, own, set(), nr, nc):
            _turn(state)  # type: ignore[arg-type]
            dr2, dc2 = DIRS4[int(state["dir_idx"])]
            tr, tc = r + dr2, c + dc2
            if _blocked(shape, own, set(), tr, tc):
                state["active"] = False
            continue
        own.add((nr, nc))
        path.append((nr, nc))
        state["r"], state["c"] = int(nr), int(nc)
        state["leg_progress"] = int(state["leg_progress"]) + 1
        if int(state["leg_progress"]) == int(state["leg_len"]):
            _turn(state)  # type: ignore[arg-type]
    return path


def _blocker_candidates_from_paths(
    seed_positions: list[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for seed in seed_positions:
        path = _trace_single_seed_path(seed, shape)
        lo = MIN_FIRST_BLOCK_STEPS
        hi = max(lo + 1, len(path) - 2)
        for cell in path[lo:hi]:
            if cell not in seen:
                seen.add(cell)
                candidates.append(cell)
    return candidates


def _sample_blockers(
    rng: np.random.Generator,
    h: int,
    w: int,
    seed_positions: list[tuple[int, int]],
    blocker_count: int,
) -> set[tuple[int, int]] | None:
    blockers: set[tuple[int, int]] = set()
    path_candidates = _blocker_candidates_from_paths(seed_positions, (h, w))
    rng.shuffle(path_candidates)

    def is_valid(cell: tuple[int, int]) -> bool:
        if cell in blockers or cell in seed_positions:
            return False
        if any(_manhattan(cell, seed) == 1 for seed in seed_positions):
            return False
        if any(_manhattan(cell, other) == 1 for other in blockers):
            return False
        return True

    must_take = min(len(path_candidates), max(1, blocker_count // 2))
    for cell in path_candidates:
        if len(blockers) >= must_take:
            break
        if is_valid(cell):
            blockers.add(cell)
    if len(blockers) < must_take:
        return None

    bbox_r0 = min(r for r, _c in seed_positions)
    bbox_c0 = min(c for _r, c in seed_positions)
    bbox_r1 = max(r for r, _c in seed_positions)
    bbox_c1 = max(c for _r, c in seed_positions)
    focal: list[tuple[int, int]] = []
    fringe: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if not is_valid(cell):
                continue
            if bbox_r0 - 2 <= r <= bbox_r1 + 2 and bbox_c0 - 2 <= c <= bbox_c1 + 2:
                focal.append(cell)
            else:
                fringe.append(cell)
    rng.shuffle(focal)
    rng.shuffle(fringe)
    for pool in (focal, fringe):
        for cell in pool:
            if len(blockers) >= blocker_count:
                break
            if is_valid(cell):
                blockers.add(cell)
        if len(blockers) >= blocker_count:
            break
    return blockers if len(blockers) == blocker_count else None


def _build_grid(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
    colors: list[int],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (r, c), color in zip(seeds, colors):
        grid[r, c] = int(color)
    for r, c in blockers:
        grid[r, c] = GREY
    return grid


def _analyze_regions(regions: list[dict[str, object]]) -> dict[str, int]:
    single_area = 0
    multi_area = 0
    single_count = 0
    multi_count = 0
    black_count = 0
    black_area = 0
    for region in regions:
        neighbors = set(region["adjacent_colors"])
        area = int(region["area"])
        if len(neighbors) == 0:
            black_count += 1
            black_area += area
        elif len(neighbors) == 1:
            single_count += 1
            single_area += area
        else:
            multi_count += 1
            multi_area += area
    return {
        "single_area": single_area,
        "multi_area": multi_area,
        "single_count": single_count,
        "multi_count": multi_count,
        "black_count": black_count,
        "black_area": black_area,
    }


def _seed_path_lengths(sim: dict[str, object]) -> list[int]:
    return [len(own) - 1 for own in sim["owner_sets"]]  # exclude the seed cell


def _grid_without_blockers(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    g[g == GREY] = BLACK
    return g


def _accept_generated_sample(
    grid: np.ndarray,
    *,
    require_multi_region: bool,
) -> bool:
    try:
        sync_out, sync_sim = _run_core(grid, shared_walls=True)
        independent_out, _independent_sim = _run_core(grid, shared_walls=False)
        no_blocker_out = _simulate_without_blockers(grid)
    except AssertionError:
        return False

    if bool(sync_sim["had_collision"]):
        return False
    if np.array_equal(sync_out, grid):
        return False
    if np.array_equal(sync_out, independent_out):
        return False
    if np.array_equal(sync_out, no_blocker_out):
        return False

    states = sync_sim["states"]
    if any(state["first_block_after"] is None for state in states):
        return False
    if any(int(state["first_block_after"]) < MIN_FIRST_BLOCK_STEPS for state in states):
        return False
    if any(length < MIN_FIRST_BLOCK_STEPS for length in _seed_path_lengths(sync_sim)):
        return False

    path_grid = sync_sim["path_grid"]
    path_only_area = int(np.count_nonzero((path_grid != BLACK) & (path_grid != GREY)))
    colored_area = int(np.count_nonzero((sync_out != BLACK) & (sync_out != GREY)))
    if colored_area - path_only_area < MIN_TOTAL_FILLED_AREA:
        return False

    region_stats = _analyze_regions(sync_sim["regions"])
    if region_stats["single_area"] < MIN_SINGLE_REGION_AREA:
        return False
    if require_multi_region and region_stats["multi_area"] < MIN_MULTI_REGION_AREA:
        return False
    if region_stats["single_count"] == 0:
        return False

    original_seeds, _blockers = _seed_info(grid)
    seed_colors = {color for _r, _c, color in original_seeds}
    visible_seed_colors = {int(v) for v in np.unique(sync_out) if int(v) in seed_colors}
    if visible_seed_colors != seed_colors:
        return False

    clean = _grid_without_blockers(grid)
    if np.array_equal(main(clean), sync_out):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template = GEN_TEMPLATES[int(rng.integers(0, len(GEN_TEMPLATES)))]
    h = int(rng.integers(max(15, GEN_HEIGHT_RANGE[0]), GEN_HEIGHT_RANGE[1]))
    w = int(rng.integers(max(15, GEN_WIDTH_RANGE[0]), GEN_WIDTH_RANGE[1]))
    require_multi_region = bool(rng.random() < 0.75)
    seed_positions = [tuple(map(int, rc)) for rc in template["seeds"]]
    blocker_idx = int(rng.integers(0, len(template["blocker_sets"])))
    base_blockers = [tuple(map(int, rc)) for rc in template["blocker_sets"][blocker_idx]]
    extra_pool = [tuple(map(int, rc)) for rc in template["extras"]]

    row_shift = int(rng.integers(0, h - 15 + 1))
    col_shift = int(rng.integers(0, w - 15 + 1))
    shifted_seeds = [(r + row_shift, c + col_shift) for r, c in seed_positions]
    blockers = {(r + row_shift, c + col_shift) for r, c in base_blockers}

    rng.shuffle(extra_pool)
    extra_count = int(rng.integers(0, 2))
    for r, c in extra_pool:
        if len(blockers) >= len(base_blockers) + extra_count:
            break
        shifted = (r + row_shift, c + col_shift)
        if shifted in blockers or shifted in shifted_seeds:
            continue
        if any(_manhattan(shifted, seed) == 1 for seed in shifted_seeds):
            continue
        if any(_manhattan(shifted, other) == 1 for other in blockers):
            continue
        blockers.add(shifted)

    seed_count = len(shifted_seeds)
    colors = list(map(int, rng.choice(SEED_COLORS, size=seed_count, replace=False)))
    grid = _build_grid((h, w), shifted_seeds, colors, blockers)
    if not _accept_generated_sample(grid, require_multi_region=require_multi_region):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-reachability input")
