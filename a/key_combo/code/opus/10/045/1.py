from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np


BACKGROUND = 0
SEED = 1
BLOCKER = 2

SIZE_MIN = 12
SIZE_MAX = 16
GEN_ATTEMPTS = 500

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
QUADRANTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
BRANCH_WEIGHTS = (0.30, 0.40, 0.30)  # oob, mid, next


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _quadrant(seed: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    sr, sc = seed
    r, c = cell
    assert r != sr and c != sc
    return (-1 if r < sr else 1, -1 if c < sc else 1)


def _build_grid(size: int, seed: tuple[int, int], blockers: tuple[tuple[int, int], ...]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    grid[seed] = SEED
    for r, c in blockers:
        grid[r, c] = BLOCKER
    return grid


def _simulate_spiral(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    seed_coords = np.argwhere(g == SEED)
    assert seed_coords.shape == (1, 2), "expected exactly one blue seed"

    out = g.copy()
    sr, sc = map(int, seed_coords[0])
    r, c = sr, sc
    direction_index = 0
    leg_length = 2
    completed_legs = 0
    path = {(sr, sc)}
    legs: list[tuple[tuple[int, int, bool], ...]] = []

    while True:
        dr, dc = DIRECTIONS[direction_index]
        current_leg: list[tuple[int, int, bool]] = []

        for _ in range(leg_length):
            r += dr
            c += dc
            in_bounds = _in_bounds(g.shape, r, c)
            current_leg.append((r, c, in_bounds))
            if not in_bounds:
                continue
            if g[r, c] == BLOCKER:
                return out, {
                    "stop_reason": "blocker_mid",
                    "completed_legs": completed_legs,
                    "path": frozenset(path),
                    "legs": tuple(legs),
                    "stop_cell": (r, c),
                }
            if out[r, c] == BACKGROUND:
                out[r, c] = SEED
            path.add((r, c))

        legs.append(tuple(current_leg))
        completed_legs += 1

        next_direction_index = (direction_index + 1) % 4
        next_leg_length = leg_length + (2 if next_direction_index in (0, 2) else 0)
        next_dr, next_dc = DIRECTIONS[next_direction_index]
        next_r, next_c = r + next_dr, c + next_dc

        if _in_bounds(g.shape, next_r, next_c) and g[next_r, next_c] == BLOCKER:
            return out, {
                "stop_reason": "next_first_blocker",
                "completed_legs": completed_legs,
                "path": frozenset(path),
                "legs": tuple(legs),
                "stop_cell": (next_r, next_c),
            }

        probe_r, probe_c = r, c
        all_oob = True
        for _ in range(next_leg_length):
            probe_r += next_dr
            probe_c += next_dc
            if _in_bounds(g.shape, probe_r, probe_c):
                all_oob = False
                break
        if all_oob:
            return out, {
                "stop_reason": "all_oob_next",
                "completed_legs": completed_legs,
                "path": frozenset(path),
                "legs": tuple(legs),
                "stop_cell": None,
            }

        direction_index = next_direction_index
        leg_length = next_leg_length


def _count_background_components(grid: np.ndarray) -> int:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BACKGROUND or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in DIRECTIONS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BACKGROUND and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, int]:
    painted, meta = _simulate_spiral(grid)
    region_count = _count_background_components(painted)
    assert region_count >= 1
    out = np.zeros((region_count, region_count), dtype=int)
    return out, painted, meta, region_count


def main(grid: np.ndarray) -> np.ndarray:
    """Count the background regions created by a clipped blocker-stopped square spiral and output a black square of that size.

    Input structure:
    - The grid is a black background with exactly one blue seed pixel and one to three red blocker pixels.
    - The seed is away from the border, and blockers are isolated single pixels in distinct quadrants around the seed.
    - The spiral is interpreted in the grid's `(row, col)` coordinate system, with out-of-bounds steps clipped away.

    Rule:
    - Start from the blue seed and trace directions up, right, down, left with leg lengths 2, 2, 4, 4, 6, 6, ...
    - Each step that lands on black turns blue; each out-of-bounds step is skipped but still consumes length; a red blocker stops the spiral immediately without being overwritten.
    - After the spiral stops, count the number of 4-connected black background regions and return an `N x N` all-black grid, where `N` is that count.

    Key insight:
    - The output size is controlled indirectly: blockers change where the spiral ends, the finished spiral changes the topology of the remaining black background, and only that final region count matters.
    """
    out, _painted, _meta, _region_count = _solve(np.asarray(grid, dtype=int))
    return out


def _candidate_stop_cells(profile: dict, branch: str) -> tuple[tuple[int, int], ...]:
    return profile["mid_cells"] if branch == "mid" else profile["next_cells"]


@lru_cache(maxsize=None)
def _seed_catalog(size: int) -> tuple[dict, ...]:
    profiles: list[dict] = []
    center = (size - 1) / 2.0

    for sr in range(2, size - 2):
        for sc in range(2, size - 2):
            seed = (sr, sc)
            base_grid = _build_grid(size, seed, ())
            base_painted, meta = _simulate_spiral(base_grid)
            base_regions = _count_background_components(base_painted)
            if meta["completed_legs"] < 3 or base_regions < 2:
                continue

            mid_cells: list[tuple[int, int]] = []
            next_cells: list[tuple[int, int]] = []
            for leg_index, leg in enumerate(meta["legs"]):
                if leg_index < 3:
                    continue
                in_bounds_cells = [(step_index, (r, c)) for step_index, (r, c, ok) in enumerate(leg) if ok]
                if not in_bounds_cells:
                    continue
                if in_bounds_cells[0][0] == 0 and in_bounds_cells[0][1][0] != sr and in_bounds_cells[0][1][1] != sc:
                    next_cells.append(in_bounds_cells[0][1])
                mid_cells.extend(
                    cell
                    for step_index, cell in in_bounds_cells[1:]
                    if cell[0] != sr and cell[1] != sc
                )

            profiles.append(
                {
                    "seed": seed,
                    "base_painted": base_painted,
                    "base_regions": base_regions,
                    "base_path": meta["path"],
                    "edge_bias": max(abs(sr - center), abs(sc - center)),
                    "mid_cells": tuple(mid_cells),
                    "next_cells": tuple(next_cells),
                }
            )

    return tuple(profiles)


def _choose_seed_profile(size: int, branch: str) -> dict | None:
    profiles = list(_seed_catalog(size))
    if branch == "oob":
        candidates = [p for p in profiles if 2 <= p["base_regions"] <= 6]
        key = lambda p: (p["base_regions"], p["edge_bias"])
    elif branch == "mid":
        candidates = [p for p in profiles if p["mid_cells"]]
        key = lambda p: (len(p["mid_cells"]), p["base_regions"], p["edge_bias"])
    else:
        candidates = [p for p in profiles if p["next_cells"]]
        key = lambda p: (len(p["next_cells"]), p["base_regions"], p["edge_bias"])

    if not candidates:
        return None

    candidates.sort(key=key, reverse=True)
    top_n = min(len(candidates), max(10, len(candidates) // 2))
    return random.choice(candidates[:top_n])


def _quadrant_cells(
    size: int,
    seed: tuple[int, int],
    quadrant: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    sr, sc = seed
    cells = []
    for r in range(size):
        for c in range(size):
            if (r, c) == seed or r == sr or c == sc:
                continue
            if max(abs(r - sr), abs(c - sc)) < 2:
                continue
            if (r, c) in forbidden:
                continue
            if _quadrant(seed, (r, c)) == quadrant:
                cells.append((r, c))
    return cells


def _pick_isolated_cell(
    candidates: list[tuple[int, int]],
    used: list[tuple[int, int]],
) -> tuple[int, int] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for cell in shuffled:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in used):
            return cell
    return None


def _check_semantics(grid: np.ndarray, branch: str) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not (SIZE_MIN <= g.shape[0] <= SIZE_MAX):
        return False

    seed_coords = np.argwhere(g == SEED)
    blocker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == BLOCKER)]
    if seed_coords.shape != (1, 2) or not (1 <= len(blocker_coords) <= 3):
        return False

    seed = tuple(map(int, seed_coords[0]))
    if min(seed[0], seed[1], g.shape[0] - 1 - seed[0], g.shape[1] - 1 - seed[1]) < 2:
        return False

    blocker_quadrants = []
    for blocker in blocker_coords:
        if blocker[0] == seed[0] or blocker[1] == seed[1]:
            return False
        if max(abs(blocker[0] - seed[0]), abs(blocker[1] - seed[1])) < 2:
            return False
        blocker_quadrants.append(_quadrant(seed, blocker))
    if len(set(blocker_quadrants)) != len(blocker_coords):
        return False
    for i, a in enumerate(blocker_coords):
        for b in blocker_coords[i + 1 :]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                return False

    expected, painted, meta, regions = _solve(g)
    if meta["completed_legs"] < 3 or not (2 <= regions <= 6):
        return False

    if branch == "oob":
        if meta["stop_reason"] != "all_oob_next":
            return False
        base_painted, base_meta = _simulate_spiral(_build_grid(g.shape[0], seed, ()))
        if base_meta["stop_reason"] != "all_oob_next":
            return False
        if not np.array_equal(painted == SEED, base_painted == SEED):
            return False
    else:
        target_reason = "blocker_mid" if branch == "mid" else "next_first_blocker"
        if meta["stop_reason"] != target_reason:
            return False
        stop_cell = meta["stop_cell"]
        if stop_cell not in blocker_coords:
            return False
        reduced_blockers = tuple(cell for cell in blocker_coords if cell != stop_cell)
        alt_painted, alt_meta = _simulate_spiral(_build_grid(g.shape[0], seed, reduced_blockers))
        same_blue_path = np.array_equal(alt_painted == SEED, painted == SEED)
        if alt_meta["stop_reason"] == meta["stop_reason"] and same_blue_path:
            return False

    return np.array_equal(main(g), expected)


def _generate_oob_case(profile: dict, obstacle_count: int) -> np.ndarray | None:
    seed = profile["seed"]
    size = profile["base_painted"].shape[0]
    forbidden = set(profile["base_path"])
    quadrants = random.sample(QUADRANTS, obstacle_count)
    blockers: list[tuple[int, int]] = []

    for quadrant in quadrants:
        cell = _pick_isolated_cell(_quadrant_cells(size, seed, quadrant, forbidden), blockers)
        if cell is None:
            return None
        blockers.append(cell)

    grid = _build_grid(size, seed, tuple(blockers))
    return grid if _check_semantics(grid, "oob") else None


def _generate_blocker_case(profile: dict, branch: str, obstacle_count: int) -> np.ndarray | None:
    seed = profile["seed"]
    size = profile["base_painted"].shape[0]
    forbidden = set(profile["base_path"])
    accepted: list[tuple[int, np.ndarray]] = []
    stop_cells = list(_candidate_stop_cells(profile, branch))
    random.shuffle(stop_cells)

    for stop_cell in stop_cells[:80]:
        stop_quadrant = _quadrant(seed, stop_cell)
        remaining_quadrants = [quadrant for quadrant in QUADRANTS if quadrant != stop_quadrant]
        if obstacle_count - 1 > len(remaining_quadrants):
            continue

        for extra_quadrants in combinations(remaining_quadrants, obstacle_count - 1):
            blockers = [stop_cell]
            valid = True
            for quadrant in extra_quadrants:
                cell = _pick_isolated_cell(_quadrant_cells(size, seed, quadrant, forbidden), blockers)
                if cell is None:
                    valid = False
                    break
                blockers.append(cell)
            if not valid:
                continue

            grid = _build_grid(size, seed, tuple(blockers))
            if not _check_semantics(grid, branch):
                continue

            regions = main(grid).shape[0]
            accepted.append((regions, grid))

    if not accepted:
        return None

    accepted.sort(key=lambda item: item[0], reverse=True)
    top_n = min(len(accepted), 12)
    return random.choice(accepted[:top_n])[1]


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        branch = random.choices(("oob", "mid", "next"), weights=BRANCH_WEIGHTS, k=1)[0]
        size = random.randint(SIZE_MIN, SIZE_MAX)
        obstacle_count = random.randint(1, 3)
        profile = _choose_seed_profile(size, branch)
        if profile is None:
            continue

        if branch == "oob":
            grid = _generate_oob_case(profile, obstacle_count)
        else:
            grid = _generate_blocker_case(profile, branch, obstacle_count)

        if grid is not None:
            return grid

    raise RuntimeError("failed to generate a valid spiral-region puzzle")
