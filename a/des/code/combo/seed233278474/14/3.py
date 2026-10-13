from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_RETRIES = 500


def _bg_color(grid: np.ndarray) -> int:
    return 0 if np.any(grid == 0) else int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _distance_to_edge(r: int, c: int, h: int, w: int, direction: int) -> int:
    if direction == 0:
        return r
    if direction == 1:
        return w - 1 - c
    if direction == 2:
        return h - 1 - r
    return c


def _geometry_direction(r: int, c: int, h: int, w: int) -> int:
    best_dir = 0
    best_dist = -1
    for direction in range(4):
        dist = _distance_to_edge(r, c, h, w, direction)
        if dist > best_dist:
            best_dist = dist
            best_dir = direction
    return best_dir


def _open_distance(grid: np.ndarray, r: int, c: int, bg: int, direction: int) -> int | None:
    h, w = grid.shape
    dr, dc = DIRS_URDL[direction]
    steps = _distance_to_edge(r, c, h, w, direction)
    rr, cc = r + dr, c + dc
    for _ in range(steps):
        if int(grid[rr, cc]) != bg:
            return None
        rr += dr
        cc += dc
    return steps


def _direction_info(grid: np.ndarray, r: int, c: int, bg: int) -> dict[str, int | bool] | None:
    h, w = grid.shape
    best_dir = None
    best_dist = -1
    tie = False
    open_dists: list[int | None] = []

    for direction in range(4):
        dist = _open_distance(grid, r, c, bg, direction)
        open_dists.append(dist)
        if dist is None:
            continue
        if dist > best_dist:
            best_dist = dist
            best_dir = direction
            tie = False
        elif dist == best_dist:
            tie = True

    if best_dir is None:
        return None

    blocked = any(
        open_dists[direction] is None and _distance_to_edge(r, c, h, w, direction) > best_dist
        for direction in range(4)
    )
    return {
        "dir": int(best_dir),
        "dist": int(best_dist),
        "tie": bool(tie),
        "blocked": bool(blocked),
        "geometry_dir": int(_geometry_direction(r, c, h, w)),
    }


def _growth_path(r: int, c: int, direction: int, length: int, h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = DIRS_URDL[direction]
    path = [(r, c)]
    rr, cc = r, c
    for _ in range(length - 1):
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        path.append((rr, cc))
    return path


def _corridor_to_edge(r: int, c: int, direction: int, h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = DIRS_URDL[direction]
    rr, cc = r + dr, c + dc
    cells = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _seed_cells(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != bg)]


def _has_adjacent_seed(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != bg:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a straight branch from each colored seed.

    The branch direction is the longest unobstructed cardinal direction, with
    ties resolved by the fixed order up, right, down, left. The branch length
    equals the total number of seeds of the same color, counting the seed
    itself and truncating at the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    seeds = _seed_cells(g, bg)
    if not seeds:
        return out

    color_counts = Counter(color for _r, _c, color in seeds)
    h, w = g.shape

    for r, c, color in seeds:
        info = _direction_info(g, r, c, bg)
        assert info is not None, "every seed must have at least one open direction"
        for rr, cc in _growth_path(r, c, int(info["dir"]), int(color_counts[color]), h, w):
            out[rr, cc] = color

    return out


def _collect_candidates(
    grid: np.ndarray,
    length: int,
    occupied_output: set[tuple[int, int]],
    reserved_corridors: set[tuple[int, int]],
    bg: int,
) -> list[dict[str, object]]:
    h, w = grid.shape
    candidates = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            if (r, c) in occupied_output or (r, c) in reserved_corridors:
                continue
            if _has_adjacent_seed(grid, r, c, bg):
                continue

            info = _direction_info(grid, r, c, bg)
            if info is None:
                continue

            direction = int(info["dir"])
            path = _growth_path(r, c, direction, length, h, w)
            corridor = _corridor_to_edge(r, c, direction, h, w)
            if any(cell in occupied_output or cell in reserved_corridors for cell in path):
                continue
            if any(cell in occupied_output or cell in reserved_corridors for cell in corridor):
                continue

            candidates.append(
                {
                    "pos": (r, c),
                    "dir": direction,
                    "path": path,
                    "corridor": corridor,
                    "tie": bool(info["tie"]),
                    "blocked": bool(info["blocked"]),
                    "truncated": len(path) < length,
                }
            )

    return candidates


def _pick_candidate(
    candidates: list[dict[str, object]],
    dir_hist: Counter,
    need_trunc: bool,
    need_tie: bool,
    rng: random.Random,
) -> dict[str, object]:
    pool = list(candidates)

    if need_tie:
        tie_pool = [cand for cand in pool if bool(cand["tie"])]
        if tie_pool:
            pool = tie_pool
    if need_trunc:
        trunc_pool = [cand for cand in pool if bool(cand["truncated"])]
        if trunc_pool:
            pool = trunc_pool

    min_dir_load = min(int(dir_hist[int(cand["dir"])]) for cand in pool)
    pool = [cand for cand in pool if int(dir_hist[int(cand["dir"])]) == min_dir_load]

    blocked_pool = [cand for cand in pool if bool(cand["blocked"])]
    if blocked_pool and rng.random() < 0.45:
        pool = blocked_pool

    return rng.choice(pool)


def _verify_instance(
    grid: np.ndarray,
    target_counts: dict[int, int],
    need_trunc: bool,
    need_tie: bool,
) -> bool:
    bg = _bg_color(grid)
    seeds = _seed_cells(grid, bg)
    if not seeds:
        return False

    if Counter(color for _r, _c, color in seeds) != Counter(target_counts):
        return False

    used_dirs = set()
    claimed: set[tuple[int, int]] = set()
    has_tie = False
    has_trunc = False
    has_blocked = False
    h, w = grid.shape

    for r, c, color in seeds:
        if _has_adjacent_seed(grid, r, c, bg):
            return False
        info = _direction_info(grid, r, c, bg)
        if info is None:
            return False
        used_dirs.add(int(info["dir"]))
        has_tie |= bool(info["tie"])
        has_blocked |= bool(info["blocked"])
        path = _growth_path(r, c, int(info["dir"]), int(target_counts[color]), h, w)
        has_trunc |= len(path) < int(target_counts[color])
        for cell in path:
            if cell in claimed:
                return False
            claimed.add(cell)

    if len(used_dirs) < 2:
        return False
    if need_trunc and not has_trunc:
        return False
    if need_tie and not has_tie:
        return False
    if not (has_trunc or has_tie or has_blocked):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_colors = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, num_colors)
    color_counts = {color: rng.randint(2, 5) for color in colors}
    if num_colors > 1 and len(set(color_counts.values())) == 1:
        color = rng.choice(colors)
        choices = [count for count in range(2, 6) if count != color_counts[color]]
        color_counts[color] = rng.choice(choices)

    total_seeds = sum(color_counts.values())
    side_lo = max(10, total_seeds // 2 + 6, max(color_counts.values()) + 4)
    side_hi = min(24, side_lo + 6)
    if side_lo > side_hi:
        side_lo = side_hi

    h = rng.randint(side_lo, side_hi)
    w = rng.randint(side_lo, side_hi)
    grid = np.zeros((h, w), dtype=int)

    specs = []
    for color in colors:
        specs.extend([{"color": color, "length": color_counts[color]}] * color_counts[color])
    rng.shuffle(specs)
    specs.sort(key=lambda spec: -int(spec["length"]))

    occupied_output: set[tuple[int, int]] = set()
    reserved_corridors: set[tuple[int, int]] = set()
    dir_hist: Counter = Counter()
    seen_tie = False
    seen_trunc = False

    need_trunc = rng.random() < 0.6
    need_tie = rng.random() < 0.35

    for idx, spec in enumerate(specs):
        remaining = len(specs) - idx
        candidates = _collect_candidates(grid, int(spec["length"]), occupied_output, reserved_corridors, 0)
        if not candidates:
            return None

        candidate = _pick_candidate(
            candidates,
            dir_hist,
            need_trunc=need_trunc and not seen_trunc and remaining <= max(5, len(specs) // 2),
            need_tie=need_tie and not seen_tie,
            rng=rng,
        )

        r, c = candidate["pos"]
        grid[r, c] = int(spec["color"])
        occupied_output.update(candidate["path"])
        reserved_corridors.update(candidate["corridor"])
        dir_hist[int(candidate["dir"])] += 1
        seen_tie |= bool(candidate["tie"])
        seen_trunc |= bool(candidate["truncated"])

    if not _verify_instance(grid, color_counts, need_trunc=need_trunc, need_tie=need_tie):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
