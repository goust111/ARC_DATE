from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLOCKER = 5
SEED = 6
LEGEND_COLOR_POOL = (1, 2, 3, 4, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _parse_legend(grid: np.ndarray) -> list[int]:
    top = np.asarray(grid[0], dtype=int)
    k = 0
    while k < top.size and int(top[k]) != BACKGROUND:
        k += 1
    assert 3 <= k <= 5, "expected a 3-5 cell legend prefix on the top row"
    assert bool(np.all(top[k:] == BACKGROUND)), "legend must be a left-aligned prefix"
    legend = [int(v) for v in top[:k]]
    assert len(set(legend)) == k, "legend colors must be distinct"
    assert all(color not in (BACKGROUND, BLOCKER, SEED) for color in legend)
    return legend


def _seed_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(np.asarray(grid, dtype=int) == SEED)
    assert coords.shape == (1, 2), "expected exactly one seed pixel"
    r, c = map(int, coords[0])
    assert r > 0, "seed must not be on the legend row"
    return r, c


def _blockers(grid: np.ndarray) -> set[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid, dtype=int) == BLOCKER)}
    assert 1 <= len(coords) <= 2, "expected one or two blockers"
    return coords


def _spiral_steps(shape: tuple[int, int], seed: tuple[int, int]) -> list[tuple[int, int, int, bool]]:
    h, w = map(int, shape)
    r, c = map(int, seed)
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg_length = 2 * (max_delta + 1)
    steps: list[tuple[int, int, int, bool]] = []
    step_index = 0
    leg_length = 2
    direction_index = 0

    while leg_length <= max_leg_length:
        dr, dc = DIRS[direction_index]
        for _ in range(leg_length):
            r += dr
            c += dc
            steps.append((step_index, r, c, _in_bounds((h, w), r, c)))
            step_index += 1
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 2
    return steps


def _run_spiral(
    base_grid: np.ndarray,
    legend: list[int],
    seed: tuple[int, int],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(base_grid, dtype=int).copy()
    legend_len = len(legend)
    for step_index, r, c, inside in _spiral_steps(out.shape, seed):
        if not inside:
            continue
        if (r, c) in blockers:
            break
        if out[r, c] == BACKGROUND:
            out[r, c] = legend[step_index % legend_len]
    return out


def _simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend = _parse_legend(g)
    blockers = _blockers(g)
    seed = _seed_coord(g)
    return _run_spiral(g, legend, seed, blockers)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-edge legend string as a cyclic color program for a clipped spiral that stops at the first blocker.

    Input structure:
    - The grid is mostly black background.
    - Row 0 starts with a contiguous 3-5 cell horizontal legend string of distinct non-black colors, followed only by black cells.
    - Away from the top row there is exactly one magenta seed pixel and one or two gray blocker pixels.

    Rule:
    - Read the legend colors left-to-right from the top-row prefix.
    - Starting from the seed, trace the unbounded spiral with direction order up, right, down, left and leg lengths 2, 2, 4, 4, 6, 6, ...
    - For each spiral step, out-of-bounds positions are skipped but still advance the global step index; the first in-bounds blocker cell stops the process before recoloring it; any other in-bounds background cell is colored by legend[step_index mod legend_length].
    - The seed, the legend string, the blockers, and every untouched background cell remain unchanged.

    Key insight:
    - The top-row legend is not a spatial template: it is a repeating instruction sequence whose index is driven by the spiral's full clipped trajectory, including skipped off-grid steps.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _painted_mask(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    return (np.asarray(inp, dtype=int) == BACKGROUND) & (np.asarray(out, dtype=int) != BACKGROUND)


def _prefix_suffix_stats(
    steps: list[tuple[int, int, int, bool]],
    legend_len: int,
    grid_shape: tuple[int, int],
) -> tuple[list[dict], list[int]]:
    h, w = map(int, grid_shape)
    paintable = []
    for step_index, r, c, inside in steps:
        legend_cell = inside and r == 0 and 0 <= c < legend_len
        paintable.append(bool(inside and not legend_cell))

    suffix_future = [0] * (len(steps) + 1)
    for i in range(len(steps) - 1, -1, -1):
        suffix_future[i] = suffix_future[i + 1] + int(paintable[i])

    prefix: list[dict] = []
    painted_before = 0
    residues_before: set[int] = set()
    out_of_bounds_before = False
    for idx, (step_index, r, c, inside) in enumerate(steps):
        prefix.append(
            {
                "idx": idx,
                "step_index": int(step_index),
                "r": int(r),
                "c": int(c),
                "inside": bool(inside),
                "legend_cell": bool(inside and r == 0 and 0 <= c < legend_len),
                "painted_before": int(painted_before),
                "residue_count_before": int(len(residues_before)),
                "oob_before": bool(out_of_bounds_before),
                "future_paintable": int(suffix_future[idx + 1]),
            }
        )
        if not inside:
            out_of_bounds_before = True
        elif not (r == 0 and 0 <= c < legend_len):
            painted_before += 1
            residues_before.add(int(step_index) % int(legend_len))
    return prefix, suffix_future


def _seed_pool(h: int, w: int, legend_len: int, require_oob: bool) -> list[tuple[int, int]]:
    seeds = []
    for r in range(1, h - 1):
        for c in range(max(1, legend_len), w - 1):
            edge_distance = min(r, h - 1 - r, c, w - 1 - c)
            if require_oob and edge_distance > 2:
                continue
            if not require_oob and edge_distance < 3:
                continue
            seeds.append((r, c))
    if not seeds:
        for r in range(1, h - 1):
            for c in range(max(1, legend_len), w - 1):
                seeds.append((r, c))
    random.shuffle(seeds)
    return seeds


def _stop_candidates(
    h: int,
    w: int,
    legend_len: int,
    seed: tuple[int, int],
    require_oob: bool,
) -> list[tuple[int, int, int]]:
    steps = _spiral_steps((h, w), seed)
    prefix, _ = _prefix_suffix_stats(steps, legend_len, (h, w))
    candidates = []
    min_painted = max(8, legend_len + 2)
    min_residues = min(3, legend_len)

    for meta in prefix:
        if not meta["inside"] or meta["legend_cell"]:
            continue
        if meta["painted_before"] < min_painted:
            continue
        if meta["residue_count_before"] < min_residues:
            continue
        if bool(meta["oob_before"]) != bool(require_oob):
            continue
        if meta["future_paintable"] <= 0:
            continue
        if meta["step_index"] < legend_len + 3:
            continue
        candidates.append((meta["idx"], meta["r"], meta["c"]))
    return candidates


def _extra_blocker_choices(
    h: int,
    w: int,
    legend_len: int,
    seed: tuple[int, int],
    stop_idx: int,
    stop_cell: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    steps = _spiral_steps((h, w), seed)
    before_stop = {
        (int(r), int(c))
        for idx, (_step_index, r, c, inside) in enumerate(steps)
        if idx < stop_idx and inside and not (r == 0 and 0 <= c < legend_len)
    }
    later_path = [
        (int(r), int(c))
        for idx, (_step_index, r, c, inside) in enumerate(steps)
        if idx > stop_idx
        and inside
        and not (r == 0 and 0 <= c < legend_len)
        and (int(r), int(c)) != stop_cell
        and abs(int(r) - seed[0]) + abs(int(c) - seed[1]) >= 2
    ]
    later_path = list(dict.fromkeys(later_path))

    off_path = []
    path_cells = {(int(r), int(c)) for _step_index, r, c, inside in steps if inside}
    for r in range(h):
        for c in range(w):
            cell = (int(r), int(c))
            if r == 0 and c < legend_len:
                continue
            if cell == seed or cell == stop_cell:
                continue
            if abs(r - seed[0]) + abs(c - seed[1]) < 2:
                continue
            if cell in before_stop:
                continue
            if cell not in path_cells:
                off_path.append(cell)
    return later_path, off_path


def _build_attempt() -> np.ndarray | None:
    h = random.randint(12, 17)
    w = random.randint(12, 17)
    legend_len = random.randint(3, 5)
    require_oob = random.random() < 0.55
    legend = random.sample(list(LEGEND_COLOR_POOL), legend_len)
    seed = None
    stop_idx = None
    stop_cell = None

    seeds = _seed_pool(h, w, legend_len, require_oob)
    for candidate_seed in seeds:
        candidates = _stop_candidates(h, w, legend_len, candidate_seed, require_oob)
        if candidates:
            seed = candidate_seed
            stop_idx, stop_r, stop_c = random.choice(candidates)
            stop_cell = (int(stop_r), int(stop_c))
            break
    if seed is None or stop_idx is None or stop_cell is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :legend_len] = np.array(legend, dtype=int)
    grid[seed] = SEED
    grid[stop_cell] = BLOCKER

    total_blockers = 1 if random.random() < 0.5 else 2
    if total_blockers == 2:
        later_path, off_path = _extra_blocker_choices(h, w, legend_len, seed, stop_idx, stop_cell)
        use_later_path = later_path and (not off_path or random.random() < 0.6)
        pool = later_path if use_later_path else off_path
        if not pool:
            return None
        grid[random.choice(pool)] = BLOCKER

    return grid if _passes_semantic_checks(grid) else None


def _passes_semantic_checks(grid: np.ndarray) -> bool:
    try:
        legend = _parse_legend(grid)
        seed = _seed_coord(grid)
        blockers = _blockers(grid)
    except AssertionError:
        return False

    if set(map(int, np.unique(grid))) - ({BACKGROUND, BLOCKER, SEED} | set(legend)):
        return False
    if np.count_nonzero(grid[1:] == SEED) != 1 or np.any(grid[0] == SEED):
        return False
    if np.any(grid[0, len(legend) :] != BACKGROUND):
        return False

    out = main(grid)
    expected = _simulate(grid)
    if not np.array_equal(out, expected):
        return False

    painted = _painted_mask(grid, out)
    painted_colors = {int(v) for v in out[painted]}
    if int(painted.sum()) < max(8, len(legend) + 2):
        return False
    if len(painted_colors) < min(3, len(legend)):
        return False

    blocker_mask = grid == BLOCKER
    no_blocker = grid.copy()
    no_blocker[blocker_mask] = BACKGROUND
    out_without_blockers = _run_spiral(no_blocker, legend, seed, set())
    if not np.any(out_without_blockers[~blocker_mask] != out[~blocker_mask]):
        return False

    rotated_legend = legend[1:] + legend[:1]
    alt = grid.copy()
    alt[0, : len(legend)] = np.array(rotated_legend, dtype=int)
    alt_out = main(alt)
    if not np.any(alt_out[1:] != out[1:]):
        return False

    steps = _spiral_steps(grid.shape, seed)
    oob_before_stop = False
    stop_seen = False
    for _step_index, r, c, inside in steps:
        if not inside:
            oob_before_stop = True
            continue
        if (r, c) in blockers:
            stop_seen = True
            break
    if not stop_seen:
        return False

    if oob_before_stop:
        residues = {i % len(legend) for i, r, c, inside in steps if inside and (r, c) not in blockers and out[r, c] in legend}
        if len(residues) < 2:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-legend ARC input")
