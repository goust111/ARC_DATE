import random
from collections import Counter

import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
COUNT_POOL = (1, 2, 3, 4, 5)
MAX_TOTAL_SEEDS = 11
GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _paint_cell(out: np.ndarray, best_power: np.ndarray, r: int, c: int, color: int, power: int) -> None:
    if power > int(best_power[r, c]):
        out[r, c] = color
        best_power[r, c] = power


def _paint_cross(
    out: np.ndarray,
    best_power: np.ndarray,
    seed: tuple[int, int],
    color: int,
    power: int,
) -> None:
    h, w = out.shape
    r, c = seed
    for delta in range(-power, power + 1):
        rr = r + delta
        if 0 <= rr < h:
            _paint_cell(out, best_power, rr, c, color, power)
        cc = c + delta
        if 0 <= cc < w:
            _paint_cell(out, best_power, r, cc, color, power)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow every seed pixel into a same-color cross whose arm length equals that color's total seed count.

    Input structure:
    - The background is the most common color, typically black.
    - Every non-background pixel is an isolated one-cell seed.
    - There are 2 to 4 non-background colors, and each color appears a distinct number of times.

    Rule:
    - For each color, count its total number of seed pixels N in the whole grid.
    - From every seed of that color, extend N cells up, down, left, and right, clipped by the grid boundary.
    - Keep the seed pixel itself as the center of the cross.
    - If multiple colors try to paint the same cell, the color with larger N wins; equal-N ties keep the earlier paint.

    Key insight:
    - Both the cross size and the overlap priority come from one global property: how many seeds of that color exist anywhere in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    seed_counts = Counter(int(v) for v in g.flat if int(v) != bg)

    out = np.full_like(g, bg)
    best_power = np.full(g.shape, -1, dtype=int)
    for color, power in seed_counts.items():
        for r, c in np.argwhere(g == color):
            _paint_cross(out, best_power, (int(r), int(c)), int(color), int(power))
    return out


# --------------- generator ---------------


def _seed_footprint(shape: tuple[int, int], seed: tuple[int, int], arm: int) -> set[tuple[int, int]]:
    h, w = shape
    r, c = seed
    cells: set[tuple[int, int]] = set()
    for delta in range(-arm, arm + 1):
        rr = r + delta
        if 0 <= rr < h:
            cells.add((rr, c))
        cc = c + delta
        if 0 <= cc < w:
            cells.add((r, cc))
    return cells


def _add_seed(
    placed: list[tuple[int, tuple[int, int], set[tuple[int, int]]]],
    seeds_by_color: dict[int, list[tuple[int, int]]],
    color: int,
    seed: tuple[int, int],
    counts: dict[int, int],
    shape: tuple[int, int],
) -> None:
    footprint = _seed_footprint(shape, seed, counts[color])
    placed.append((color, seed, footprint))
    seeds_by_color[color].append(seed)


def _can_place(
    seed: tuple[int, int],
    color: int,
    shape: tuple[int, int],
    placed: list[tuple[int, tuple[int, int], set[tuple[int, int]]]],
    counts: dict[int, int],
) -> set[tuple[int, int]] | None:
    r, c = seed
    for _other_color, (orow, ocol), _other_footprint in placed:
        if max(abs(r - orow), abs(c - ocol)) <= 1:
            return None

    footprint = _seed_footprint(shape, seed, counts[color])
    for _other_color, _other_seed, other_footprint in placed:
        if footprint & other_footprint:
            return None
    return footprint


def _interior_candidates(shape: tuple[int, int], arm: int) -> list[tuple[int, int]]:
    h, w = shape
    if h <= 2 * arm or w <= 2 * arm:
        return []
    return [(r, c) for r in range(arm, h - arm) for c in range(arm, w - arm)]


def _truncation_candidates(shape: tuple[int, int], arm: int) -> list[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(arm):
        for c in range(arm, w - arm):
            cells.add((r, c))
            cells.add((h - 1 - r, c))
    for r in range(arm, h - arm):
        for c in range(arm):
            cells.add((r, c))
            cells.add((r, w - 1 - c))
    return list(cells)


def _sample_counts(rng: random.Random, n_colors: int) -> list[int]:
    while True:
        counts = sorted(rng.sample(COUNT_POOL, n_colors))
        if sum(counts) <= MAX_TOTAL_SEEDS:
            rng.shuffle(counts)
            return counts


def _grid_shape_for_counts(rng: random.Random, counts: list[int]) -> tuple[int, int]:
    max_arm = max(counts)
    footprint_load = sum(arm * (4 * arm + 1) for arm in counts)
    min_side = max(14, 2 * max_arm + 7, int(np.ceil(np.sqrt(footprint_load * 2.4))))
    max_side = min(30, min_side + 5)
    return rng.randint(min_side, max_side), rng.randint(min_side, max_side)


def _place_overlap_pair(
    rng: random.Random,
    shape: tuple[int, int],
    placed: list[tuple[int, tuple[int, int], set[tuple[int, int]]]],
    seeds_by_color: dict[int, list[tuple[int, int]]],
    counts: dict[int, int],
    long_color: int,
    short_color: int,
) -> bool:
    long_arm = counts[long_color]
    short_arm = counts[short_color]
    large_gap = short_arm + 1
    small_gap = 1

    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr_abs, dc_abs in ((small_gap, large_gap), (large_gap, small_gap)):
        for dr_sign in (-1, 1):
            for dc_sign in (-1, 1):
                dr = dr_sign * dr_abs
                dc = dc_sign * dc_abs

                row_lo = max(long_arm, short_arm - dr)
                row_hi = min(shape[0] - 1 - long_arm, shape[0] - 1 - short_arm - dr)
                col_lo = max(long_arm, short_arm - dc)
                col_hi = min(shape[1] - 1 - long_arm, shape[1] - 1 - short_arm - dc)
                if row_lo > row_hi or col_lo > col_hi:
                    continue

                for r in range(row_lo, row_hi + 1):
                    for c in range(col_lo, col_hi + 1):
                        long_seed = (r, c)
                        short_seed = (r + dr, c + dc)
                        long_fp = _seed_footprint(shape, long_seed, long_arm)
                        short_fp = _seed_footprint(shape, short_seed, short_arm)
                        if len(long_fp & short_fp) == 1:
                            options.append((long_seed, short_seed))

    if not options:
        return False

    long_seed, short_seed = rng.choice(options)
    _add_seed(placed, seeds_by_color, long_color, long_seed, counts, shape)
    _add_seed(placed, seeds_by_color, short_color, short_seed, counts, shape)
    return True


def _build_grid(shape: tuple[int, int], seeds_by_color: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, seeds in seeds_by_color.items():
        for r, c in seeds:
            grid[r, c] = color
    return grid


def _coverage_by_cell(
    shape: tuple[int, int],
    seeds_by_color: dict[int, list[tuple[int, int]]],
    counts: dict[int, int],
) -> dict[tuple[int, int], set[int]]:
    coverage: dict[tuple[int, int], set[int]] = {}
    for color, seeds in seeds_by_color.items():
        arm = counts[color]
        for seed in seeds:
            for cell in _seed_footprint(shape, seed, arm):
                coverage.setdefault(cell, set()).add(color)
    return coverage


def _has_truncation(
    shape: tuple[int, int],
    seeds_by_color: dict[int, list[tuple[int, int]]],
    counts: dict[int, int],
) -> bool:
    for color, seeds in seeds_by_color.items():
        arm = counts[color]
        full_size = 4 * arm + 1
        for seed in seeds:
            if len(_seed_footprint(shape, seed, arm)) < full_size:
                return True
    return False


def _seeds_are_isolated(seeds_by_color: dict[int, list[tuple[int, int]]]) -> bool:
    all_seeds = [seed for seeds in seeds_by_color.values() for seed in seeds]
    for i, (r1, c1) in enumerate(all_seeds):
        for r2, c2 in all_seeds[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _validate_scene(
    grid: np.ndarray,
    seeds_by_color: dict[int, list[tuple[int, int]]],
    counts: dict[int, int],
    expect_overlap: bool,
    expect_truncation: bool,
) -> bool:
    if _get_bg(grid) != 0:
        return False
    if not _seeds_are_isolated(seeds_by_color):
        return False

    present = {int(v) for v in grid.flat if int(v) != 0}
    if present != set(counts):
        return False

    actual_counts = {color: int(np.count_nonzero(grid == color)) for color in counts}
    if actual_counts != counts:
        return False
    if len(set(actual_counts.values())) != len(actual_counts):
        return False

    coverage = _coverage_by_cell(grid.shape, seeds_by_color, counts)
    overlap_cells = {cell for cell, colors in coverage.items() if len(colors) >= 2}
    if bool(overlap_cells) != expect_overlap:
        return False
    if _has_truncation(grid.shape, seeds_by_color, counts) != expect_truncation:
        return False

    out = main(grid)
    for cell in overlap_cells:
        winning_color = max(coverage[cell], key=lambda color: counts[color])
        if int(out[cell]) != int(winning_color):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_colors = rng.randint(2, 4)
    counts_list = _sample_counts(rng, n_colors)
    shape = _grid_shape_for_counts(rng, counts_list)

    colors = rng.sample(PALETTE, n_colors)
    counts = {color: count for color, count in zip(colors, counts_list)}
    seeds_by_color = {color: [] for color in colors}
    placed: list[tuple[int, tuple[int, int], set[tuple[int, int]]]] = []

    expect_overlap = bool(rng.randrange(2))
    expect_truncation = bool(rng.randrange(2))

    if expect_overlap:
        ordered = sorted(colors, key=lambda color: counts[color])
        short_color = ordered[0]
        long_color = ordered[-1]
        if not _place_overlap_pair(rng, shape, placed, seeds_by_color, counts, long_color, short_color):
            return None

    if expect_truncation:
        available = [color for color in colors if len(seeds_by_color[color]) < counts[color]]
        if not available:
            return None
        trunc_color = max(available, key=lambda color: counts[color])
        candidates = _truncation_candidates(shape, counts[trunc_color])
        rng.shuffle(candidates)

        placed_truncation = False
        for seed in candidates:
            footprint = _can_place(seed, trunc_color, shape, placed, counts)
            if footprint is None:
                continue
            if len(footprint) == 4 * counts[trunc_color] + 1:
                continue
            placed.append((trunc_color, seed, footprint))
            seeds_by_color[trunc_color].append(seed)
            placed_truncation = True
            break
        if not placed_truncation:
            return None

    for color in sorted(colors, key=lambda value: counts[value], reverse=True):
        remaining = counts[color] - len(seeds_by_color[color])
        if remaining <= 0:
            continue

        candidates = _interior_candidates(shape, counts[color])
        if not candidates:
            return None

        for _ in range(remaining):
            rng.shuffle(candidates)
            placed_one = False
            for seed in candidates:
                footprint = _can_place(seed, color, shape, placed, counts)
                if footprint is None:
                    continue
                placed.append((color, seed, footprint))
                seeds_by_color[color].append(seed)
                placed_one = True
                break
            if not placed_one:
                return None

    grid = _build_grid(shape, seeds_by_color)
    if not _validate_scene(grid, seeds_by_color, counts, expect_overlap, expect_truncation):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
