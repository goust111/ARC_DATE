from __future__ import annotations

import random
from collections import Counter

import numpy as np


ACTIVE_PALETTE = (1, 2, 3, 6)
GEN_ATTEMPTS = 400


def _trail_coverage(shape: tuple[int, int], seed: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = seed
    covered: set[tuple[int, int]] = set()
    starts = (
        (r, c),
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
    )
    for sr, sc in starts:
        if 0 <= sr < h and 0 <= sc < w:
            for rr in range(sr, h, 2):
                covered.add((rr, sc))
    return covered


def _cross_count(shape: tuple[int, int], seed_a: tuple[int, int], seed_b: tuple[int, int]) -> int:
    return len(_trail_coverage(shape, seed_a) & _trail_coverage(shape, seed_b))


def _pack_squares(specs: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: (item[1], item[0]))
    max_side = max(side for _color, side in ordered)
    width = sum(side for _color, side in ordered) + len(ordered) - 1
    out = np.zeros((max_side, width), dtype=int)

    col = 0
    for index, (color, side) in enumerate(ordered):
        out[max_side - side :, col : col + side] = color
        col += side
        if index != len(ordered) - 1:
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only colors with exactly two seeds, count same-color trail intersections, and repack them as ordered squares.

    Input structure:
    - The grid has a black background and 6-8 isolated seed pixels in the first three rows.
    - Seed colors come from a small foreground palette; some colors appear once and some appear exactly twice.
    - Only the double-seed colors can create same-color overlaps in the downward period-2 trail pattern.

    Rule:
    - Count how many seed pixels each non-black color has and keep only the colors that appear exactly twice.
    - For each kept color, grow the two seed-based trail systems using the five diagonal-offset starts and a row step of 2.
    - Count the cells covered by both same-color trail systems, then output one solid square of that side length in the same color.
    - Pack the squares left-to-right in increasing side-length order, bottom-aligned, with one black column between neighbors.

    Key insight:
    - The important quantity is not the visible seed placement itself but the number of self-intersections produced by each two-seed color's periodic trail pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g.flat))
    active_colors = [color for color, count in counts.items() if color != 0 and count == 2]
    assert active_colors, "expected at least one double-seed color"

    specs: list[tuple[int, int]] = []
    for color in sorted(active_colors):
        seeds = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert len(seeds) == 2
        side = _cross_count(g.shape, seeds[0], seeds[1])
        assert side > 0, "double-seed colors must create at least one self-intersection"
        specs.append((int(color), int(side)))

    return _pack_squares(specs)


def _choose_profile() -> tuple[int, list[int], tuple[int, int]]:
    roll = random.random()
    if roll < 0.22:
        return 7, [3, 4], (12, 18)
    if roll < 0.44:
        return 9, [4, 5], (12, 18)
    if roll < 0.76:
        active_count = 2
    elif roll < 0.94:
        active_count = 3
    else:
        active_count = 4

    counts = sorted(random.sample([2, 3, 4, 5], active_count))
    width_min = {2: 12, 3: 15, 4: 16}[active_count]
    return 5, counts, (width_min, min(24, width_min + 6))


def _pair_candidates(height: int, width: int, wanted_counts: set[int]) -> dict[int, list[dict[str, object]]]:
    buckets = {count: [] for count in wanted_counts}
    for r1 in range(3):
        for c1 in range(width):
            for r2 in range(3):
                for c2 in range(width):
                    if (r1, c1) >= (r2, c2) or c1 == c2:
                        continue
                    seeds = ((r1, c1), (r2, c2))
                    count = _cross_count((height, width), seeds[0], seeds[1])
                    if count not in buckets:
                        continue
                    trail = _trail_coverage((height, width), seeds[0]) | _trail_coverage((height, width), seeds[1])
                    buckets[count].append({"seeds": seeds, "trail": trail})
    return buckets


def _seed_gap_ok(seed_a: tuple[int, int], seed_b: tuple[int, int]) -> bool:
    return max(abs(seed_a[0] - seed_b[0]), abs(seed_a[1] - seed_b[1])) > 1


def _search_active_pairs(height: int, width: int, counts: list[int]) -> list[dict[str, object]] | None:
    candidates = _pair_candidates(height, width, set(counts))
    if any(not candidates[count] for count in counts):
        return None

    order = sorted(counts, key=lambda count: len(candidates[count]))
    chosen: list[dict[str, object]] = []
    used_seeds: list[tuple[int, int]] = []
    used_trails: list[set[tuple[int, int]]] = []

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True

        count = order[index]
        options = candidates[count][:]
        random.shuffle(options)
        for option in options:
            pair_seeds = option["seeds"]
            trail = option["trail"]
            assert isinstance(pair_seeds, tuple)
            assert isinstance(trail, set)

            if any(not _seed_gap_ok(seed, used) for seed in pair_seeds for used in used_seeds):
                continue
            if any(trail & prev_trail for prev_trail in used_trails):
                continue

            chosen.append({"count": count, "seeds": pair_seeds, "trail": trail})
            used_seeds.extend(pair_seeds)
            used_trails.append(trail)
            if backtrack(index + 1):
                return True
            chosen.pop()
            used_trails.pop()
            used_seeds.pop()
            used_seeds.pop()
        return False

    if backtrack(0):
        return chosen
    return None


def _singleton_positions(
    height: int,
    width: int,
    used_seeds: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    slots = [
        (r, c)
        for r in range(min(3, height))
        for c in range(width)
        if all(_seed_gap_ok((r, c), used) for used in used_seeds)
    ]
    random.shuffle(slots)
    if len(slots) < count:
        return None
    return slots[:count]


def _build_expected(active_specs: list[tuple[int, int]]) -> np.ndarray:
    return _pack_squares(active_specs)


def _gen_attempt() -> np.ndarray | None:
    height, wanted_counts, width_range = _choose_profile()
    width = random.randint(*width_range)

    pairs = _search_active_pairs(height, width, wanted_counts)
    if pairs is None:
        return None

    colors = list(ACTIVE_PALETTE)
    random.shuffle(colors)
    active_colors = colors[: len(wanted_counts)]
    singleton_colors = colors[len(wanted_counts) :]

    used_seeds = [seed for pair in pairs for seed in pair["seeds"]]
    singleton_slots = _singleton_positions(height, width, used_seeds, len(singleton_colors))
    if singleton_slots is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    active_specs: list[tuple[int, int]] = []

    random.shuffle(pairs)
    for color, pair in zip(active_colors, pairs):
        for r, c in pair["seeds"]:
            grid[r, c] = color
        active_specs.append((int(color), int(pair["count"])))

    for color, (r, c) in zip(singleton_colors, singleton_slots):
        grid[r, c] = color

    if any(int(np.count_nonzero(grid == color)) != 2 for color in active_colors):
        return None
    if any(int(np.count_nonzero(grid == color)) != 1 for color in singleton_colors):
        return None

    expected = _build_expected(active_specs)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
