from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(color for color in range(10) if color not in (BLACK, GREY))

GRID_MIN = 16
GRID_MAX = 28
TARGET_REGION_MIN = 3
TARGET_REGION_MAX = 7
MIN_REGION_AREA = 8
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    return None


def _candidate_paths(
    start: tuple[int, int], end: tuple[int, int], rng: random.Random
) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = []
    straight = _segment(start, end)
    if straight is not None:
        paths.append(straight)

    for pivot in ((start[0], end[1]), (end[0], start[1])):
        first = _segment(start, pivot)
        second = _segment(pivot, end)
        if first is None or second is None:
            continue
        paths.append(first + second[1:])

    rng.shuffle(paths)
    unique: list[list[tuple[int, int]]] = []
    seen = set()
    for path in paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _seed_sites(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(comp)
    return [(r, c) for r, c in comp if (r - 1, c) in cells]


def _boundary_buckets(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    h, w = grid.shape
    buckets = {"top": [], "bottom": [], "left": [], "right": []}
    for r, c in comp:
        if r > 0 and int(grid[r - 1, c]) == GREY:
            buckets["top"].append((r, c))
        if r + 1 < h and int(grid[r + 1, c]) == GREY:
            buckets["bottom"].append((r, c))
        if c > 0 and int(grid[r, c - 1]) == GREY:
            buckets["left"].append((r, c))
        if c + 1 < w and int(grid[r, c + 1]) == GREY:
            buckets["right"].append((r, c))
    return buckets


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    grid[rows, cols] = GREY


def _try_add_wall(grid: np.ndarray, rng: random.Random) -> bool:
    black_comps = _components(grid == BLACK)
    rng.shuffle(black_comps)
    for comp in black_comps:
        if len(comp) < 2 * MIN_REGION_AREA + 3:
            continue

        comp_set = set(comp)
        buckets = _boundary_buckets(grid, comp)
        side_pairs = list(itertools.combinations([side for side, cells in buckets.items() if cells], 2))
        rng.shuffle(side_pairs)
        if not side_pairs:
            continue

        region_mask = np.zeros_like(grid, dtype=bool)
        for r, c in comp:
            region_mask[r, c] = True

        for side_a, side_b in side_pairs:
            starts = buckets[side_a][:]
            ends = buckets[side_b][:]
            rng.shuffle(starts)
            rng.shuffle(ends)
            for start in starts[:10]:
                for end in ends[:10]:
                    if start == end:
                        continue
                    for path in _candidate_paths(start, end, rng):
                        if len(path) < 3:
                            continue
                        if any(cell not in comp_set for cell in path):
                            continue
                        if any(int(grid[r, c]) != BLACK for r, c in path):
                            continue

                        trial = grid.copy()
                        _paint_path(trial, path)
                        split_parts = _components(region_mask & (trial == BLACK))
                        if len(split_parts) != 2:
                            continue
                        if any(len(part) < MIN_REGION_AREA for part in split_parts):
                            continue
                        if any(not _seed_sites(part) for part in split_parts):
                            continue

                        _paint_path(grid, path)
                        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded enclosed region with the color shown by the band directly above the seed.

    Input structure:
    - Black background cells are partitioned by one-pixel-thick grey walls into enclosed regions.
    - Some regions contain exactly one non-black, non-grey seed pixel.
    - The cell immediately above each seed is a non-black, non-grey band marker with a different color, inside the same region.

    Rule:
    - Treat every maximal non-grey 4-connected component as one enclosed region.
    - If a region contains the vertical band/seed pair, read the upper marker color.
    - Recolor every non-grey pixel in that region, including the seed and the marker, to that upper marker color.
    - Regions without such a seed pair stay unchanged, and all grey wall pixels remain grey.

    Key insight:
    - The seed only selects which enclosed region should grow; the actual fill color is the marker directly above it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g != GREY):
        colored = [(r, c) for r, c in comp if int(g[r, c]) != BLACK]
        if not colored:
            continue

        assert len(colored) == 2, "seeded region must contain exactly two colored cells"
        colored.sort()
        upper, lower = colored
        assert lower[0] == upper[0] + 1 and lower[1] == upper[1], "marker must sit directly above seed"
        assert int(g[upper]) != int(g[lower]), "marker and seed colors must differ"

        fill_color = int(g[upper])
        for r, c in comp:
            out[r, c] = fill_color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    target_regions = rng.randint(TARGET_REGION_MIN, TARGET_REGION_MAX)
    split_budget = 10 * target_regions
    while len(_components(grid == BLACK)) < target_regions and split_budget > 0:
        split_budget -= 1
        if not _try_add_wall(grid, rng):
            break

    regions = _components(grid == BLACK)
    if len(regions) < 2:
        return None

    eligible = [comp for comp in regions if _seed_sites(comp)]
    if len(eligible) < 2:
        return None

    rng.shuffle(eligible)
    seed_count = rng.randint(1, len(eligible) - 1)
    seeded_regions = eligible[:seed_count]

    for comp in seeded_regions:
        seed_row, seed_col = rng.choice(_seed_sites(comp))
        band_color = rng.choice(COLOR_POOL)
        seed_choices = [color for color in COLOR_POOL if color != band_color]
        seed_color = rng.choice(seed_choices)
        grid[seed_row - 1, seed_col] = band_color
        grid[seed_row, seed_col] = seed_color

    colored_total = int(np.count_nonzero((grid != BLACK) & (grid != GREY)))
    if colored_total != 2 * seed_count:
        return None
    if seed_count == len(regions):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
