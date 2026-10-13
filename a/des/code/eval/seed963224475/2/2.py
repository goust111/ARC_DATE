from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
WALL = 3
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)

ROW_BAND_OPTIONS = (2, 3)
COL_BAND_OPTIONS = (2, 3)
BAND_SIZE_CHOICES = (9, 11, 13)
MAX_GENERATION_ATTEMPTS = 200


def _find_walls(grid: np.ndarray) -> tuple[int | None, list[int], list[int]]:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    candidates = [color for color in row_colors + col_colors if color != BLACK]
    if not candidates:
        return None, [], []
    wall_color = Counter(candidates).most_common(1)[0][0]
    wall_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == wall_color)]
    wall_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == wall_color)]
    return wall_color, wall_rows, wall_cols


def _region_ranges(length: int, walls: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(walls) + [length]
    ranges: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = left + 1
        end = right - 1
        if start <= end:
            ranges.append((start, end))
    return ranges


def _extract_seed_info(region: np.ndarray, wall_color: int | None) -> tuple[int, int]:
    mask = region != BLACK
    if wall_color is not None:
        mask &= region != wall_color
    seeds = region[mask]
    if seeds.size == 0:
        return BLACK, 0
    color, count = Counter(map(int, seeds.tolist())).most_common(1)[0]
    return int(color), int(count)


def _paint_region(height: int, width: int, color: int, ring_count: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    if ring_count <= 0:
        return patch

    row_depth = np.minimum(np.arange(height), np.arange(height - 1, -1, -1))[:, None]
    col_depth = np.minimum(np.arange(width), np.arange(width - 1, -1, -1))[None, :]
    depth = np.minimum(row_depth, col_depth)
    color_mask = (depth % 2 == 0) & (depth // 2 < ring_count)
    patch[color_mask] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-delimited rectangle with spaced concentric rings.

    Every region contains 1 to 4 same-colored seed pixels. The seed color sets the
    ring color, and the seed count sets how many odd-numbered layers are filled
    from the outside inward. Even-numbered layers stay black.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    height, width = g.shape
    wall_color, wall_rows, wall_cols = _find_walls(g)
    row_ranges = _region_ranges(height, wall_rows)
    col_ranges = _region_ranges(width, wall_cols)

    out = g.copy()
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            color, ring_count = _extract_seed_info(region, wall_color)
            out[r0 : r1 + 1, c0 : c1 + 1] = _paint_region(
                r1 - r0 + 1,
                c1 - c0 + 1,
                color,
                ring_count,
            )

    if wall_color is not None:
        if wall_rows:
            out[wall_rows, :] = wall_color
        if wall_cols:
            out[:, wall_cols] = wall_color

    return out


def _band_layout(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges: list[tuple[int, int]] = []
    walls: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        ranges.append((cursor, cursor + size - 1))
        cursor += size
        if idx != len(sizes) - 1:
            walls.append(cursor)
            cursor += 1
    return ranges, walls, cursor


def _sample_band_sizes(count: int, rng: random.Random) -> list[int]:
    sizes = [rng.choice(BAND_SIZE_CHOICES) for _ in range(count)]
    sizes[rng.randrange(count)] = 13
    return sizes


def _max_ring_count(region_h: int, region_w: int) -> int:
    return min(4, (min(region_h, region_w) + 3) // 4)


def _assign_ring_counts(regions: list[tuple[int, int, int, int]], rng: random.Random) -> list[int] | None:
    max_counts = [_max_ring_count(r1 - r0 + 1, c1 - c0 + 1) for r0, r1, c0, c1 in regions]
    if len(regions) < 4 or max(max_counts, default=0) < 4:
        return None

    counts: list[int | None] = [None] * len(regions)
    remaining = list(range(len(regions)))

    chosen4 = rng.choice([idx for idx in remaining if max_counts[idx] >= 4])
    counts[chosen4] = 4
    remaining.remove(chosen4)

    for target in (3, 2, 1):
        eligible = [idx for idx in remaining if max_counts[idx] >= target]
        if not eligible:
            return None
        chosen = rng.choice(eligible)
        counts[chosen] = target
        remaining.remove(chosen)

    for idx in remaining:
        counts[idx] = rng.randint(1, max_counts[idx])

    return [int(count) for count in counts]


def _choose_region_colors(region_count: int, rng: random.Random) -> list[int]:
    pool_size = min(len(SEED_COLORS), rng.choice((2, 3, 4, 5)))
    color_pool = rng.sample(SEED_COLORS, pool_size)
    colors = [rng.choice(color_pool) for _ in range(region_count)]
    if region_count > 1 and len(set(colors)) == 1:
        alternatives = [color for color in SEED_COLORS if color != colors[0]]
        colors[rng.randrange(region_count)] = rng.choice(alternatives)
    return colors


def _sample_seed_positions(
    region_h: int,
    region_w: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(region_h) for c in range(region_w)]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - pr), abs(c - pc)) >= 2 for pr, pc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    raise RuntimeError("failed to place non-touching seeds")


def _build_expected(
    shape: tuple[int, int],
    wall_rows: list[int],
    wall_cols: list[int],
    regions: list[tuple[int, int, int, int]],
    ring_counts: list[int],
    region_colors: list[int],
) -> np.ndarray:
    expected = np.zeros(shape, dtype=int)
    if wall_rows:
        expected[wall_rows, :] = WALL
    if wall_cols:
        expected[:, wall_cols] = WALL

    for (r0, r1, c0, c1), ring_count, color in zip(regions, ring_counts, region_colors):
        expected[r0 : r1 + 1, c0 : c1 + 1] = _paint_region(
            r1 - r0 + 1,
            c1 - c0 + 1,
            color,
            ring_count,
        )
    return expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_band_count = rng.choice(ROW_BAND_OPTIONS)
    col_band_count = rng.choice(COL_BAND_OPTIONS)

    row_sizes = _sample_band_sizes(row_band_count, rng)
    col_sizes = _sample_band_sizes(col_band_count, rng)
    row_ranges, wall_rows, height = _band_layout(row_sizes)
    col_ranges, wall_cols, width = _band_layout(col_sizes)

    regions = [(r0, r1, c0, c1) for r0, r1 in row_ranges for c0, c1 in col_ranges]
    ring_counts = _assign_ring_counts(regions, rng)
    if ring_counts is None:
        return None

    region_colors = _choose_region_colors(len(regions), rng)
    grid = np.zeros((height, width), dtype=int)
    if wall_rows:
        grid[wall_rows, :] = WALL
    if wall_cols:
        grid[:, wall_cols] = WALL

    for (r0, r1, c0, c1), ring_count, color in zip(regions, ring_counts, region_colors):
        region_h = r1 - r0 + 1
        region_w = c1 - c0 + 1
        for dr, dc in _sample_seed_positions(region_h, region_w, ring_count, rng):
            grid[r0 + dr, c0 + dc] = color

    expected = _build_expected(
        grid.shape,
        wall_rows,
        wall_cols,
        regions,
        ring_counts,
        region_colors,
    )
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
