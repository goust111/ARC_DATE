from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GEN_ATTEMPTS = 400


def _seed_catalog() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for mask in range(1, 1 << 4):
        if not 1 <= int(mask.bit_count()) <= 3:
            continue
        seed = np.zeros((2, 2), dtype=int)
        for bit in range(4):
            if (mask >> bit) & 1:
                seed[bit // 2, bit % 2] = BLUE
        patterns.append(seed)
    return patterns


SEED_PATTERNS = _seed_catalog()


def _full_line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=bool)
    return np.kron(mask.astype(bool), block)


def _quadrants(shape: tuple[int, int], split_r: int, split_c: int) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    return [
        (0, split_r, 0, split_c),
        (0, split_r, split_c + 1, w),
        (split_r + 1, h, 0, split_c),
        (split_r + 1, h, split_c + 1, w),
    ]


def _transform_region(region: np.ndarray) -> np.ndarray:
    out = np.zeros_like(region)
    seed_mask = region[:2, :2] == BLUE
    factor = int(np.count_nonzero(region == RED)) + 1
    scaled = _scale_mask(seed_mask, factor)
    ph = min(region.shape[0], scaled.shape[0])
    pw = min(region.shape[1], scaled.shape[1])
    patch = out[:ph, :pw]
    patch[scaled[:ph, :pw]] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge each quadrant's 2x2 blue seed by a factor encoded by that quadrant's red marker count.

    Input structure:
    - The grid is mostly black, with one full grey horizontal separator and one full grey vertical separator.
    - The separators split the canvas into four rectangular regions.
    - In each region, the top-left 2x2 area contains a blue seed pattern, and the rest of the region may contain red marker pixels on black background.

    Rule:
    - Detect the grey separator row and column to recover the four regions.
    - For each region, read the blue seed from its top-left 2x2 area and count the region's red markers.
    - Enlarge the seed by factor `(marker_count + 1)` using pixel replication, place the enlarged blue pattern at the region's top-left corner, and clear all red markers.
    - Keep the grey separators unchanged and return the full grid.

    Key insight:
    - The red pixels are not part of the final drawing; they only encode each region's independent enlargement factor for the same local seed-to-macro expansion rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    split_rows = _full_line_indices(g, GREY, axis=0)
    split_cols = _full_line_indices(g, GREY, axis=1)
    assert split_rows and split_cols, "expected one grey separator row and one grey separator column"

    split_r = int(split_rows[0])
    split_c = int(split_cols[0])

    out = np.asarray(g, dtype=int).copy()
    for r0, r1, c0, c1 in _quadrants(g.shape, split_r, split_c):
        region = g[r0:r1, c0:c1]
        assert region.shape[0] >= 2 and region.shape[1] >= 2
        out[r0:r1, c0:c1] = _transform_region(region)

    out[split_r, :] = GREY
    out[:, split_c] = GREY
    return out


def _allowed_marker_counts(region_h: int, region_w: int) -> list[int]:
    limit = min(region_h, region_w)
    return [n for n in (1, 2, 3) if 2 * (n + 1) <= limit]


def _choose_marker_counts(options: list[list[int]]) -> list[int] | None:
    if not options or any(not opt for opt in options):
        return None

    for _ in range(120):
        counts = [random.choice(opt) for opt in options]
        if len(set(counts)) < 2:
            continue
        if any(3 in opt for opt in options) and 3 not in counts and random.random() < 0.55:
            continue
        if 2 not in counts and all(max(opt) >= 2 for opt in options):
            continue
        return counts
    return None


def _marker_pool(region_h: int, region_w: int, count: int) -> list[tuple[int, int]]:
    scaled_side = 2 * (count + 1)
    preferred = [
        (r, c)
        for r in range(region_h)
        for c in range(region_w)
        if not (r < 2 and c < 2) and not (r < scaled_side and c < scaled_side)
    ]
    fallback = [
        (r, c)
        for r in range(region_h)
        for c in range(region_w)
        if not (r < 2 and c < 2)
    ]
    if len(preferred) >= count and random.random() < 0.8:
        random.shuffle(preferred)
        return preferred
    random.shuffle(fallback)
    return fallback


def _build_region(region_h: int, region_w: int, seed: np.ndarray, count: int) -> np.ndarray | None:
    region = np.zeros((region_h, region_w), dtype=int)
    region[:2, :2] = seed

    pool = _marker_pool(region_h, region_w, count)
    if len(pool) < count:
        return None

    for r, c in pool[:count]:
        region[r, c] = RED
    return region


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(6, 11)
    bottom_h = random.randint(6, 11)
    left_w = random.randint(6, 11)
    right_w = random.randint(6, 11)

    grid_h = top_h + bottom_h + 1
    grid_w = left_w + right_w + 1
    if max(grid_h, grid_w) > 30:
        return None

    region_shapes = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, left_w),
        (bottom_h, right_w),
    ]
    marker_options = [_allowed_marker_counts(h, w) for h, w in region_shapes]
    marker_counts = _choose_marker_counts(marker_options)
    if marker_counts is None:
        return None

    seeds = random.sample(SEED_PATTERNS, 4)
    if sum(int(np.count_nonzero(seed)) == 1 for seed in seeds) == 4:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    split_r = top_h
    split_c = left_w
    grid[split_r, :] = GREY
    grid[:, split_c] = GREY

    expected = np.zeros_like(grid)
    expected[split_r, :] = GREY
    expected[:, split_c] = GREY

    for (r0, r1, c0, c1), seed, count in zip(_quadrants(grid.shape, split_r, split_c), seeds, marker_counts):
        region = _build_region(r1 - r0, c1 - c0, seed, count)
        if region is None:
            return None
        grid[r0:r1, c0:c1] = region
        expected[r0:r1, c0:c1] = _transform_region(region)

    if len(set(marker_counts)) < 2:
        return None
    if not np.any(grid == RED):
        return None
    if not np.any(expected == BLUE):
        return None
    if int(np.count_nonzero(expected == BLUE)) <= int(np.count_nonzero(grid == BLUE)):
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
