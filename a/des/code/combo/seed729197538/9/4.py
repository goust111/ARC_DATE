from __future__ import annotations

import random

import numpy as np


BLACK = 0
INPUT_COLORS = (1, 2, 3)  # blue, red, green
BOUNDARY_COLOR_BY_DIFF = {1: 4, 2: 7, 3: 6}  # yellow, orange, magenta
GEN_RETRIES = 400
GRID_MIN = 9
GRID_MAX = 25
MAX_SEEDS_PER_COLOR = 4


def _seed_groups(grid: np.ndarray) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BLACK]
    assert len(colors) == 2, "expected exactly two seed colors"

    groups: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        groups[color] = [(int(r), int(c)) for r, c in coords]
        assert groups[color], "each seed color must appear at least once"
    return colors, groups


def _grown_regions(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    colors, groups = _seed_groups(grid)
    counts = {color: len(groups[color]) for color in colors}

    count_a = counts[colors[0]]
    count_b = counts[colors[1]]
    assert count_a != count_b, "seed counts must be different"
    dominant_color = colors[0] if count_a > count_b else colors[1]

    h, w = grid.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    dist_maps: dict[int, np.ndarray] = {}
    for color in colors:
        dist = np.full((h, w), h + w, dtype=int)
        for sr, sc in groups[color]:
            dist = np.minimum(dist, np.abs(rows - sr) + np.abs(cols - sc))
        dist_maps[color] = dist

    color_a, color_b = colors
    dist_a = dist_maps[color_a]
    dist_b = dist_maps[color_b]
    regions = np.full((h, w), dominant_color, dtype=int)
    regions[dist_a < dist_b] = color_a
    regions[dist_b < dist_a] = color_b
    return regions, counts


def _boundary_mask(regions: np.ndarray) -> np.ndarray:
    h, w = regions.shape
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if r + 1 < h and int(regions[r, c]) != int(regions[r + 1, c]):
                mask[r, c] = True
                mask[r + 1, c] = True
            if c + 1 < w and int(regions[r, c]) != int(regions[r, c + 1]):
                mask[r, c] = True
                mask[r, c + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed color synchronously, then recolor the interface by the
    difference in the two seed counts.
    """
    g = np.asarray(grid, dtype=int)
    regions, counts = _grown_regions(g)

    seed_counts = list(counts.values())
    diff = abs(seed_counts[0] - seed_counts[1])
    assert diff in BOUNDARY_COLOR_BY_DIFF, "seed-count difference must be 1, 2, or 3"

    out = regions.copy()
    out[_boundary_mask(regions)] = BOUNDARY_COLOR_BY_DIFF[diff]
    return out


def _sample_counts(rng: random.Random) -> tuple[int, int, int]:
    diff = rng.choice((1, 2, 3))
    smaller = rng.randint(1, MAX_SEEDS_PER_COLOR - diff)
    larger = smaller + diff
    if rng.random() < 0.5:
        return larger, smaller, diff
    return smaller, larger, diff


def _place_seed_positions(h: int, w: int, total: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)

    blocked = np.zeros((h, w), dtype=bool)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if blocked[r, c]:
            continue
        chosen.append((r, c))
        blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
        if len(chosen) == total:
            return chosen
    return None


def _is_interesting(grid: np.ndarray, colors: tuple[int, int], diff: int) -> bool:
    regions, _counts = _grown_regions(grid)
    boundary = _boundary_mask(regions)
    boundary_cells = int(np.count_nonzero(boundary))
    if boundary_cells < max(4, min(grid.shape) // 2):
        return False
    if boundary_cells > (grid.size * 2) // 5:
        return False
    if not bool(boundary[1:-1, 1:-1].any()):
        return False

    for color in colors:
        interior = int(np.count_nonzero((regions == color) & ~boundary))
        if interior < 2:
            return False

    output = regions.copy()
    output[boundary] = BOUNDARY_COLOR_BY_DIFF[diff]
    return set(map(int, np.unique(output))) == {colors[0], colors[1], BOUNDARY_COLOR_BY_DIFF[diff]}


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(GEN_RETRIES):
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)

        color_a, color_b = rng.sample(INPUT_COLORS, 2)
        count_a, count_b, diff = _sample_counts(rng)

        positions = _place_seed_positions(h, w, count_a + count_b, rng)
        if positions is None:
            continue
        rng.shuffle(positions)

        grid = np.zeros((h, w), dtype=int)
        for r, c in positions[:count_a]:
            grid[r, c] = color_a
        for r, c in positions[count_a:]:
            grid[r, c] = color_b

        if _is_interesting(grid, (color_a, color_b), diff):
            return grid

    raise RuntimeError("generate_input: exhausted attempts")
