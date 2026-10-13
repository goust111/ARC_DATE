from __future__ import annotations

import random

import numpy as np


SEED_PALETTE = (2, 1, 3, 4, 7)
BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 2000


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BACKGROUND)
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert 3 <= len(seeds) <= 5, "expected 3 to 5 seed pixels"
    colors = [color for _r, _c, color in seeds]
    assert len(set(colors)) == len(colors), "seed colors must be distinct"
    return seeds


def _sorted_seed_indices(seeds: list[tuple[int, int, int]]) -> list[int]:
    return sorted(range(len(seeds)), key=lambda idx: (seeds[idx][1], seeds[idx][0]))


def _territory_counts_vectorized(
    shape: tuple[int, int], seeds: list[tuple[int, int, int]]
) -> tuple[list[int], np.ndarray]:
    h, w = map(int, shape)
    counts = np.ones(len(seeds), dtype=int)

    rr, cc = np.indices((h, w))
    seed_rows = np.array([r for r, _c, _color in seeds], dtype=int)[:, None, None]
    seed_cols = np.array([c for _r, c, _color in seeds], dtype=int)[:, None, None]
    dist = np.abs(rr[None, :, :] - seed_rows) + np.abs(cc[None, :, :] - seed_cols)

    seed_mask = np.zeros((h, w), dtype=bool)
    for r, c, _color in seeds:
        seed_mask[r, c] = True
    bg_mask = ~seed_mask

    min_dist = dist.min(axis=0)
    winners = dist == min_dist[None, :, :]
    winner_count = winners.sum(axis=0)
    contested = bg_mask & (winner_count > 1)

    for idx in range(len(seeds)):
        counts[idx] += int(np.count_nonzero(bg_mask & winners[idx] & (winner_count == 1)))

    return counts.tolist(), contested


def _territory_counts_naive(
    shape: tuple[int, int], seeds: list[tuple[int, int, int]]
) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = map(int, shape)
    counts = [1] * len(seeds)
    contested: set[tuple[int, int]] = set()
    seed_cells = {(r, c) for r, c, _color in seeds}

    for r in range(h):
        for c in range(w):
            if (r, c) in seed_cells:
                continue
            distances = [abs(r - sr) + abs(c - sc) for sr, sc, _color in seeds]
            best = min(distances)
            winners = [idx for idx, dist in enumerate(distances) if dist == best]
            if len(winners) == 1:
                counts[winners[0]] += 1
            else:
                contested.add((r, c))

    return counts, contested


def _render_histogram(
    counts: list[int], seeds: list[tuple[int, int, int]], order: list[int]
) -> np.ndarray:
    height = max(counts)
    width = len(seeds)
    out = np.zeros((height, width), dtype=int)
    for col, seed_idx in enumerate(order):
        color = seeds[seed_idx][2]
        count = counts[seed_idx]
        out[height - count :, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the background by Manhattan-distance competition and encode each seed's territory as a histogram bar.

    Input structure:
    - A black grid contains 3 to 5 single-pixel seeds, each in a distinct foreground color.
    - Seeds are the only foreground cells; every other cell is background.
    - Each seed color identifies exactly one source location.

    Rule:
    - For every background cell, compare its Manhattan distance to all seeds.
    - Assign the cell to the unique nearest seed; if the nearest distance is tied, leave that cell unclaimed.
    - Each seed's territory area equals its claimed background cells plus the seed pixel itself.
    - Sort seeds by input column from left to right, breaking equal columns by row from top to bottom, then draw one bottom-aligned histogram column per seed with height equal to its territory area.

    Key insight:
    - The output depends on a full multi-seed Manhattan Voronoi competition with ties discarded, and only the final territory sizes survive as an ordered bar chart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.count_nonzero(g == BACKGROUND) > 0, "expected background cells"

    seeds = _extract_seeds(g)
    counts, _contested = _territory_counts_vectorized(g.shape, seeds)
    return _render_histogram(counts, seeds, _sorted_seed_indices(seeds))


def _valid_seed_spacing(positions: list[tuple[int, int]]) -> bool:
    for idx, (r1, c1) in enumerate(positions):
        for r2, c2 in positions[idx + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) < 2:
                return False
            if abs(r1 - r2) + abs(c1 - c2) < 3:
                return False
    return True


def _sample_positions(
    h: int, w: int, n_seeds: int, sort_mode: str
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(400):
        positions = random.sample(cells, n_seeds)
        if not _valid_seed_spacing(positions):
            continue

        distinct_cols = len({c for _r, c in positions}) == len(positions)
        if sort_mode == "same_col" and distinct_cols:
            continue
        if sort_mode == "distinct_cols" and not distinct_cols:
            continue
        return positions
    return None


def _semantic_checks(
    grid: np.ndarray, seeds: list[tuple[int, int, int]], sort_mode: str
) -> bool:
    counts_fast, contested_mask = _territory_counts_vectorized(grid.shape, seeds)
    counts_slow, contested_cells = _territory_counts_naive(grid.shape, seeds)
    if counts_fast != counts_slow:
        return False
    if int(np.count_nonzero(contested_mask)) != len(contested_cells):
        return False
    if len(contested_cells) == 0:
        return False
    if min(counts_fast) <= 1:
        return False
    if len(set(counts_fast)) == 1:
        return False
    if max(counts_fast) > 30:
        return False

    positions = [(r, c) for r, c, _color in seeds]
    if not _valid_seed_spacing(positions):
        return False

    distinct_cols = len({c for _r, c in positions}) == len(positions)
    if sort_mode == "same_col" and distinct_cols:
        return False
    if sort_mode == "distinct_cols" and not distinct_cols:
        return False

    order = _sorted_seed_indices(seeds)
    output = main(grid)
    expected = _render_histogram(counts_slow, seeds, order)
    if not np.array_equal(output, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    n_seeds = random.randint(3, min(5, h))
    sort_mode = random.choices(
        ["same_col", "distinct_cols", "free"], weights=[4, 3, 3], k=1
    )[0]
    if sort_mode == "distinct_cols" and w < n_seeds:
        sort_mode = "free"

    positions = _sample_positions(h, w, n_seeds, sort_mode)
    if positions is None:
        return None

    colors = random.sample(SEED_PALETTE, n_seeds)
    grid = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(positions, colors):
        grid[r, c] = color

    seeds = _extract_seeds(grid)
    if not _semantic_checks(grid, seeds, sort_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid territory-competition input")
