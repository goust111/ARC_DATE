from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300

HEIGHT_RANGE = (8, 18)
WIDTH_RANGE = (12, 24)
MIN_MARGIN = 3
MAX_SEEDS = 6


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _find_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for c in range(grid.shape[1]):
        column = grid[:, c]
        if np.all(column == column[0]) and int(column[0]) != bg:
            candidates.append((c, int(column[0])))
    assert len(candidates) == 1, "expected exactly one full-height monochrome separator column"
    return candidates[0]


def _paint_from_seed(out: np.ndarray, row: int, seed_col: int, sep_col: int, color: int) -> None:
    if seed_col < sep_col:
        period = sep_col - seed_col
        out[row, seed_col:sep_col] = color
        for c in range(sep_col + period, out.shape[1], period):
            out[row, c] = color
    else:
        period = seed_col - sep_col
        out[row, sep_col + 1 : seed_col + 1] = color
        for c in range(sep_col - period, -1, -period):
            out[row, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed pixel into a solid ray up to the separator, then continue as a periodic dashed ray beyond it.

    Input structure:
    - The background is black.
    - Exactly one non-black vertical separator column spans the full grid height.
    - One side of the separator contains isolated seed pixels, while the opposite side is empty.

    Rule:
    - For each seed pixel, draw a horizontal solid segment of the same color from the seed to the cell adjacent to the separator.
    - Let k be that solid segment length, counting the seed cell itself.
    - Beyond the separator, keep extending on the same row but only paint cells at distances k, 2k, 3k, ... from the separator.

    Key insight:
    - The pre-separator solid length is measured separately for each seed and becomes that seed's post-separator dash period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    sep_col, sep_color = _find_separator_column(g, bg)
    left_mask = g[:, :sep_col] != bg
    right_mask = g[:, sep_col + 1 :] != bg
    left_count = int(np.sum(left_mask))
    right_count = int(np.sum(right_mask))
    assert (left_count == 0) != (right_count == 0), "expected seeds on exactly one side of the separator"

    seeds_on_left = left_count > 0
    seed_coords = np.argwhere(left_mask if seeds_on_left else right_mask)
    if not seeds_on_left:
        seed_coords[:, 1] += sep_col + 1

    rows = set()
    out = g.copy()
    for r, c in seed_coords:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        assert color != sep_color, "seed color must differ from separator color"
        assert r not in rows, "expected at most one seed per row"
        rows.add(r)
        _paint_from_seed(out, r, c, sep_col, color)

    return out


def _choose_spaced_rows(rng: np.random.Generator, height: int, count: int) -> list[int]:
    base = np.sort(rng.choice(height - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _oracle_output(
    grid: np.ndarray,
    sep_col: int,
    seeds_on_left: bool,
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for row, distance, color in seeds:
        seed_col = sep_col - distance if seeds_on_left else sep_col + distance
        _paint_from_seed(out, row, seed_col, sep_col, color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    sep_col = int(rng.integers(MIN_MARGIN, width - MIN_MARGIN))
    seeds_on_left = bool(rng.integers(0, 2))

    seed_side_width = sep_col if seeds_on_left else width - sep_col - 1
    blank_side_width = width - sep_col - 1 if seeds_on_left else sep_col
    max_distance = min(seed_side_width, blank_side_width)
    max_count = min(MAX_SEEDS, 8, max_distance, (height + 1) // 2)
    if max_count < 2:
        return None

    seed_count = int(rng.integers(2, max_count + 1))
    distances = list(map(int, rng.choice(np.arange(1, max_distance + 1), size=seed_count, replace=False)))
    rows = _choose_spaced_rows(rng, height, seed_count)

    sep_color = int(rng.choice(np.array(COLORS, dtype=int)))
    seed_palette = [c for c in COLORS if c != sep_color]
    seed_colors = list(map(int, rng.choice(np.array(seed_palette, dtype=int), size=seed_count, replace=False)))

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = sep_color

    seeds: list[tuple[int, int, int]] = []
    for row, distance, color in zip(rows, distances, seed_colors):
        seed_col = sep_col - distance if seeds_on_left else sep_col + distance
        grid[row, seed_col] = color
        seeds.append((int(row), int(distance), int(color)))

    expected = _oracle_output(grid, sep_col, seeds_on_left, seeds)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
