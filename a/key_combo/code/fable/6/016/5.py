from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, get_bg


MAX_GRID_SIZE = 30
GEN_MAX_ATTEMPTS = 400


def _find_stripe(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one full-width stripe")
    row = int(candidates[0])
    return row, int(grid[row, 0])


def _seed_candidates(grid: np.ndarray, stripe_row: int, bg: int) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for col in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:stripe_row, col] != bg)
        if rows.size == 0:
            continue
        vals = grid[rows, col]
        if np.unique(vals).size != 1:
            continue
        top = int(rows[0])
        if np.array_equal(rows, np.arange(top, stripe_row)):
            candidates.append((col, int(vals[0]), top, stripe_row - top))
    return candidates


def _find_seed(grid: np.ndarray, stripe_row: int, bg: int) -> tuple[int, int, int]:
    candidates = _seed_candidates(grid, stripe_row, bg)
    if not candidates:
        raise ValueError("missing seed bar")

    color_counts = {
        int(color): int(np.count_nonzero(grid[:stripe_row] == color))
        for _, color, _, _ in candidates
    }
    scored = sorted(
        candidates,
        key=lambda item: (
            item[3] > 1,
            color_counts[item[1]] == item[3],
            item[3],
            -item[0],
        ),
        reverse=True,
    )
    seed_col, seed_color, top, height = scored[0]
    return int(seed_col), int(seed_color), int(height)


def _collect_dots(
    grid: np.ndarray,
    stripe_row: int,
    bg: int,
    seed_col: int,
) -> dict[int, int]:
    dots: dict[int, int] = {}
    for col in range(grid.shape[1]):
        if col == seed_col:
            continue
        rows = np.flatnonzero(grid[:stripe_row, col] != bg)
        if rows.size == 1:
            dots[int(col)] = int(rows[0])
    return dots


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Follow the staircase chain from the seed bar.

    Each bar height becomes the horizontal jump distance to the next sampled
    column. If that landing column holds a single dot, expand it into a full
    bar of the seed color; otherwise the chain stops and all untouched dots stay
    unchanged.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = int(get_bg(grid))
    stripe_row, _stripe_color = _find_stripe(grid, bg)
    seed_col, seed_color, current_height = _find_seed(grid, stripe_row, bg)
    dots = _collect_dots(grid, stripe_row, bg, seed_col)

    out = grid.copy()
    current_col = seed_col
    while True:
        landing_col = current_col + current_height
        if landing_col >= grid.shape[1] or landing_col not in dots:
            break
        dot_row = int(dots[landing_col])
        out[dot_row:stripe_row, landing_col] = seed_color
        current_col = landing_col
        current_height = stripe_row - dot_row

    return out


def _build_expected(
    grid: np.ndarray,
    stripe_row: int,
    seed_color: int,
    chain_dots: list[tuple[int, int]],
) -> np.ndarray:
    expected = grid.copy()
    for col, row in chain_dots:
        expected[row:stripe_row, col] = seed_color
    return expected


def _make_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    height = int(rng.integers(8, 19))
    width = int(rng.integers(10, MAX_GRID_SIZE + 1))
    stripe_row = height - 1

    palette = [int(v) for v in rng.choice(np.array(Color.ALL_COLORS, dtype=int), size=4, replace=False)]
    bg, stripe_color, seed_color, dot_color = palette

    grid = np.full((height, width), bg, dtype=int)
    grid[stripe_row, :] = stripe_color

    seed_col = int(rng.integers(0, max(1, width // 4 + 1)))
    max_seed_height = min(6, stripe_row, width - seed_col - 1)
    if max_seed_height < 2:
        raise ValueError("insufficient room for the seed bar")
    seed_height = int(rng.integers(2, max_seed_height + 1))
    grid[stripe_row - seed_height : stripe_row, seed_col] = seed_color

    current_col = seed_col
    current_height = seed_height
    chain_dots: list[tuple[int, int]] = []
    max_chain_len = int(rng.integers(1, 6))

    for step in range(max_chain_len):
        landing_col = current_col + current_height
        if landing_col >= width:
            break

        remaining = width - 1 - landing_col
        if step == 0:
            force_continue = True
        else:
            force_continue = remaining > 0 and rng.random() < 0.7
        if not force_continue:
            break

        if step < max_chain_len - 1 and remaining > 0 and rng.random() < 0.7:
            max_next_height = min(6, stripe_row, remaining)
        else:
            max_next_height = min(6, stripe_row)
        if max_next_height < 1:
            break

        next_height = int(rng.integers(1, max_next_height + 1))
        dot_row = stripe_row - next_height
        grid[dot_row, landing_col] = dot_color
        chain_dots.append((landing_col, dot_row))
        current_col = landing_col
        current_height = next_height

    if not chain_dots:
        raise ValueError("expected at least one chained dot")

    stop_col = current_col + current_height
    blocked_cols = {seed_col, *(col for col, _ in chain_dots)}
    if stop_col < width:
        blocked_cols.add(int(stop_col))

    free_cols = [col for col in range(width) if col not in blocked_cols]
    bystander_limit = min(len(free_cols), max(2, width // 3))
    bystander_count = int(rng.integers(0, bystander_limit + 1))
    if bystander_count:
        chosen = rng.choice(np.array(free_cols, dtype=int), size=bystander_count, replace=False)
        for col in map(int, np.atleast_1d(chosen)):
            row = int(rng.integers(0, stripe_row))
            grid[row, col] = dot_color

    expected = _build_expected(grid, stripe_row, seed_color, chain_dots)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            grid, expected = _make_instance(rng)
        except ValueError:
            continue
        if grid.ndim != 2 or expected.ndim != 2:
            continue
        if max(grid.shape) > MAX_GRID_SIZE:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid staircase-chain instance")
