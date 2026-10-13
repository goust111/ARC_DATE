from __future__ import annotations

from collections import Counter
import random

import numpy as np


BORDER_COLOR = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GRID_HEIGHT_RANGE = (6, 14)
GRID_WIDTH_RANGE = (6, 16)
COLOR_COUNT_CHOICES = (2, 3, 4)
GENERATION_RETRIES = 400


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0],
            grid[1:-1, -1],
        ]
    )
    return Counter(map(int, border)).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor interior stripes using the correspondence encoded on the top and left border.

    Read the non-border-color markers on the top edge from left to right and the
    non-border-color markers on the left edge from top to bottom. Pair those
    sequences positionally to form a color mapping, then apply that mapping to the
    interior while leaving the full border unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "expected a bordered grid"

    border_color = _border_background(g)
    top_seq = [int(g[0, c]) for c in range(1, width - 1) if int(g[0, c]) != border_color]
    left_seq = [int(g[r, 0]) for r in range(1, height - 1) if int(g[r, 0]) != border_color]

    assert len(top_seq) == len(left_seq) >= 1, "top and left marker counts must match"
    assert len(set(top_seq)) == len(top_seq), "top markers must be unique"
    assert len(set(left_seq)) == len(left_seq), "left markers must be unique"
    assert set(top_seq) == set(left_seq), "top and left markers must use the same color set"

    mapping = dict(zip(top_seq, left_seq))
    out = g.copy()
    interior = g[1:-1, 1:-1]
    recolored = interior.copy()
    for src, dst in mapping.items():
        recolored[interior == src] = dst
    out[1:-1, 1:-1] = recolored
    return out


def _non_identity_permutation(colors: list[int], rng: random.Random) -> list[int]:
    if len(colors) == 2:
        return [colors[1], colors[0]]

    if rng.random() < 0.65:
        for _ in range(60):
            perm = colors[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(colors, perm)):
                return perm

    for _ in range(60):
        perm = colors[:]
        rng.shuffle(perm)
        if perm != colors:
            return perm

    shift = rng.randrange(1, len(colors))
    return colors[shift:] + colors[:shift]


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height >= 6 and width >= 6

    border_color = _border_background(g)
    assert int(g[0, 0]) == border_color
    assert int(g[0, -1]) == border_color
    assert int(g[-1, 0]) == border_color
    assert int(g[-1, -1]) == border_color
    assert np.all(g[-1, :] == border_color)
    assert np.all(g[:, -1] == border_color)

    top_seq = [int(g[0, c]) for c in range(1, width - 1) if int(g[0, c]) != border_color]
    left_seq = [int(g[r, 0]) for r in range(1, height - 1) if int(g[r, 0]) != border_color]
    assert 2 <= len(top_seq) <= 4
    assert len(top_seq) == len(left_seq)
    assert len(set(top_seq)) == len(top_seq)
    assert set(top_seq) == set(left_seq)
    assert top_seq != left_seq

    interior = g[1:-1, 1:-1]
    row_colors = [int(interior[r, 0]) for r in range(interior.shape[0])]
    assert all(np.all(interior[r, :] == row_colors[r]) for r in range(interior.shape[0]))
    assert set(row_colors) == set(top_seq)
    assert not np.array_equal(main(g), g)


def _gen_attempt(rng: random.Random) -> np.ndarray:
    color_count = rng.choice(COLOR_COUNT_CHOICES)
    chosen_colors = rng.sample(COLOR_POOL, color_count)

    min_height = max(GRID_HEIGHT_RANGE[0], color_count + 2)
    min_width = max(GRID_WIDTH_RANGE[0], color_count + 2)
    height = rng.randint(min_height, GRID_HEIGHT_RANGE[1])
    width = rng.randint(min_width, GRID_WIDTH_RANGE[1])

    top_positions = sorted(rng.sample(range(1, width - 1), color_count))
    left_positions = sorted(rng.sample(range(1, height - 1), color_count))

    top_order = chosen_colors[:]
    rng.shuffle(top_order)
    left_order = _non_identity_permutation(top_order, rng)

    stripe_order = chosen_colors[:]
    rng.shuffle(stripe_order)
    phase = rng.randrange(color_count)

    grid = np.full((height, width), BORDER_COLOR, dtype=int)
    for col, color in zip(top_positions, top_order):
        grid[0, col] = color
    for row, color in zip(left_positions, left_order):
        grid[row, 0] = color

    for inner_r in range(height - 2):
        color = stripe_order[(inner_r + phase) % color_count]
        grid[inner_r + 1, 1:-1] = color

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input grid")
