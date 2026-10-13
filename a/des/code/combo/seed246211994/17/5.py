from __future__ import annotations

import math
from collections import Counter

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Pack colors by descending frequency into bottom-up columns.

    Input structure
    - The input is a full foreground rectangle: every cell is non-background.
    - Each present color appears a distinct number of times.

    Rule
    - Count the pixels of each color and sort colors from most frequent to least frequent.
    - Rebuild a same-size grid by scanning columns from left to right and, within
      each column, from bottom to top.
    - Fill that scan order with one contiguous run per color, exhausting one color
      before moving to the next.

    Key insight
    - The transformation preserves only the multiset of colors and their counts;
      all original positions are discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "grid must be non-empty"
    assert np.all(g != 0), "input must be all foreground"

    flat = g.reshape(-1)
    counts = Counter(map(int, flat))

    first_seen: dict[int, int] = {}
    for idx, color in enumerate(map(int, flat)):
        first_seen.setdefault(color, idx)

    colors = sorted(counts, key=lambda color: (-counts[color], first_seen[color]))
    packed = np.concatenate([np.full(counts[color], int(color), dtype=int) for color in colors])

    height, width = g.shape
    rows = np.tile(np.arange(height - 1, -1, -1, dtype=int), width)
    cols = np.repeat(np.arange(width, dtype=int), height)

    out = np.empty_like(g)
    out[rows, cols] = packed
    return out


# --------------- generator ---------------

COLOR_POOL = np.arange(1, 10, dtype=int)
HEIGHT_RANGE = (2, 12)
WIDTH_RANGE = (2, 12)
MIN_COLORS = 2
MAX_COLORS = 6
MAX_GENERATION_ATTEMPTS = 300


def _max_unique_count_parts(area: int) -> int:
    """Largest k with 1 + 2 + ... + k <= area."""
    return (math.isqrt(1 + 8 * area) - 1) // 2


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    height = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    return height, width


def _sample_color_count(area: int, rng: np.random.Generator) -> int:
    max_by_area = _max_unique_count_parts(area)
    upper = min(MAX_COLORS, len(COLOR_POOL), max_by_area)
    assert upper >= MIN_COLORS
    return int(rng.integers(MIN_COLORS, upper + 1))


def _sample_unique_counts(area: int, n_colors: int, rng: np.random.Generator) -> list[int]:
    """Sample positive distinct counts that sum to the full grid area."""
    min_sum = n_colors * (n_colors + 1) // 2
    assert min_sum <= area

    remainder = area - min_sum
    increments = rng.multinomial(remainder, np.full(n_colors, 1.0 / n_colors))
    increments.sort()

    counts = (np.arange(1, n_colors + 1, dtype=int) + increments).astype(int)
    counts = counts.tolist()
    rng.shuffle(counts)
    assert len(set(counts)) == n_colors
    assert sum(counts) == area
    return counts


def _shuffled_layout(colors: np.ndarray, counts: list[int], rng: np.random.Generator) -> np.ndarray:
    pixels = np.repeat(colors, counts)
    rng.shuffle(pixels)
    return pixels.astype(int, copy=False)


def _run_layout(colors: np.ndarray, counts: list[int], rng: np.random.Generator) -> np.ndarray:
    remaining = {int(color): int(count) for color, count in zip(colors, counts)}
    active = [int(color) for color in colors]
    sequence: list[int] = []

    while active:
        color = int(rng.choice(active))
        max_run = remaining[color]
        run_len = int(rng.integers(1, max_run + 1))
        sequence.extend([color] * run_len)
        remaining[color] -= run_len
        if remaining[color] == 0:
            active.remove(color)

    return np.array(sequence, dtype=int)


def _snake_cell_order(height: int, width: int) -> np.ndarray:
    order = []
    for row in range(height):
        cols = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
        order.extend(row * width + col for col in cols)
    return np.array(order, dtype=int)


def _column_cell_order(height: int, width: int) -> np.ndarray:
    return np.array([row * width + col for col in range(width) for row in range(height)], dtype=int)


def _layout_on_order(
    colors: np.ndarray,
    counts: list[int],
    order: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sequence = _run_layout(colors, counts, rng)
    pixels = np.empty(order.size, dtype=int)
    pixels[order] = sequence
    return pixels


def _build_pixels(
    colors: np.ndarray,
    counts: list[int],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    area = height * width
    mode = int(rng.integers(0, 3))
    if mode == 0:
        pixels = _shuffled_layout(colors, counts, rng)
    elif mode == 1:
        pixels = _layout_on_order(colors, counts, _snake_cell_order(height, width), rng)
    else:
        pixels = _layout_on_order(colors, counts, _column_cell_order(height, width), rng)

    assert pixels.shape == (area,)
    return pixels


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height, width = _sample_shape(rng)
    area = height * width

    n_colors = _sample_color_count(area, rng)
    colors = rng.choice(COLOR_POOL, size=n_colors, replace=False)
    counts = _sample_unique_counts(area, n_colors, rng)
    pixels = _build_pixels(colors, counts, height, width, rng)

    grid = pixels.reshape(height, width)
    if not np.all(grid != 0):
        return None

    observed = Counter(map(int, grid.reshape(-1)))
    if len(observed) != n_colors or len(set(observed.values())) != n_colors:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if Counter(map(int, out.reshape(-1))) != observed:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input in {MAX_GENERATION_ATTEMPTS} attempts")
