from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
COUNT_OPTIONS = (2, 3, 4, 5, 6)
N_COLOR_RANGE = (3, 5)
HEIGHT_RANGE = (8, 14)
WIDTH_PADDING_RANGE = (0, 4)
MAX_ATTEMPTS = 500


def _nonzero_counts(grid: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in np.asarray(grid, dtype=int).flat if int(value) != BG)


def _render_output(width: int, ordered_counts: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((len(ordered_counts), int(width)), dtype=int)
    for r, (color, freq) in enumerate(ordered_counts):
        out[r, :: int(freq)] = int(color)
    return out


def _ordered_by_frequency(counts: Counter[int]) -> list[tuple[int, int]]:
    items = [(int(color), int(freq)) for color, freq in counts.items()]
    freqs = [freq for _color, freq in items]
    assert items, "expected at least one foreground color"
    assert len(set(freqs)) == len(freqs), "foreground frequencies must be unique"
    return sorted(items, key=lambda item: item[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Convert scattered color frequencies into sorted periodic rows.

    Input structure:
    - The grid has black background and 3-5 non-black colors appearing only as isolated scatter pixels.
    - Each foreground color appears a unique number of times, and no geometric grouping is relevant.

    Rule:
    - Count how many times each non-black color appears.
    - Sort the colors by increasing frequency.
    - Create one output row per color, with output width equal to the input width.
    - For a color with frequency f, tile the 1D pattern [color, 0, 0, ..., 0] of length f from the left edge, truncating the last tile if needed.

    Key insight:
    - The same frequency value is reused twice: vertically it decides row order, and horizontally it decides the spacing of the repeated color pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ordered = _ordered_by_frequency(_nonzero_counts(g))
    return _render_output(g.shape[1], ordered)


def _first_seen_order(grid: np.ndarray) -> list[int]:
    seen: list[int] = []
    used: set[int] = set()
    for value in map(int, grid.flat):
        if value != BG and value not in used:
            used.add(value)
            seen.append(value)
    return seen


def _isolated_positions(height: int, width: int) -> list[tuple[int, int]]:
    phase_r = random.randint(0, 1)
    phase_c = random.randint(0, 1)
    return [(r, c) for r in range(phase_r, height, 2) for c in range(phase_c, width, 2)]


def _build_input(
    height: int,
    width: int,
    color_to_count: list[tuple[int, int]],
) -> np.ndarray | None:
    total = sum(count for _color, count in color_to_count)
    coords = _isolated_positions(height, width)
    if len(coords) < total:
        return None

    random.shuffle(coords)
    grid = np.zeros((height, width), dtype=int)
    index = 0
    for color, count in color_to_count:
        for _ in range(count):
            r, c = coords[index]
            grid[r, c] = int(color)
            index += 1
    return grid


def _frequency_signature(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(freq for _color, freq in _ordered_by_frequency(_nonzero_counts(grid)))


def _is_valid_instance(grid: np.ndarray) -> bool:
    counts = _nonzero_counts(grid)
    if not (N_COLOR_RANGE[0] <= len(counts) <= N_COLOR_RANGE[1]):
        return False

    freqs = [int(freq) for freq in counts.values()]
    if any(freq not in COUNT_OPTIONS for freq in freqs):
        return False
    if len(set(freqs)) != len(freqs):
        return False

    height, width = map(int, grid.shape)
    if max(freqs) > width // 2:
        return False

    # Keep the scatter visually clear: foreground pixels should not touch, even diagonally.
    fg = np.argwhere(grid != BG)
    occupied = {tuple(map(int, rc)) for rc in fg}
    for r, c in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in occupied:
                    return False

    ordered = [color for color, _freq in _ordered_by_frequency(counts)]
    asc = sorted(ordered)
    desc = sorted(ordered, reverse=True)
    if ordered == asc or ordered == desc:
        return False

    first_seen = _first_seen_order(grid)
    if first_seen == ordered or first_seen == ordered[::-1]:
        return False

    # Show both exact-period and truncated-period rows in the same sample.
    if not any(width % freq == 0 for freq in freqs):
        return False
    if not any(width % freq != 0 for freq in freqs):
        return False

    # Avoid a fully deterministic consecutive frequency ladder in every example.
    if len(freqs) < len(COUNT_OPTIONS) and _frequency_signature(grid) == tuple(range(min(freqs), max(freqs) + 1)):
        return False

    expected = _render_output(width, _ordered_by_frequency(counts))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if actual.shape != (len(counts), width):
        return False
    return height <= 30 and width <= 30 and actual.shape[0] <= 30 and actual.shape[1] <= 30


def _attempt() -> np.ndarray | None:
    n_colors = random.randint(*N_COLOR_RANGE)
    counts = sorted(random.sample(COUNT_OPTIONS, n_colors))
    max_count = counts[-1]

    min_width = max(8, 2 * max_count)
    max_width = min(15, 2 * max_count + WIDTH_PADDING_RANGE[1])
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    total = sum(counts)
    min_height = max(HEIGHT_RANGE[0], (2 * total + width - 1) // width)
    max_height = HEIGHT_RANGE[1]
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    palette = random.sample(COLOR_POOL, n_colors)
    random.shuffle(palette)
    color_to_count = list(zip(palette, counts))

    grid = _build_input(height, width, color_to_count)
    if grid is None:
        return None
    if not _is_valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-sorted tiling input")
