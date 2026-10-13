from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8
CATEGORY_COLORS = (RED, GREEN, YELLOW, AZURE)

BLOCK_SIZE = 3
MAX_ATTEMPTS = 300

GRID_HEIGHT_RANGE = (15, 24)
GRID_WIDTH_RANGE = (15, 23)
COUNT_VALUE_MAX = 8


def _detect_blocks(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return all valid 3x3 grey blocks as (center_row, center_col, category_color)."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color not in CATEGORY_COLORS:
                continue

            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            ring = patch.copy()
            ring[1, 1] = GREY
            if np.all(ring == GREY):
                blocks.append((r, c, color))

    return blocks


def _render_histogram(shape: tuple[int, int], counts: Counter[int]) -> np.ndarray:
    """Render the sorted bottom-aligned histogram on an all-black canvas."""
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    ordered = sorted(counts.items(), key=lambda item: -item[1])

    for i, (color, count) in enumerate(ordered):
        col = 2 * i
        assert col < w, "histogram columns must fit the canvas width"
        assert 1 <= count <= h, "bar height must fit the canvas height"
        out[h - count : h, col] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count 3x3 grey blocks by their center color, then draw a bottom histogram.

    Input structure:
    - Black background.
    - Several disjoint 3x3 solid grey blocks in the upper half.
    - The center cell of each block is replaced by one of {Red, Green, Yellow, Azure}.

    Rule:
    - Count how many blocks belong to each center color.
    - Sort the colors by count from high to low.
    - Clear the whole canvas to black.
    - Draw one-cell-wide vertical bars from left to right at the bottom edge.
      Each bar uses its category color and has height equal to that category count.
      Leave exactly one black column between neighboring bars.
    """
    g = np.asarray(grid, dtype=int)
    counts: Counter[int] = Counter(color for _r, _c, color in _detect_blocks(g))

    if not counts:
        return np.full_like(g, BLACK)

    freq_values = list(counts.values())
    assert len(freq_values) == len(set(freq_values)), "category counts must be unique"
    return _render_histogram(g.shape, counts)


def _random_composition(total: int, parts: int) -> list[int]:
    """Split a nonnegative integer into `parts` nonnegative pieces."""
    pieces = [0] * parts
    for _ in range(total):
        pieces[random.randrange(parts)] += 1
    return pieces


def _sample_axis_positions(length: int, count: int) -> list[int]:
    """Sample start positions for `count` size-3 blocks with at least one-cell gaps."""
    assert count >= 1
    min_used = BLOCK_SIZE * count + (count - 1)
    assert min_used <= length

    extra_gaps = _random_composition(length - min_used, count + 1)
    positions: list[int] = []
    cursor = extra_gaps[0]
    for i in range(count):
        positions.append(cursor)
        cursor += BLOCK_SIZE
        if i < count - 1:
            cursor += 1 + extra_gaps[i + 1]
    return positions


def _paint_block(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = GREY
    grid[top + 1, left + 1] = int(color)


def _sample_counts(category_count: int, capacity: int, height: int) -> dict[int, int] | None:
    """Assign distinct positive block counts to a random visible color subset."""
    pool = list(range(1, min(COUNT_VALUE_MAX, height) + 1))
    candidates = [combo for combo in combinations(pool, category_count) if sum(combo) <= capacity]
    if not candidates:
        return None

    colors = random.sample(list(CATEGORY_COLORS), category_count)
    values = list(random.choice(candidates))
    random.shuffle(values)
    return {color: value for color, value in zip(colors, values)}


def _build_input(
    height: int,
    width: int,
    counts: dict[int, int],
    row_slots: int,
    col_slots: int,
) -> np.ndarray:
    """Construct a valid input grid from counts and a random non-touching layout."""
    upper_height = height // 2
    row_positions = _sample_axis_positions(upper_height, row_slots)
    col_positions = _sample_axis_positions(width, col_slots)

    slots = [(r0, c0) for r0 in row_positions for c0 in col_positions]
    total_blocks = sum(counts.values())
    chosen_slots = random.sample(slots, total_blocks)

    colors: list[int] = []
    for color, count in counts.items():
        colors.extend([color] * count)
    random.shuffle(colors)

    grid = np.full((height, width), BLACK, dtype=int)
    for (r0, c0), color in zip(chosen_slots, colors):
        _paint_block(grid, r0, c0, color)
    return grid


def _expected_from_counts(shape: tuple[int, int], counts: dict[int, int]) -> np.ndarray:
    return _render_histogram(shape, Counter({int(color): int(count) for color, count in counts.items()}))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    upper_height = height // 2

    max_row_slots = (upper_height + 1) // 4
    max_col_slots = (width + 1) // 4
    capacity = max_row_slots * max_col_slots
    if max_row_slots == 0 or max_col_slots == 0:
        return None

    feasible_categories = [
        k
        for k in range(2, len(CATEGORY_COLORS) + 1)
        if k <= (width + 1) // 2 and k * (k + 1) // 2 <= capacity
    ]
    if not feasible_categories:
        return None

    category_count = random.choice(feasible_categories)
    counts = _sample_counts(category_count, capacity, height)
    if counts is None:
        return None

    total_blocks = sum(counts.values())
    slot_shapes = [
        (rows, cols)
        for rows in range(1, max_row_slots + 1)
        for cols in range(1, max_col_slots + 1)
        if rows * cols >= total_blocks
    ]
    if not slot_shapes:
        return None

    row_slots, col_slots = random.choice(slot_shapes)
    grid = _build_input(height, width, counts, row_slots, col_slots)

    detected = _detect_blocks(grid)
    if len(detected) != total_blocks:
        return None
    if Counter(color for _r, _c, color in detected) != Counter(counts):
        return None

    expected = _expected_from_counts(grid.shape, counts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
