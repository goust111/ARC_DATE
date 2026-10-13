from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact every non-background pixel downward within its own column.

    Input structure:
    - A rectangular 2D grid has one dominant background color.
    - Foreground pixels are sparse cells whose columns are independent.

    Rule:
    - For each column, read the non-background pixels from top to bottom.
    - Clear the column to background, then write those pixels into the bottom
      cells of that same column, preserving their read order.

    Key insight:
    - This is stable vertical gravity: pixels fall straight down without changing
      columns or crossing each other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    height, width = map(int, g.shape)
    out = np.full_like(g, bg)

    for col in range(width):
        values = g[:, col][g[:, col] != bg]
        out[height - values.size :, col] = values

    return out

# --------------- generator (o5x) ---------------

# Canvas and density bounds.
SIZE_MIN, SIZE_MAX = 4, 6
FOREGROUND_MIN, FOREGROUND_MAX = 5, 9
DENSITY_MIN, DENSITY_MAX = 0.20, 0.36

# Column population and color constraints.
COLUMN_COUNTS = (1, 2, 3)
POPULATED_MIN, POPULATED_MAX = 4, 5
EMPTY_COLUMNS_MAX = 2
MULTI_PIXEL_MIN = 2
COLOR_MIN, COLOR_MAX = 1, 9

# Scatter requirements and retry behavior.
INTERNAL_GAP_MIN, INTERNAL_GAP_MAX = 1, 3
EXTRA_UNSETTLED_PROB = 0.45
GEN_ATTEMPTS = 300


def _gen_valid_totals(size: int) -> list[int]:
    return [
        total
        for total in range(FOREGROUND_MIN, FOREGROUND_MAX + 1)
        if DENSITY_MIN <= total / (size * size) <= DENSITY_MAX
    ]


def _gen_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    return [
        counts
        for counts in itertools.product(COLUMN_COUNTS, repeat=parts)
        if sum(counts) == total
    ]


def _gen_column_counts(size: int) -> list[int] | None:
    options = []
    for total in _gen_valid_totals(size):
        for populated in range(POPULATED_MIN, min(POPULATED_MAX, size, total) + 1):
            if size - populated > EMPTY_COLUMNS_MAX:
                continue
            for counts in _gen_compositions(total, populated):
                if any(count >= MULTI_PIXEL_MIN for count in counts):
                    options.append((populated, counts))
    if not options:
        return None

    populated, packed_counts = random.choice(options)
    columns = random.sample(range(size), populated)
    counts = [0] * size
    shuffled_counts = list(packed_counts)
    random.shuffle(shuffled_counts)
    for col, count in zip(columns, shuffled_counts):
        counts[col] = count
    return counts


def _gen_position_sets(size: int, count: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(size), count))


def _gen_settled_positions(size: int, count: int) -> tuple[int, ...]:
    return tuple(range(size - count, size))


def _gen_scattered_positions(
    size: int,
    count: int,
    *,
    require_unsettled: bool = False,
    require_internal_gap: bool = False,
) -> tuple[int, ...] | None:
    settled = _gen_settled_positions(size, count)
    choices = []
    for positions in _gen_position_sets(size, count):
        unsettled = positions != settled
        gap_sizes = [
            positions[i + 1] - positions[i] - 1
            for i in range(count - 1)
        ]
        has_internal_gap = any(
            INTERNAL_GAP_MIN <= gap <= INTERNAL_GAP_MAX for gap in gap_sizes
        )
        if require_unsettled and not unsettled:
            continue
        if require_internal_gap and not has_internal_gap:
            continue
        choices.append(positions)
    if not choices:
        return None
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(SIZE_MIN, SIZE_MAX)
    counts = _gen_column_counts(size)
    if counts is None:
        return None

    populated_cols = [col for col, count in enumerate(counts) if count]
    multi_cols = [col for col in populated_cols if counts[col] >= MULTI_PIXEL_MIN]
    if not multi_cols or len(populated_cols) < POPULATED_MIN:
        return None

    colors = random.sample(range(COLOR_MIN, COLOR_MAX + 1), len(populated_cols))
    color_by_col = dict(zip(populated_cols, colors))

    out = np.zeros((size, size), dtype=int)
    for col in populated_cols:
        count = counts[col]
        out[size - count:, col] = color_by_col[col]

    internal_gap_col = random.choice(multi_cols)
    possible_unsettled = [col for col in populated_cols if col != internal_gap_col]
    if not possible_unsettled:
        return None
    unsettled_cols = {internal_gap_col, random.choice(possible_unsettled)}
    for col in possible_unsettled:
        if random.random() < EXTRA_UNSETTLED_PROB:
            unsettled_cols.add(col)

    inp = np.zeros_like(out)
    for col in populated_cols:
        count = counts[col]
        positions = _gen_scattered_positions(
            size,
            count,
            require_unsettled=col in unsettled_cols,
            require_internal_gap=col == internal_gap_col,
        )
        if positions is None:
            return None
        inp[list(positions), col] = color_by_col[col]
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 1e0a9b12 input")
