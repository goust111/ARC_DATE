from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
GREY_NOISE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_SIZE_RANGE = (10, 11)
FOREGROUND_COLOR_COUNTS = (3, 4, 5)
PIXEL_COUNT_CHOICES = (1, 2, 3, 4, 5, 6)
GREY_NOISE_CHOICES = (0, 1, 2)
MAX_GENERATION_ATTEMPTS = 300


def _foreground_counts(grid: np.ndarray) -> dict[int, int]:
    counts = Counter(map(int, grid.flat))
    return {
        color: count
        for color, count in counts.items()
        if color not in (BACKGROUND, GREY_NOISE)
    }


def _sorted_colors_by_count(color_counts: dict[int, int]) -> list[int]:
    assert color_counts, "expected at least one non-black, non-grey foreground color"
    counts_only = list(color_counts.values())
    assert len(counts_only) == len(set(counts_only)), "foreground color counts must be unique"
    return [color for color, _count in sorted(color_counts.items(), key=lambda item: item[1])]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each non-gray foreground color and encode the ascending ranking on the main diagonal.

    Input structure:
    - The grid has a black background with scattered single-pixel foreground cells.
    - Three to five non-black colors appear, and gray cells may appear as singleton distractors.
    - Each non-gray foreground color has a distinct total pixel count.

    Rule:
    - Ignore black background cells and gray distractor cells.
    - Count the total number of pixels of each remaining color and sort the colors from the smallest count to the largest.
    - Return a black KxK grid, where K is the number of counted colors, and place the i-th sorted color at diagonal cell (i, i).

    Key insight:
    - The output is not a tally of counts; it is a ranked color code where count order determines diagonal position and each diagonal cell keeps the original color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color_counts = _foreground_counts(g)
    ordered_colors = _sorted_colors_by_count(color_counts)

    k = len(ordered_colors)
    out = np.full((k, k), BACKGROUND, dtype=int)
    for i, color in enumerate(ordered_colors):
        out[i, i] = int(color)
    return out


def _build_expected_output(color_counts: dict[int, int]) -> np.ndarray:
    ordered_colors = _sorted_colors_by_count(color_counts)
    out = np.full((len(ordered_colors), len(ordered_colors)), BACKGROUND, dtype=int)
    for i, color in enumerate(ordered_colors):
        out[i, i] = int(color)
    return out


def _candidate_cells(height: int, width: int) -> list[tuple[int, int]]:
    row_offset = random.randint(0, 1)
    col_offset = random.randint(0, 1)
    rows = list(range(row_offset, height, 2))
    cols = list(range(col_offset, width, 2))
    cells = [(r, c) for r in rows for c in cols]
    random.shuffle(cells)
    return cells


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    n_colors = random.choice(FOREGROUND_COLOR_COUNTS)
    grey_count = random.choice(GREY_NOISE_CHOICES)

    colors = random.sample(PALETTE, n_colors)
    counts = random.sample(PIXEL_COUNT_CHOICES, n_colors)
    color_to_count = dict(zip(colors, counts))

    total_needed = sum(counts) + grey_count
    cells = _candidate_cells(height, width)
    if total_needed > len(cells):
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    cursor = 0
    for color, count in zip(colors, counts):
        for r, c in cells[cursor : cursor + count]:
            grid[r, c] = int(color)
        cursor += count

    for r, c in cells[cursor : cursor + grey_count]:
        grid[r, c] = GREY_NOISE

    expected = _build_expected_output(color_to_count)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
