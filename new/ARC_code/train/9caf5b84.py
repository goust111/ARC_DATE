from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


def _two_most_frequent_colors(g: np.ndarray) -> set[int]:
    colors, counts = np.unique(g, return_counts=True)
    assert colors.size >= 3, "input must contain at least three colors"

    max_count = int(counts.max())
    max_colors = {int(c) for c, n in zip(colors, counts) if int(n) == max_count}
    assert len(max_colors) <= 2, "at most two colors may share the largest count"
    if len(max_colors) == 2:
        return max_colors

    lower_counts = [int(n) for n in counts if int(n) < max_count]
    assert lower_counts, "expected a second most frequent color"
    second_count = max(lower_counts)
    second_colors = {int(c) for c, n in zip(colors, counts) if int(n) == second_count}
    assert len(second_colors) == 1, "second most frequent color must be unique"
    return max_colors | second_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every pixel whose color is not one of the two most frequent input colors to orange.

    Input structure:
    - The grid contains at least three colors.
    - Orange is absent from the input and is used only as the output replacement color.
    - Exactly two colors occupy the top two frequency ranks; they may tie with each
      other, but no third color ties them at the cutoff.

    Rule:
    - Preserve every pixel whose color is one of the two most frequent input colors.
    - Replace every other pixel with orange.

    Key insight:
    - The visible distinction is not the numeric identity of the colors but their
      population: the two majority colors remain, while all minority colors merge
      into the new orange color.
    """
    g = np.asarray(grid, dtype=int)
    assert not np.any(g == Color.ORANGE), "orange must not already appear in the input"

    keep = _two_most_frequent_colors(g)
    assert len(keep) == 2, "expected exactly two colors to preserve"

    out = np.full(g.shape, Color.ORANGE, dtype=int)
    for color in keep:
        out[g == color] = color
    return out

# --------------- generator (att2) ---------------

# CFG: shape, palette, count constraints, and retry budget.
GRID_MIN_SIZE, GRID_MAX_SIZE = 4, 6
INPUT_COLOR_COUNT = 5
AVAILABLE_INPUT_COLORS = tuple(c for c in range(10) if c != Color.ORANGE)
MINORITY_FRACTION_MIN, MINORITY_FRACTION_MAX = 0.29, 0.50
MINORITY_COLOR_COUNT = 3
COUNT_GAP_MIN, COUNT_GAP_MAX = 1, 5
GENERATION_RETRIES = 300


def _gen_valid_count_vectors(total: int) -> list[tuple[int, int, int, int, int]]:
    vectors: list[tuple[int, int, int, int, int]] = []
    min_minority = int(np.ceil(MINORITY_FRACTION_MIN * total))
    max_minority = int(np.floor(MINORITY_FRACTION_MAX * total))
    for second in range(2, total):
        for first in range(second, total):
            minority_total = total - first - second
            if not min_minority <= minority_total <= max_minority:
                continue
            if minority_total < MINORITY_COLOR_COUNT:
                continue
            if minority_total > MINORITY_COLOR_COUNT * (second - 1):
                continue
            for minor0 in range(1, second):
                for minor1 in range(1, second):
                    minor2 = minority_total - minor0 - minor1
                    if 1 <= minor2 < second:
                        gap = second - max(minor0, minor1, minor2)
                        if COUNT_GAP_MIN <= gap <= COUNT_GAP_MAX:
                            vectors.append((first, second, minor0, minor1, minor2))
    return vectors


def _gen_expected_output(grid: np.ndarray, keep_colors: set[int]) -> np.ndarray:
    out = np.full(grid.shape, Color.ORANGE, dtype=int)
    for color in keep_colors:
        out[grid == color] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    total = height * width

    colors = random.sample(AVAILABLE_INPUT_COLORS, INPUT_COLOR_COUNT)
    counts = random.choice(_gen_valid_count_vectors(total))

    pixels = [color for color, count in zip(colors, counts) for _ in range(count)]
    random.shuffle(pixels)
    grid = np.array(pixels, dtype=int).reshape(height, width)

    expected = _gen_expected_output(grid, {colors[0], colors[1]})
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9caf5b84 input")
