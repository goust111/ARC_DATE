from __future__ import annotations

from collections import Counter
import random

import numpy as np


ORANGE = 7
ALL_COLORS = tuple(range(10))
INPUT_COLORS = tuple(color for color in ALL_COLORS if color != ORANGE)

GRID_HEIGHT_RANGE = (4, 10)
GRID_WIDTH_RANGE = (4, 10)
INPUT_COLOR_COUNT_RANGE = (3, 6)
GENERATION_RETRIES = 400


def _top_two_colors(grid: np.ndarray) -> set[int]:
    counts = Counter(map(int, grid.ravel()))
    assert len(counts) >= 3, "input must contain at least three colors"

    ranked_counts = sorted(set(counts.values()), reverse=True)
    assert len(ranked_counts) >= 2, "there must be at least two frequency ranks"

    top_count = ranked_counts[0]
    top_colors = {color for color, count in counts.items() if count == top_count}
    assert len(top_colors) <= 2, "no more than two colors may share the top rank"
    if len(top_colors) == 2:
        return top_colors

    second_count = ranked_counts[1]
    second_colors = {color for color, count in counts.items() if count == second_count}
    assert len(second_colors) == 1, "second rank must contain exactly one color"
    return top_colors | second_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the two most frequent input colors and recolor every other pixel orange.

    Input structure:
    - The grid contains at least three colors.
    - Orange is absent from the input and is only used as the replacement color.
    - Exactly two colors occupy the top two frequency ranks; they may tie with
      each other, but no third color ties at the cutoff.

    Rule:
    - Preserve all pixels whose color is one of the two highest-frequency colors.
    - Replace every other pixel with orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == ORANGE), "orange must not appear in the input"

    keep_colors = _top_two_colors(g)
    assert len(keep_colors) == 2, "expected exactly two preserved colors"

    out = np.full(g.shape, ORANGE, dtype=int)
    for color in keep_colors:
        out[g == color] = color
    return out


def _sample_distinct_counts(total: int, n_colors: int) -> list[int] | None:
    if n_colors < 3 or total < n_colors:
        return None

    top2_sum_min = max(5, total // 2)
    top2_sum_max = total - (n_colors - 2)
    if top2_sum_min > top2_sum_max:
        return None

    for _ in range(200):
        major_total = random.randint(top2_sum_min, top2_sum_max)
        max_second = min((major_total - 1) // 2, total - major_total - (n_colors - 2))
        if max_second < 2:
            continue
        second = random.randint(2, max_second)
        first = major_total - second
        if first < second + 1:
            continue

        remaining = total - first - second
        minor = [1] * (n_colors - 2)
        remaining -= n_colors - 2
        while remaining > 0:
            choices = [i for i, value in enumerate(minor) if value < second - 1]
            if not choices:
                break
            idx = random.choice(choices)
            minor[idx] += 1
            remaining -= 1
        if remaining != 0:
            continue

        counts = [first, second] + minor
        if len(set(counts)) >= 2 and max(minor) < second:
            return counts
    return None


def _sample_tied_top_counts(total: int, n_colors: int) -> list[int] | None:
    if n_colors < 3 or total < n_colors:
        return None

    for _ in range(200):
        top = random.randint(2, max(2, total // 3 + 1))
        if 2 * top > total - (n_colors - 2):
            continue
        remaining = total - 2 * top
        if remaining < n_colors - 2:
            continue

        minor = [1] * (n_colors - 2)
        remaining -= n_colors - 2
        while remaining > 0:
            choices = [i for i, value in enumerate(minor) if value < top - 1]
            if not choices:
                break
            idx = random.choice(choices)
            minor[idx] += 1
            remaining -= 1
        if remaining != 0:
            continue

        counts = [top, top] + minor
        if max(minor) < top:
            return counts
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    total = height * width

    max_colors = min(INPUT_COLOR_COUNT_RANGE[1], len(INPUT_COLORS), total)
    n_colors = random.randint(INPUT_COLOR_COUNT_RANGE[0], max_colors)
    branch = random.choice(("distinct", "tied_top"))
    counts = (
        _sample_distinct_counts(total, n_colors)
        if branch == "distinct"
        else _sample_tied_top_counts(total, n_colors)
    )
    if counts is None:
        return None

    colors = random.sample(INPUT_COLORS, n_colors)
    pixels = np.concatenate(
        [np.full(count, color, dtype=int) for color, count in zip(colors, counts)]
    )
    if pixels.size != total:
        return None

    random.shuffle(pixels)
    grid = pixels.reshape(height, width)

    try:
        out = main(grid)
    except Exception:
        return None

    expected_keep = set(colors[:2])
    expected = np.full(grid.shape, ORANGE, dtype=int)
    for color in expected_keep:
        expected[grid == color] = color
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
