from __future__ import annotations

import numpy as np

from arc_utils import Color


def _unique_most_frequent_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    assert colors.size >= 2, "input must contain a majority color and at least one other color"

    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "the most frequent color must be unique"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve the uniquely most frequent input color and recolor all other pixels gray.

    Input structure:
    - The grid contains at least two colors.
    - Gray is absent from the input and is used only as the output replacement color.
    - Exactly one input color has the largest pixel count.

    Rule:
    - Find the unique most frequent color in the whole grid.
    - Keep pixels of that color in their original locations.
    - Replace every other pixel with gray.

    Key insight:
    - The visible role of the preserved object is its population: the majority
      color remains, while all minority colors are merged into the same gray fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert not np.any(g == Color.GREY), "gray must not already appear in the input"

    keep_color = _unique_most_frequent_color(g)
    out = np.full(g.shape, Color.GREY, dtype=int)
    out[g == keep_color] = keep_color
    return out

# --------------- generator (o5x) ---------------

# Grid and color palette.
GRID_SHAPE = (3, 3)
CELL_COUNT = GRID_SHAPE[0] * GRID_SHAPE[1]
REPLACEMENT_COLOR = int(Color.GREY)
INPUT_PALETTE = [c for c in range(10) if c != REPLACEMENT_COLOR]
MIN_COLORS, MAX_COLORS = 2, 4
MIN_MAJORITY_COUNT, MAX_MAJORITY_COUNT = 3, 7
MAX_RUNNER_UP_COUNT = 3
GENERATION_ATTEMPTS = 300


def _gen_bounded_composition(total: int, parts: int, max_value: int) -> list[int] | None:
    values: list[int] = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        lo = max(1, remaining - left * max_value)
        hi = min(max_value, remaining - left)
        if lo > hi:
            return None
        value = int(np.random.randint(lo, hi + 1))
        values.append(value)
        remaining -= value
    np.random.shuffle(values)
    return values


def _gen_expected_output(inp: np.ndarray, keep_color: int) -> np.ndarray:
    out = np.full(inp.shape, REPLACEMENT_COLOR, dtype=int)
    out[inp == keep_color] = keep_color
    return out


def _gen_attempt() -> np.ndarray | None:
    num_colors = int(np.random.randint(MIN_COLORS, MAX_COLORS + 1))
    min_majority = max(
        MIN_MAJORITY_COUNT,
        (CELL_COUNT // num_colors) + 1,
        CELL_COUNT - (num_colors - 1) * MAX_RUNNER_UP_COUNT,
    )
    max_majority = min(MAX_MAJORITY_COUNT, CELL_COUNT - (num_colors - 1))
    if min_majority > max_majority:
        return None

    majority_count = int(np.random.randint(min_majority, max_majority + 1))
    minority_counts = _gen_bounded_composition(
        CELL_COUNT - majority_count,
        num_colors - 1,
        min(MAX_RUNNER_UP_COUNT, majority_count - 1),
    )
    if minority_counts is None:
        return None

    colors = np.random.choice(INPUT_PALETTE, size=num_colors, replace=False).astype(int).tolist()
    keep_color = colors[0]

    cells = [keep_color] * majority_count
    for color, count in zip(colors[1:], minority_counts):
        cells.extend([color] * count)
    np.random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(GRID_SHAPE)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        colors, counts = np.unique(inp, return_counts=True)
        winners = colors[counts == counts.max()]
        if winners.size != 1:
            continue
        expected = _gen_expected_output(inp, int(winners[0]))
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            pass
    raise RuntimeError("failed to generate a valid input")
