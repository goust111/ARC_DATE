from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


def _unique_least_frequent_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    least_count = int(counts.min())
    least_colors = colors[counts == least_count]
    assert least_colors.size == 1, "least frequent color must be unique"
    return int(least_colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the whole 3x3 input tile into the 9x9 output at positions marked by the least frequent input color.

    Input structure:
    - A 3x3 grid with no black pixels; black is reserved for the output background.
    - Exactly one color has the smallest pixel count.

    Rule:
    - Find the unique least frequent color in the input.
    - Build a 3-by-3 array of 3x3 blocks on a black canvas.
    - For each input cell of the least frequent color, copy the entire input grid
      into the corresponding output block; leave all other blocks black.

    Key insight:
    - The rare-color mask selects whole-tile positions; output cells are not scaled
      input pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (3, 3), "expected a 3x3 input grid"
    assert not np.any(g == Color.BLACK), "black is the output background color"

    marker_color = _unique_least_frequent_color(g)
    tile_h, tile_w = g.shape
    out = np.full((tile_h * tile_h, tile_w * tile_w), Color.BLACK, dtype=int)

    for r, c in np.argwhere(g == marker_color):
        r0 = int(r) * tile_h
        c0 = int(c) * tile_w
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = g

    return out

# --------------- generator (o5x) ---------------

# Grid geometry.
GEN_SIDE = 3
GEN_CELL_COUNT = GEN_SIDE * GEN_SIDE
GEN_OUTPUT_SHAPE = (GEN_CELL_COUNT, GEN_CELL_COUNT)

# Palette and count-plan sampling.
GEN_N_COLORS = range(2, 5)
GEN_MARKER_COUNTS = range(1, 5)
GEN_COLORS = range(1, 10)
GEN_OTHER_MIN_EXTRA = 1
GEN_FEASIBLE_COUNT_PAIRS = tuple(
    (n_colors, marker_count)
    for n_colors in GEN_N_COLORS
    for marker_count in GEN_MARKER_COUNTS
    if GEN_CELL_COUNT - marker_count >= (n_colors - 1) * (marker_count + GEN_OTHER_MIN_EXTRA)
)

# Retry budget.
GEN_ATTEMPTS = 300


def _gen_random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _gen_count_vector(n_colors: int, marker_count: int) -> list[int] | None:
    other_count = n_colors - 1
    min_other = marker_count + GEN_OTHER_MIN_EXTRA
    extra = GEN_CELL_COUNT - marker_count - other_count * min_other
    if extra < 0:
        return None
    extras = _gen_random_composition(extra, other_count)
    return [marker_count] + [min_other + x for x in extras]


def _gen_unique_least_frequent_color(grid: np.ndarray) -> int | None:
    colors, counts = np.unique(grid, return_counts=True)
    least = int(counts.min())
    least_colors = colors[counts == least]
    if least_colors.size != 1:
        return None
    return int(least_colors[0])


def _gen_expected_output(grid: np.ndarray) -> np.ndarray | None:
    marker = _gen_unique_least_frequent_color(grid)
    if marker is None:
        return None
    out = np.zeros(GEN_OUTPUT_SHAPE, dtype=int)
    for r, c in np.argwhere(grid == marker):
        r0 = int(r) * GEN_SIDE
        c0 = int(c) * GEN_SIDE
        out[r0 : r0 + GEN_SIDE, c0 : c0 + GEN_SIDE] = grid
    return out


def _gen_grid_from_counts(colors: list[int], counts: list[int]) -> np.ndarray:
    cells = []
    for color, count in zip(colors, counts):
        cells.extend([color] * count)
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(GEN_SIDE, GEN_SIDE)


def _gen_attempt() -> np.ndarray | None:
    n_colors, marker_count = random.choice(GEN_FEASIBLE_COUNT_PAIRS)
    counts = _gen_count_vector(n_colors, marker_count)
    if counts is None:
        return None

    colors = random.sample(GEN_COLORS, n_colors)
    marker_color = colors[0]
    grid = _gen_grid_from_counts(colors, counts)
    if _gen_unique_least_frequent_color(grid) != marker_color:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        expected = _gen_expected_output(grid)
        if expected is not None and np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid 48f8583b input")
