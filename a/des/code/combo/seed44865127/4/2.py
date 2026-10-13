from __future__ import annotations

import itertools
import random
from collections import Counter
from functools import lru_cache

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _marker_rows(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if np.all(row == color):
            if color == bg:
                continue
            rows.append((int(r), color))
            continue
        assert np.all(row == bg), "non-marker rows must be pure background"
    assert rows, "expected at least one full-width marker row"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract full-width colored marker rows and expand them into a compact stripe stack.

    Input structure
    - The dominant color is a uniform background.
    - Some complete horizontal rows are filled with non-background colors.
    - Every other row is entirely background.

    Rule
    - Read the marker rows from top to bottom.
    - For each marker row, use the distance to the next marker row as that color's
      stripe height in the output; equivalently, repeat the color for gap+1 rows.
    - The last marker row contributes a stripe of height 1.
    - Stack those full-width stripes tightly in the same top-to-bottom color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = _dominant_color(g)
    markers = _marker_rows(g, bg)

    row_positions = [row for row, _color in markers]
    stripe_heights = [row_positions[i + 1] - row_positions[i] for i in range(len(markers) - 1)] + [1]
    assert all(h >= 1 for h in stripe_heights), "marker rows must be strictly increasing"

    out = np.empty((sum(stripe_heights), width), dtype=int)
    cursor = 0
    for (_row, color), stripe_height in zip(markers, stripe_heights):
        out[cursor : cursor + stripe_height, :] = color
        cursor += stripe_height
    return out


# --------------- generator ---------------

ARC_COLORS = tuple(range(10))
GRID_WIDTH_RANGE = (4, 18)
MARKER_COUNT_RANGE = (3, 6)
TOP_MARGIN_RANGE = (0, 5)
BOTTOM_MARGIN_RANGE = (0, 5)
OUTPUT_HEIGHT_RANGE = (5, 24)
INPUT_HEIGHT_RANGE = (8, 30)
MAX_GENERATION_ATTEMPTS = 400


@lru_cache(maxsize=None)
def _gap_sets(total: int, count: int) -> tuple[tuple[int, ...], ...]:
    if count == 0:
        return ((),) if total == 0 else ()
    values = [
        combo
        for combo in itertools.combinations(range(1, total + 1), count)
        if sum(combo) == total
    ]
    return tuple(values)


def _expected_output(rows: list[int], colors: list[int], width: int) -> np.ndarray:
    stripe_heights = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)] + [1]
    out = np.empty((sum(stripe_heights), width), dtype=int)
    cursor = 0
    for color, stripe_height in zip(colors, stripe_heights):
        out[cursor : cursor + stripe_height, :] = color
        cursor += stripe_height
    return out


def _attempt() -> np.ndarray | None:
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    gap_count = marker_count - 1
    min_gap_sum = gap_count * (gap_count + 1) // 2
    min_output_height = marker_count + min_gap_sum
    if min_output_height > OUTPUT_HEIGHT_RANGE[1]:
        return None

    output_height = random.randint(
        max(OUTPUT_HEIGHT_RANGE[0], min_output_height),
        OUTPUT_HEIGHT_RANGE[1],
    )
    total_gap_sum = output_height - marker_count
    gap_values = list(random.choice(_gap_sets(total_gap_sum, gap_count)))
    random.shuffle(gap_values)

    margin_pairs = [
        (top, bottom)
        for top in range(TOP_MARGIN_RANGE[0], TOP_MARGIN_RANGE[1] + 1)
        for bottom in range(BOTTOM_MARGIN_RANGE[0], BOTTOM_MARGIN_RANGE[1] + 1)
        if INPUT_HEIGHT_RANGE[0] <= top + output_height + bottom <= INPUT_HEIGHT_RANGE[1]
    ]
    if not margin_pairs:
        return None
    top_margin, bottom_margin = random.choice(margin_pairs)

    input_height = top_margin + output_height + bottom_margin

    bg = random.choice(ARC_COLORS)
    marker_colors = random.sample([color for color in ARC_COLORS if color != bg], marker_count)

    rows = [top_margin]
    for gap in gap_values:
        rows.append(rows[-1] + gap + 1)
    assert rows[-1] < input_height

    grid = np.full((input_height, width), bg, dtype=int)
    for row, color in zip(rows, marker_colors):
        grid[row, :] = color

    expected = _expected_output(rows, marker_colors, width)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    gaps = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
    if len(set(gaps)) != len(gaps):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
