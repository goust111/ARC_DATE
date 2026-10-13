import numpy as np
import random
from itertools import combinations


_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
_CORNER_COUNT_TO_LABEL = {
    0: 6,
    1: 1,
    2: 3,
    4: 2,
}


def main(grid: np.ndarray) -> np.ndarray:
    """Classify the 3x3 five-pixel mask into a one-cell label by its corner occupancy.

    Input structure
    - The input is a 3x3 grid with exactly two colors.
    - One color appears five times and forms the mask; the other appears four
      times as the background.

    Rule
    - Count how many of the four grid corners are part of the five-pixel mask.
    - Return the class label for that count: 0 corners -> 6, 1 -> 1,
      2 -> 3, and 4 -> 2.

    Key insight
    - Repeated masks with different colors produce the same output, so the
      mask's color is irrelevant; only the occupied corner positions determine
      the label.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3), "expected a 3x3 input grid"

    colors, counts = np.unique(g, return_counts=True)
    assert len(colors) == 2, "expected exactly two colors"

    mask_colors = [int(color) for color, count in zip(colors, counts) if int(count) == 5]
    bg_colors = [int(color) for color, count in zip(colors, counts) if int(count) == 4]
    assert len(mask_colors) == 1 and len(bg_colors) == 1, "expected 5 mask pixels and 4 background pixels"

    mask = g == mask_colors[0]
    corner_count = sum(int(mask[r, c]) for r, c in _CORNERS)
    assert corner_count in _CORNER_COUNT_TO_LABEL, "unexpected corner occupancy class"

    return np.array([[_CORNER_COUNT_TO_LABEL[corner_count]]], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas and mask geometry.
_GEN_GRID_SHAPE = (3, 3)
_GEN_MASK_CELL_COUNT = 5
_GEN_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
_GEN_NON_CORNERS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))

# Sampling, labels, and retry budget.
_GEN_ALLOWED_CORNER_COUNTS = (0, 1, 2, 4)
_GEN_MASK_COLOR_RANGE = (1, 9)
_GEN_RETRY_LIMIT = 300
_GEN_LABEL_BY_CORNER_COUNT = {0: 6, 1: 1, 2: 3, 4: 2}
_GEN_PATTERNS_BY_CORNER_COUNT = {
    corner_count: [
        tuple(corner_cells) + tuple(non_corner_cells)
        for corner_cells in combinations(_GEN_CORNERS, corner_count)
        for non_corner_cells in combinations(_GEN_NON_CORNERS, _GEN_MASK_CELL_COUNT - corner_count)
    ]
    for corner_count in _GEN_ALLOWED_CORNER_COUNTS
}


def _sample_layout():
    """Return (corner_count, mask_cells) using the configured uniform draws."""
    corner_count = random.choice(_GEN_ALLOWED_CORNER_COUNTS)
    return corner_count, random.choice(_GEN_PATTERNS_BY_CORNER_COUNT[corner_count])


def _render_grid(pattern, mask_color):
    """Render the sampled mask on a zero background."""
    grid = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    grid[tuple(zip(*pattern))] = mask_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    corner_count, pattern = _sample_layout()
    mask_color = random.randint(*_GEN_MASK_COLOR_RANGE)
    grid = _render_grid(pattern, mask_color)
    expected = np.array([[_GEN_LABEL_BY_CORNER_COUNT[corner_count]]], dtype=int)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid 27a28665 input")
