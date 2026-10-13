import numpy as np

from arc_utils import Color
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress two stacked masks into one mask marking positions empty in both.

    Input structure:
    - The grid is two same-size square masks stacked vertically.
    - Each half contains the same background color plus one distinct foreground color.
    - Red is absent from the input and is used as the output marker color.

    Rule:
    - Compare the two halves cell by cell.
    - Output red where both cells are the shared background color.
    - Output the shared background color everywhere else.

    Key insight:
    - The output is the intersection of the background masks, not a transformation
      of either colored foreground object alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0, "expected two equally tall stacked masks"
    half = height // 2
    assert half == width, "expected each half to be square"

    top = g[:half, :]
    bottom = g[half:, :]
    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))

    shared_colors = top_colors & bottom_colors
    assert len(shared_colors) == 1, "expected exactly one shared background color"
    bg = shared_colors.pop()

    assert len(top_colors - {bg}) == 1, "expected one foreground color in the top mask"
    assert len(bottom_colors - {bg}) == 1, "expected one foreground color in the bottom mask"
    assert Color.RED not in (top_colors | bottom_colors), "red must be a new marker color"

    out = np.full((half, width), bg, dtype=int)
    out[(top == bg) & (bottom == bg)] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Grid shape and retry behavior.
_GEN_GRID_SIZE = 3
_GEN_RETRY_LIMIT = 300

# Palette selection.
_GEN_OBSERVED_PALETTE_PROB = 0.85
_GEN_OBSERVED_COLORS = (Color.BLACK, Color.MAROON, Color.BLUE)
_GEN_NON_RED_COLORS = [color for color in Color.ALL_COLORS if color != Color.RED]

# Foreground and output-marker count ranges for the 3x3 halves.
_GEN_TOP_FG_COUNTS = range(2, 8)
_GEN_BOTTOM_FG_COUNTS = range(2, 6)
_GEN_BOTH_BG_COUNTS = range(2, 6)

def _gen_feasible_counts():
    feasible = []
    area = _GEN_GRID_SIZE * _GEN_GRID_SIZE
    for top_count in _GEN_TOP_FG_COUNTS:
        for bottom_count in _GEN_BOTTOM_FG_COUNTS:
            for both_bg_count in _GEN_BOTH_BG_COUNTS:
                union_count = area - both_bg_count
                both_fg_count = top_count + bottom_count - union_count
                if 0 <= both_fg_count <= min(top_count, bottom_count):
                    feasible.append((top_count, bottom_count, both_bg_count, both_fg_count))
    return feasible


_GEN_FEASIBLE_COUNTS = _gen_feasible_counts()


def _gen_colors():
    if random.random() < _GEN_OBSERVED_PALETTE_PROB:
        return _GEN_OBSERVED_COLORS
    return tuple(random.sample(_GEN_NON_RED_COLORS, 3))


def _gen_cell_sets(n, top_count, bottom_count, both_bg_count, both_fg_count):
    cells = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(cells)

    both_bg = set(cells[:both_bg_count])
    union_cells = cells[both_bg_count:]
    random.shuffle(union_cells)

    top_only_count = top_count - both_fg_count
    bottom_start = both_fg_count + top_only_count
    both_fg = set(union_cells[:both_fg_count])
    top_only = set(union_cells[both_fg_count : both_fg_count + top_only_count])
    bottom_only = set(union_cells[bottom_start : bottom_start + bottom_count - both_fg_count])
    return both_bg, top_only | both_fg, bottom_only | both_fg


def _paint_cells(grid, cells, color):
    if cells:
        rows, cols = zip(*cells)
        grid[rows, cols] = color


def _gen_attempt():
    n = _GEN_GRID_SIZE
    bg, top_fg, bottom_fg = _gen_colors()
    counts = random.choice(_GEN_FEASIBLE_COUNTS)
    both_bg, top_cells, bottom_cells = _gen_cell_sets(n, *counts)

    top = np.full((n, n), bg, dtype=int)
    bottom = np.full((n, n), bg, dtype=int)
    _paint_cells(top, top_cells, top_fg)
    _paint_cells(bottom, bottom_cells, bottom_fg)

    inp = np.vstack([top, bottom])
    out = np.full((n, n), bg, dtype=int)
    _paint_cells(out, both_bg, Color.RED)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fafffa47 input")
