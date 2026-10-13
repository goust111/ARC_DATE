import random

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = list(range(1, 10))
GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the compact color table from the intersections of complete rows and columns.

    Input structure:
    - The background is black.
    - Some rows and some columns are completely foreground: every pixel in each such line is non-black.
    - Row and column strokes may use different foreground colors, and their crossings show whichever stroke is on top.

    Rule:
    - Identify every complete row and every complete column.
    - Build a new grid whose rows correspond to the complete input columns, from left to right.
    - Its columns correspond to the complete input rows, from top to bottom.
    - Each output cell is the color at the corresponding complete-column and complete-row intersection in the input.

    Key insight:
    The full foreground rows and columns form a crossing network; the answer is the compressed matrix of crossing colors, not a crop of the original space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    full_rows = np.flatnonzero(np.all(g != BACKGROUND, axis=1))
    full_cols = np.flatnonzero(np.all(g != BACKGROUND, axis=0))
    assert full_rows.size > 0 and full_cols.size > 0

    return g[np.ix_(full_rows, full_cols)].T.copy()


def _spaced_positions(length: int, count: int) -> list[int] | None:
    """Choose line positions with at least one background line between neighbors."""
    for _ in range(200):
        positions = sorted(random.sample(range(length), count))
        if all(b - a >= 2 for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _paint_lines(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    strokes = (
        [("row", int(r), int(color)) for r, color in zip(rows, row_colors)]
        + [("col", int(c), int(color)) for c, color in zip(cols, col_colors)]
    )
    random.shuffle(strokes)

    for kind, index, color in strokes:
        if kind == "row":
            grid[index, :] = color
        else:
            grid[:, index] = color
    return grid


def _sample_line_colors(row_count: int, col_count: int) -> tuple[list[int], list[int]]:
    palette_size = random.randint(3, min(7, len(FOREGROUND_COLORS)))
    palette = random.sample(FOREGROUND_COLORS, palette_size)

    row_colors = [random.choice(palette) for _ in range(row_count)]
    col_colors = [random.choice(palette) for _ in range(col_count)]

    if len(set(row_colors + col_colors)) == 1:
        replacement = random.choice([c for c in palette if c != row_colors[0]])
        if random.choice([True, False]):
            row_colors[random.randrange(row_count)] = replacement
        else:
            col_colors[random.randrange(col_count)] = replacement

    return row_colors, col_colors


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(2, 7)
    col_count = random.randint(2, 7)

    min_height = max(8, 2 * row_count + 1)
    min_width = max(8, 2 * col_count + 1)
    if min_height > 30 or min_width > 30:
        return None

    height = random.randint(min_height, 30)
    width = random.randint(min_width, 30)

    rows = _spaced_positions(height, row_count)
    cols = _spaced_positions(width, col_count)
    if rows is None or cols is None:
        return None

    row_colors, col_colors = _sample_line_colors(row_count, col_count)
    grid = _paint_lines(height, width, rows, cols, row_colors, col_colors)

    detected_rows = np.flatnonzero(np.all(grid != BACKGROUND, axis=1)).tolist()
    detected_cols = np.flatnonzero(np.all(grid != BACKGROUND, axis=0)).tolist()
    if detected_rows != rows or detected_cols != cols:
        return None

    expected = grid[np.ix_(rows, cols)].T.copy()
    if len(np.unique(expected)) < 2:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid complete-row-column input")
