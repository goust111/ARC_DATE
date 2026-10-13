"""ARC task: crossing horizontal and vertical rays exchange their colors."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _markers(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (row, color) and (column, color) edge markers."""
    horizontal = [
        (r, int(grid[r, 0]))
        for r in range(1, grid.shape[0])
        if int(grid[r, 0]) != BACKGROUND
    ]
    vertical = [
        (c, int(grid[0, c]))
        for c in range(1, grid.shape[1])
        if int(grid[0, c]) != BACKGROUND
    ]
    return horizontal, vertical


def _validate_input(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Validate the visible task grammar and return its two marker families."""
    assert grid.ndim == 2, "expected a two-dimensional grid"
    height, width = map(int, grid.shape)
    assert height >= 4 and width >= 4, "grid is too small for crossing rays"
    assert int(grid[0, 0]) == BACKGROUND, "the shared corner is not a marker"

    horizontal, vertical = _markers(grid)
    assert horizontal and vertical, "both edge-marker families are required"

    interior = grid[1:, 1:]
    assert bool(np.all(interior == BACKGROUND)), "the input interior must be black"

    colors = [color for _, color in horizontal + vertical]
    assert all(color in MARKER_COLORS for color in colors), "markers use non-black, non-grey colors"
    assert len(colors) == len(set(colors)), "all marker colors must be distinct"
    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """Extend edge rays across the grid, swapping their carried colors at every grey crossing.

    Input structure:
    - A black grid has mutually different colored marker pixels only on the left
      and top edges; the shared top-left corner and the entire interior are black.
    - Each left-edge marker starts a rightward horizontal ray, and each top-edge
      marker starts a downward vertical ray.

    Rule:
    - Advance every ray toward its opposite edge while retaining each marker.
      At each horizontal/vertical intersection, paint the cell grey and exchange
      the two incoming carried colors; both rays then continue with the received
      colors. Repeated crossings apply these exchanges successively.

    Key insight:
    - A ray's later segments are determined by its complete ordered crossing
      history, so the output is a directed wire diagram rather than a set of
      independently drawn monochrome lines.
    """
    g = np.asarray(grid, dtype=int)
    horizontal, vertical = _validate_input(g)
    height, width = map(int, g.shape)

    horizontal_rows = {row for row, _ in horizontal}
    vertical_cols = {col for col, _ in vertical}
    horizontal_color = {row: color for row, color in horizontal}
    vertical_color = {col: color for col, color in vertical}

    out = g.copy()
    # Row-major order is a topological order: a crossing's horizontal input has
    # already arrived from the left and its vertical input from above.
    for r in range(1, height):
        for c in range(1, width):
            has_horizontal = r in horizontal_rows
            has_vertical = c in vertical_cols
            if has_horizontal and has_vertical:
                out[r, c] = GREY
                horizontal_color[r], vertical_color[c] = (
                    vertical_color[c],
                    horizontal_color[r],
                )
            elif has_horizontal:
                out[r, c] = horizontal_color[r]
            elif has_vertical:
                out[r, c] = vertical_color[c]
    return out


def _independent_oracle(grid: np.ndarray) -> np.ndarray:
    """Construct the answer by filling between successive crossing events."""
    horizontal, vertical = _validate_input(grid)
    height, width = map(int, grid.shape)
    rows = [r for r, _ in horizontal]
    cols = [c for c, _ in vertical]
    h_color = dict(horizontal)
    v_color = dict(vertical)
    out = grid.copy()

    previous_row = 0
    for r in rows:
        for c in cols:
            out[previous_row + 1 : r, c] = v_color[c]
        left = 1
        for c in cols:
            out[r, left:c] = h_color[r]
            out[r, c] = GREY
            h_color[r], v_color[c] = v_color[c], h_color[r]
            left = c + 1
        out[r, left:width] = h_color[r]
        previous_row = r
    for c in cols:
        out[previous_row + 1 : height, c] = v_color[c]
    return out


def _naive_unswapped(grid: np.ndarray) -> np.ndarray:
    """Deliberately wrong shortcut used to reject visually degenerate samples."""
    horizontal, vertical = _validate_input(grid)
    out = grid.copy()
    for row, color in horizontal:
        out[row, 1:] = color
    for col, color in vertical:
        out[1:, col] = color
    for row, _ in horizontal:
        for col, _ in vertical:
            out[row, col] = GREY
    return out


def _semantic_sample_ok(grid: np.ndarray, answer: np.ndarray) -> bool:
    """Check crossing semantics, visible swaps, uniqueness, and non-degeneracy."""
    horizontal, vertical = _validate_input(grid)
    rows = [r for r, _ in horizontal]
    cols = [c for c, _ in vertical]

    if len(rows) < 2 or len(cols) < 2:
        return False
    if not np.array_equal(answer, _independent_oracle(grid)):
        return False
    if np.array_equal(answer, _naive_unswapped(grid)):
        return False

    crossings = np.ix_(rows, cols)
    if not bool(np.all(answer[crossings] == GREY)):
        return False
    expected_grey = np.zeros(grid.shape, dtype=bool)
    expected_grey[crossings] = True
    if not np.array_equal(answer == GREY, expected_grey):
        return False

    # Margins on all four sides make the first incoming and final outgoing
    # segments visible. Distinct marker colors make every individual swap real.
    if min(rows) < 2 or max(rows) > grid.shape[0] - 3:
        return False
    if min(cols) < 2 or max(cols) > grid.shape[1] - 3:
        return False
    for row, start_color in horizontal:
        if int(answer[row, 1]) != start_color:
            return False
        if int(answer[row, -1]) == start_color:
            return False
    for col, start_color in vertical:
        if int(answer[1, col]) != start_color:
            return False
        if int(answer[-1, col]) == start_color:
            return False
    return True


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance with several crossings per ray."""
    for _ in range(300):
        height = random.randint(8, 18)
        width = random.randint(8, 18)

        max_horizontal = min(4, height - 4)
        max_vertical = min(4, width - 4)
        if max_horizontal < 2 or max_vertical < 2:
            continue

        n_horizontal = random.randint(2, max_horizontal)
        n_vertical = random.randint(2, max_vertical)
        if n_horizontal + n_vertical > len(MARKER_COLORS):
            continue

        rows = sorted(random.sample(range(2, height - 2), n_horizontal))
        cols = sorted(random.sample(range(2, width - 2), n_vertical))
        colors = random.sample(MARKER_COLORS, n_horizontal + n_vertical)
        random.shuffle(colors)

        grid = np.zeros((height, width), dtype=int)
        for row, color in zip(rows, colors[:n_horizontal]):
            grid[row, 0] = color
        for col, color in zip(cols, colors[n_horizontal:]):
            grid[0, col] = color

        answer = main(grid)
        if _semantic_sample_ok(grid, answer):
            return grid
    raise RuntimeError("failed to generate a semantically valid crossing-ray input")
