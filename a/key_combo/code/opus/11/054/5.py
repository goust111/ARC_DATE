from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3


def _diagonal_sweep(line_grid: np.ndarray, blue_count: int) -> np.ndarray:
    """Paste the 1xN line on successive up-right diagonals with black as transparent."""
    g = np.asarray(line_grid, dtype=int)
    _, width = g.shape
    size = int(width * blue_count)
    out = np.full((size, size), BLACK, dtype=int)
    line = g[0]
    for shift in range(size):
        r = size - 1 - shift
        if r < 0:
            break
        for c, color in enumerate(line):
            color = int(color)
            cc = shift + c
            if color != BLACK and 0 <= cc < size:
                out[r, cc] = color
    return out


def _cross_centers(blue_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return all blue centers whose four orthogonal neighbors are also blue."""
    if min(blue_mask.shape) < 3:
        return []
    centers = (
        blue_mask[1:-1, 1:-1]
        & blue_mask[:-2, 1:-1]
        & blue_mask[2:, 1:-1]
        & blue_mask[1:-1, :-2]
        & blue_mask[1:-1, 2:]
    )
    rows, cols = np.nonzero(centers)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rows, cols)]


def _cross_footprint(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    """Union of every 5-cell cross centered at the given positions."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in centers:
        mask[r, c] = True
        mask[r - 1, c] = True
        mask[r + 1, c] = True
        mask[r, c - 1] = True
        mask[r, c + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sweep a sparse line diagonally, then recolor every blue 5-cell cross that appears in the swept pattern.

    Input structure:
    - The input is a single-row line of length N with black background cells.
    - Exactly one red endpoint marker sits at the far left or far right.
    - The remaining colored cells are blue pixels whose count k sets the output scale.

    Rule:
    - Create a black square canvas of size N*k.
    - Place the whole line at the bottom-left, then repeat it one row up and one column right for each step, clipping anything that leaves the canvas.
    - In the swept pattern, find every blue pixel whose up, down, left, and right neighbors are also blue.
    - Recolor each such blue cross (center plus four arms) to green; red swept pixels stay red and are never part of a cross.

    Key insight:
    - The cross is not present in the 1D input; it emerges only after multiple parallel diagonal traces interact with one another at unit orthogonal offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == 1, "expected a 1xN input line"

    line = g[0]
    width = int(line.shape[0])
    blue_positions = np.flatnonzero(line == BLUE)
    red_positions = np.flatnonzero(line == RED)

    assert width >= 1, "line must be non-empty"
    assert blue_positions.size >= 1, "expected at least one blue pixel"
    assert red_positions.size == 1, "expected exactly one red marker"
    assert int(red_positions[0]) in (0, width - 1), "red marker must be on an endpoint"

    swept = _diagonal_sweep(g, int(blue_positions.size))
    blue_mask = swept == BLUE
    centers = _cross_centers(blue_mask)

    out = swept.copy()
    if centers:
        out[_cross_footprint(out.shape, centers)] = GREEN
    return out


def _make_input(length: int, red_on_left: bool, blue_start: int) -> np.ndarray:
    """Build a valid 1xN input with one red endpoint and one consecutive 3-blue block."""
    grid = np.full((1, int(length)), BLACK, dtype=int)
    red_col = 0 if red_on_left else length - 1
    grid[0, red_col] = RED
    for c in range(int(blue_start), int(blue_start) + 3):
        grid[0, c] = BLUE
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    """Check structure, non-degeneracy, and the necessity of the blue-spacing rule."""
    g = np.asarray(grid, dtype=int)
    assert g.shape[0] == 1, "generator must emit a 1xN line"
    n = int(g.shape[1])
    assert 5 <= n <= 8, f"length out of range: {n}"

    blues = np.flatnonzero(g[0] == BLUE)
    reds = np.flatnonzero(g[0] == RED)
    assert reds.size == 1 and int(reds[0]) in (0, n - 1), "red endpoint missing"
    assert 2 <= blues.size <= 3, "blue count out of puzzle range"
    assert blues.size == 3, "non-degenerate training instances need three blue traces to form a blue cross"
    assert np.all(np.diff(blues) == 1), "generator should create one consecutive blue triple"

    swept = _diagonal_sweep(g, int(blues.size))
    assert swept.shape == (n * int(blues.size), n * int(blues.size))
    assert max(swept.shape) <= 30, "output exceeds ARC size limit"
    assert np.array_equal(swept[-1, :n], g[0]), "bottom-left placement must preserve the original line"

    centers = _cross_centers(swept == BLUE)
    assert len(centers) >= 2, "expected at least two full blue crosses after sweeping"

    expected = swept.copy()
    footprint = _cross_footprint(expected.shape, centers)
    expected[footprint] = GREEN
    out = main(g)
    assert np.array_equal(out, expected), "main must match the explicit sweep-then-recolor construction"

    changed = out != swept
    assert changed.any(), "recolor step must have a visible effect"
    assert np.all(swept[changed] == BLUE), "only blue pixels may be recolored"
    assert np.all(out[changed] == GREEN), "recolored pixels must become green"
    assert np.array_equal(out == RED, swept == RED), "red trace must be preserved exactly"

    # Counterfactual coverage: with only two blue traces, the same mechanism cannot form a full blue cross.
    red_col = int(reds[0])
    non_red_cols = [c for c in range(n) if c != red_col]
    for a, b in combinations(non_red_cols, 2):
        two_blue = np.full((1, n), BLACK, dtype=int)
        two_blue[0, red_col] = RED
        two_blue[0, [a, b]] = BLUE
        two_swept = _diagonal_sweep(two_blue, 2)
        assert len(_cross_centers(two_swept == BLUE)) == 0, "two blue traces should never fake a valid cross"

    # Each blue in the chosen triple is necessary for the detected crosses.
    for c in blues:
        reduced = g.copy()
        reduced[0, int(c)] = BLACK
        reduced_swept = _diagonal_sweep(reduced, 2)
        assert len(_cross_centers(reduced_swept == BLUE)) == 0, "every blue in the triple must matter"


def generate_input() -> np.ndarray:
    """
    Generate a random valid input line.

    The puzzle description allows 2-3 blue pixels, but the required training property
    here is that the swept pattern must contain multiple full blue crosses. Two parallel
    blue traces can never produce such a cross, so the generator samples the valid
    non-degenerate subset with three blues and varies length, red side, and triple position.
    """
    length = random.randint(5, 8)
    red_on_left = random.choice([False, True])
    if red_on_left:
        blue_start = random.randint(1, length - 3)
    else:
        blue_start = random.randint(0, length - 4)

    grid = _make_input(length, red_on_left, blue_start)
    _validate_generated(grid)
    return grid
