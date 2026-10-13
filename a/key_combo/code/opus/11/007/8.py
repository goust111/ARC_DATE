from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GRID_MIN = 9
GRID_MAX = 12
SEGMENT_COUNT_RANGE = (3, 5)
SEGMENT_LENGTH_RANGE = (2, 5)
GEN_RETRIES = 400


def _corner_positions(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    h, w = shape
    return ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _ring_background(shape: tuple[int, int], marker: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.empty((h, w), dtype=int)
    mr, mc = marker
    for r in range(h):
        for c in range(w):
            out[r, c] = RED if max(abs(r - mr), abs(c - mc)) % 2 == 0 else GREY
    return out


def _blue_runs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == BLUE)
        if cols.size == 0:
            continue
        assert np.all(np.diff(cols) == 1), "each occupied row must contain one contiguous blue segment"
        runs.append((r, int(cols[0]), int(cols[-1])))
    return runs


def _find_corner_marker(grid: np.ndarray) -> tuple[int, int]:
    red_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == RED)]
    assert len(red_coords) == 1, "expected exactly one red marker"
    marker = red_coords[0]
    assert marker in _corner_positions(grid.shape), "red marker must be placed in a corner"
    return marker


def _validate_solver_input(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, RED}, "input may contain only black, blue, and red"
    marker = _find_corner_marker(g)
    runs = _blue_runs(g)
    assert runs, "expected at least one blue segment"
    return marker, runs


def _shift_blue_mask(grid: np.ndarray) -> np.ndarray:
    blue = grid == BLUE
    shifted = np.zeros_like(blue, dtype=bool)
    shifted[:, :-1] = blue[:, 1:]
    return shifted


def _spec_output(grid: np.ndarray) -> np.ndarray:
    marker, _runs = _validate_solver_input(grid)
    out = _ring_background(grid.shape, marker)
    out[_shift_blue_mask(grid)] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the blue foreground left and repaint the revealed background as red-grey Chebyshev rings from the corner marker.

    Input structure:
    - The grid is mostly black and contains one red single-pixel marker in exactly one corner.
    - Blue foreground pixels form one or more horizontal one-cell-thick contiguous segments on separate rows.
    - No other foreground colors are present in the input.

    Rule:
    - Move every blue pixel one column left, clipping any pixel that would leave the grid.
    - For every cell not occupied by a shifted blue pixel, compute its Chebyshev distance to the red corner marker.
    - Paint even-distance cells red and odd-distance cells grey; the marker itself stays red because its distance is zero.
    - Keep the shifted blue pixels blue so they overwrite the ring background at their new locations.

    Key insight:
    - Two independent spatial systems must be combined: the blue foreground undergoes a fixed west shift, while the entire remaining canvas becomes alternating square rings centered on the lone red corner marker.
    """
    g = np.asarray(grid, dtype=int)
    marker, _runs = _validate_solver_input(g)

    out = np.empty(g.shape, dtype=int)
    mr, mc = marker
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            d = max(abs(r - mr), abs(c - mc))
            out[r, c] = RED if d % 2 == 0 else GREY

    for r, c in np.argwhere(g == BLUE):
        rr = int(r)
        cc = int(c) - 1
        if cc >= 0:
            out[rr, cc] = BLUE
    return out


def _choose_spaced_rows(height: int, count: int) -> list[int]:
    base = sorted(random.sample(range(height - (count - 1)), count))
    return [r + i for i, r in enumerate(base)]


def _run_conflicts_with_marker(
    row: int,
    c0: int,
    c1: int,
    marker: tuple[int, int],
) -> bool:
    mr, mc = marker
    return row == mr and c0 <= mc <= c1


def _make_candidate() -> np.ndarray:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    segment_count = random.randint(*SEGMENT_COUNT_RANGE)
    assert height >= 2 * segment_count - 1

    grid = np.zeros((height, width), dtype=int)
    marker = random.choice(_corner_positions(grid.shape))
    grid[marker] = RED

    rows = _choose_spaced_rows(height, segment_count)
    for row in rows:
        length = random.randint(*SEGMENT_LENGTH_RANGE)
        c0 = random.randint(1, width - length)
        c1 = c0 + length - 1
        if _run_conflicts_with_marker(row, c0, c1, marker):
            return None
        grid[row, c0 : c1 + 1] = BLUE
    return grid


def _generator_semantics_ok(grid: np.ndarray) -> bool:
    try:
        marker, runs = _validate_solver_input(grid)
    except AssertionError:
        return False

    if not (SEGMENT_COUNT_RANGE[0] <= len(runs) <= SEGMENT_COUNT_RANGE[1]):
        return False
    if any(not (SEGMENT_LENGTH_RANGE[0] <= c1 - c0 + 1 <= SEGMENT_LENGTH_RANGE[1]) for _, c0, c1 in runs):
        return False
    if any(c0 == 0 for _, c0, _ in runs):
        return False
    rows = [r for r, _, _ in runs]
    if any(b - a < 2 for a, b in zip(rows, rows[1:])):
        return False

    shifted = _shift_blue_mask(grid)
    input_blue = grid == BLUE
    if np.array_equal(input_blue, shifted):
        return False

    expected = _spec_output(grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    vacated = input_blue & ~shifted
    vacated_colors = set(map(int, expected[vacated]))
    if vacated_colors != {RED, GREY}:
        return False

    shifted_coords = [tuple(map(int, coord)) for coord in np.argwhere(shifted)]
    shifted_parities = {_chebyshev((r, c), marker) % 2 for r, c in shifted_coords}
    if shifted_parities != {0, 1}:
        return False

    background_mask = ~shifted
    visible_background_colors = set(map(int, expected[background_mask]))
    if visible_background_colors != {RED, GREY}:
        return False

    # Require the ring pattern to be visibly corner-dependent, not explainable as a uniform fill.
    edge_samples = {
        int(expected[0, min(grid.shape[1] - 1, 2)]),
        int(expected[min(grid.shape[0] - 1, 2), 0]),
        int(expected[grid.shape[0] - 1, max(0, grid.shape[1] - 3)]),
        int(expected[max(0, grid.shape[0] - 3), grid.shape[1] - 1]),
    }
    if len(edge_samples) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_candidate()
        if grid is None:
            continue
        if _generator_semantics_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
