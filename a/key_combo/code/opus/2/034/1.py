from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

FLOW_COUNT_RANGE = (3, 6)
RED_COUNT_RANGE = (3, 8)
GRID_HEIGHT_MIN = 15
GRID_WIDTH_MIN = 16
MAX_ATTEMPTS = 300

_GEN_CALL_COUNT = 0


def _trace_path(start_row: int, red_cols: list[int], shape: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    """Follow one left-to-right flow and return its visited cells plus deflection count."""
    height, width = map(int, shape)
    row = int(start_row)
    red_set = set(int(c) for c in red_cols)
    path: list[tuple[int, int]] = []
    deflections = 0

    for col in range(width):
        assert 0 <= row < height, "flow row left the grid"
        path.append((row, col))
        if col in red_set:
            row += 1
            deflections += 1
            assert 0 <= row < height, "deflection pushed a flow out of bounds"
            path.append((row, col))

    return path, deflections


def _extract_markers(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    red_mask = g == RED
    teal_mask = g == TEAL
    other_mask = (g != BLACK) & ~red_mask & ~teal_mask

    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(other_mask), "input may only contain black, red, and teal"
    assert np.any(red_mask), "expected at least one red top-edge deflector"
    assert np.any(teal_mask), "expected at least one teal left-edge starter"
    assert np.all(np.argwhere(red_mask)[:, 0] == 0), "red markers must sit on the top edge"
    assert np.all(np.argwhere(teal_mask)[:, 1] == 0), "teal markers must sit on the left edge"

    nonblack = g != BLACK
    allowed = red_mask | teal_mask
    assert np.array_equal(nonblack, allowed), "all non-black cells must be top red or left-edge teal markers"

    start_rows = sorted(int(r) for r in np.flatnonzero(teal_mask[:, 0]))
    red_cols = sorted(int(c) for c in np.flatnonzero(red_mask[0]))
    assert all(b - a >= 2 for a, b in zip(start_rows, start_rows[1:])), "starter rows must stay separated"
    return start_rows, red_cols


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each left-edge teal flow across the grid and recolor whole paths by deflection-count parity.

    Input structure:
    - The grid is black except for teal starter pixels on the left edge and red deflector pixels on the top edge.
    - Each teal pixel starts a separate left-to-right flow, and the starter rows are spaced so the flows never overlap.
    - Each red pixel marks a column where any passing flow is pushed one cell downward.

    Rule:
    - Trace every teal flow from column 0 to the right edge, marking every visited cell.
    - Whenever a flow reaches a column containing a red top-edge marker, mark the current cell, step the flow down by one cell in that same column, and continue rightward from the lowered row.
    - Recolor the entire traced path yellow if that flow was deflected an odd number of times; otherwise keep the path teal. Leave all red markers and black background unchanged.

    Key insight:
    - The visible output color is determined by an emergent path attribute, namely the parity of how many top-edge deflectors the flow encountered while being drawn.
    """
    g = np.asarray(grid, dtype=int)
    start_rows, red_cols = _extract_markers(g)

    out = g.copy()
    painted = np.zeros_like(g, dtype=bool)
    for start_row in start_rows:
        path, deflections = _trace_path(start_row, red_cols, g.shape)
        color = YELLOW if deflections % 2 else TEAL
        for row, col in path:
            if g[row, col] == RED:
                continue
            assert not painted[row, col], "flows must not overlap"
            painted[row, col] = True
            out[row, col] = color

    return out


def _sample_spaced_rows(rng: np.random.Generator, count: int, max_row: int) -> list[int]:
    """Choose sorted row indices in [1, max_row] with pairwise gaps of at least 2."""
    assert max_row >= 2 * count - 1, "not enough vertical room for separated starters"
    base = np.sort(rng.choice(np.arange(1, max_row - count + 2), size=count, replace=False))
    return [int(v) for v in (base + np.arange(count))]


def _paths_do_not_overlap(start_rows: list[int], red_cols: list[int], shape: tuple[int, int]) -> bool:
    seen: set[tuple[int, int]] = set()
    for start_row in start_rows:
        path, _ = _trace_path(start_row, red_cols, shape)
        unique_path = set(path)
        if len(unique_path) != len(path):
            return False
        if seen & unique_path:
            return False
        seen |= unique_path
    return True


def _gen_attempt(rng: np.random.Generator, desired_parity: int) -> np.ndarray | None:
    red_options = [n for n in range(RED_COUNT_RANGE[0], RED_COUNT_RANGE[1] + 1) if n % 2 == desired_parity]
    flow_count = int(rng.integers(FLOW_COUNT_RANGE[0], FLOW_COUNT_RANGE[1] + 1))
    red_count = int(rng.choice(red_options))

    min_height = max(GRID_HEIGHT_MIN, red_count + 2 * flow_count + 1)
    max_height = min(30, min_height + 6)
    height = int(rng.integers(min_height, max_height + 1))

    min_width = max(GRID_WIDTH_MIN, red_count + 5)
    max_width = min(30, min_width + 8)
    width = int(rng.integers(min_width, max_width + 1))

    max_start_row = height - red_count - 2
    if max_start_row < 2 * flow_count - 1:
        return None

    start_rows = _sample_spaced_rows(rng, flow_count, max_start_row)
    red_cols = sorted(int(c) for c in rng.choice(np.arange(1, width), size=red_count, replace=False))
    if not _paths_do_not_overlap(start_rows, red_cols, (height, width)):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, red_cols] = RED
    grid[start_rows, 0] = TEAL
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    desired_parity = _GEN_CALL_COUNT % 2
    _GEN_CALL_COUNT += 1
    rng = np.random.default_rng()

    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng, desired_parity)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(out, grid):
            return grid

    raise RuntimeError("failed to generate a valid deflection-parity ARC input")
