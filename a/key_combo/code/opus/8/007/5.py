from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

GEN_MAX_ATTEMPTS = 400
GEN_SIZE_RANGE = (10, 17)
GEN_SEGMENT_LENGTHS = (2, 3, 4)
GEN_MARGIN = 2


def _find_horizontal_segment(grid: np.ndarray, color: int) -> tuple[int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"expected color {color} to be present"
    rows = np.unique(coords[:, 0])
    assert rows.size == 1, f"expected a single horizontal segment of color {color}"
    row = int(rows[0])
    cols = np.sort(coords[:, 1])
    assert np.all(np.diff(cols) == 1), f"expected contiguous columns for color {color}"
    return row, int(cols[0]), int(cols[-1])


def _find_vertical_segment(grid: np.ndarray, color: int) -> tuple[int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"expected color {color} to be present"
    cols = np.unique(coords[:, 1])
    assert cols.size == 1, f"expected a single vertical segment of color {color}"
    col = int(cols[0])
    rows = np.sort(coords[:, 0])
    assert np.all(np.diff(rows) == 1), f"expected contiguous rows for color {color}"
    return col, int(rows[0]), int(rows[-1])


def _trace_bouncing_ray(out: np.ndarray, start: tuple[int, int], dr: int, dc: int) -> None:
    height, width = out.shape
    r, c = start
    step_limit = 4 * (height + width)

    for _ in range(step_limit):
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= height:
            dr *= -1
            nr = r + dr
        if nc < 0 or nc >= width:
            dc *= -1
            nc = c + dc
        r, c = nr, nc
        if (r, c) == start:
            break
        out[r, c] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a blue row and a red column, mark their crossing, and launch four bouncing diagonal rays from it.

    Input structure:
    - The input is a black background with exactly one blue horizontal segment and one red vertical segment.
    - The blue cells occupy one contiguous run on a single row, and the red cells occupy one contiguous run on a single column.
    - The row/column crossing cell is black and lies at least two cells away from every border.

    Rule:
    - Extend the blue segment across its entire row and the red segment across its entire column.
    - Recolor their crossing cell yellow.
    - From that yellow crossing, trace four diagonal rays in the four sign combinations of row/column motion.
    - Whenever a ray would step outside the grid, reverse the offending direction component, continue the walk, and paint every visited cell yellow until the ray returns to the crossing or exceeds a fixed safety limit.
    - After all rays are drawn, replace every remaining black cell with teal.

    Key insight:
    - The ray source is not given directly; it is the geometric intersection created by first extending the two short orthogonal seed segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blue_row, blue_c0, blue_c1 = _find_horizontal_segment(g, BLUE)
    red_col, red_r0, red_r1 = _find_vertical_segment(g, RED)

    assert 2 <= blue_c1 - blue_c0 + 1 <= 4, "blue segment length must be 2..4"
    assert 2 <= red_r1 - red_r0 + 1 <= 4, "red segment length must be 2..4"
    assert int(g[blue_row, red_col]) == BLACK, "expected the crossing cell to be black"

    out = g.copy()
    out[blue_row, :] = BLUE
    out[:, red_col] = RED
    out[blue_row, red_col] = YELLOW

    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        _trace_bouncing_ray(out, (blue_row, red_col), dr, dc)

    out[out == BLACK] = TEAL
    return out


def _valid_horizontal_intervals(width: int, blocked_col: int, length: int) -> list[tuple[int, int]]:
    return [
        (c0, c0 + length - 1)
        for c0 in range(width - length + 1)
        if not (c0 <= blocked_col <= c0 + length - 1)
    ]


def _valid_vertical_intervals(height: int, blocked_row: int, length: int) -> list[tuple[int, int]]:
    return [
        (r0, r0 + length - 1)
        for r0 in range(height - length + 1)
        if not (r0 <= blocked_row <= r0 + length - 1)
    ]


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_SIZE_RANGE)
    width = random.randint(*GEN_SIZE_RANGE)
    grid = np.zeros((height, width), dtype=int)

    blue_row = random.randint(GEN_MARGIN, height - GEN_MARGIN - 1)
    red_col = random.randint(GEN_MARGIN, width - GEN_MARGIN - 1)

    blue_len = random.choice(GEN_SEGMENT_LENGTHS)
    red_len = random.choice(GEN_SEGMENT_LENGTHS)

    blue_options = _valid_horizontal_intervals(width, red_col, blue_len)
    red_options = _valid_vertical_intervals(height, blue_row, red_len)
    if not blue_options or not red_options:
        return None

    blue_c0, blue_c1 = random.choice(blue_options)
    red_r0, red_r1 = random.choice(red_options)

    grid[blue_row, blue_c0 : blue_c1 + 1] = BLUE
    grid[red_r0 : red_r1 + 1, red_col] = RED

    if int(grid[blue_row, red_col]) != BLACK:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
