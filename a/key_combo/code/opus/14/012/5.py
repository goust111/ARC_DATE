import hashlib
from typing import Literal

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GEN_RETRY_LIMIT = 400
GRID_MIN = 10
GRID_MAX = 16
PERIOD_CHOICES = (2, 3, 4)

_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _seed_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one seed of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _paint_row_stripe(out: np.ndarray, row: int, color: int) -> None:
    out[row, :] = color
    if color == BLUE:
        out[row, 0::2] = BLACK
    else:
        out[row, 1::2] = BLACK


def _paint_col_stripe(out: np.ndarray, col: int, color: int) -> None:
    out[:, col] = color
    if color == BLUE:
        out[0::2, col] = BLACK
    else:
        out[1::2, col] = BLACK


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, RED}, "expected only black, blue, and red"

    blue_r, blue_c = _seed_coord(g, BLUE)
    red_r, red_c = _seed_coord(g, RED)
    h, w = map(int, g.shape)

    out = np.zeros_like(g)
    row_stripes = blue_c in (0, w - 1) or red_c in (0, w - 1)

    if row_stripes:
        stride_seed_gap = abs(red_r - blue_r)
        assert stride_seed_gap > 0, "row stripes require seeds on different rows"
        stride = 2 * stride_seed_gap
        for row in range(blue_r, h, stride):
            _paint_row_stripe(out, row, BLUE)
        for row in range(red_r, h, stride):
            _paint_row_stripe(out, row, RED)
        return out

    assert blue_r == red_r, "column stripes require the two seeds to share one row"
    stride_seed_gap = abs(red_c - blue_c)
    assert stride_seed_gap > 0, "column stripes require seeds on different columns"
    stride = 2 * stride_seed_gap
    for col in range(blue_c, w, stride):
        _paint_col_stripe(out, col, BLUE)
    for col in range(red_c, w, stride):
        _paint_col_stripe(out, col, RED)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn two blue/red seeds into periodic stripes with color-dependent alternating holes.

    Input structure:
    - The grid is black except for exactly two seed pixels: one blue (1) and one red (2).
    - If either seed touches the left or right border, the seeds define full row stripes.
    - Otherwise the seeds lie on the same row and define full column stripes.

    Rule:
    - Use the two seed rows or columns as the first two stripes and repeat each color every
      2s cells along the perpendicular axis, where s is the seed separation.
    - After drawing each full stripe, erase alternating cells along the stripe itself.
    - Blue stripes erase positions 1, 3, 5, ... from the stripe start edge, while red
      stripes erase positions 2, 4, 6, ... from the stripe start edge; seeds are not preserved specially.

    Key insight:
    - The seed separation encodes the stripe period, while the seed color independently
      encodes the hole phase, so the blue and red stripes keep the same spacing but opposite parity.
    """
    return _transform(grid)


def _build_expected(
    shape: tuple[int, int],
    orientation: Literal["rows", "cols"],
    first_index: int,
    separation: int,
    blue_first: bool,
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    if orientation == "rows":
        blue_row = first_index if blue_first else first_index + separation
        red_row = first_index + separation if blue_first else first_index
        stride = 2 * separation
        for row in range(blue_row, h, stride):
            _paint_row_stripe(out, row, BLUE)
        for row in range(red_row, h, stride):
            _paint_row_stripe(out, row, RED)
        return out

    blue_col = first_index if blue_first else first_index + separation
    red_col = first_index + separation if blue_first else first_index
    stride = 2 * separation
    for col in range(blue_col, w, stride):
        _paint_col_stripe(out, col, BLUE)
    for col in range(red_col, w, stride):
        _paint_col_stripe(out, col, RED)
    return out


def _row_case(separation: int) -> tuple[np.ndarray, np.ndarray]:
    min_height = max(GRID_MIN, 3 * separation + 2)
    height = int(np.random.randint(min_height, GRID_MAX + 1))
    width = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    first_index_max = min(separation - 1, height - 3 * separation - 1)
    first_index = int(np.random.randint(0, first_index_max + 1))
    blue_first = bool(np.random.randint(0, 2))

    grid = np.zeros((height, width), dtype=int)
    blue_row = first_index if blue_first else first_index + separation
    red_row = first_index + separation if blue_first else first_index

    # Fix the invisible seed columns so the generated dataset does not contain
    # multiple distinct inputs that collapse to the same output.
    grid[blue_row, 0] = BLUE
    grid[red_row, 1] = RED
    expected = _build_expected(grid.shape, "rows", first_index, separation, blue_first)
    return grid, expected


def _col_case(separation: int) -> tuple[np.ndarray, np.ndarray]:
    min_width = max(GRID_MIN, 3 * separation + 2)
    width = int(np.random.randint(min_width, GRID_MAX + 1))
    height = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    first_index_max = min(separation, width - 3 * separation - 1)
    first_index = int(np.random.randint(1, first_index_max + 1))
    blue_first = bool(np.random.randint(0, 2))

    grid = np.zeros((height, width), dtype=int)
    seed_row = 1
    blue_col = first_index if blue_first else first_index + separation
    red_col = first_index + separation if blue_first else first_index

    # Fix the invisible seed row for the same non-collapse reason as in _row_case().
    grid[seed_row, blue_col] = BLUE
    grid[seed_row, red_col] = RED
    expected = _build_expected(grid.shape, "cols", first_index, separation, blue_first)
    return grid, expected


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    orientation: Literal["rows", "cols"],
    separation: int,
    first_index: int,
    blue_first: bool,
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.count_nonzero(grid == BLUE) != 1 or np.count_nonzero(grid == RED) != 1:
        return False
    if np.any((grid != BLACK) & (grid != BLUE) & (grid != RED)):
        return False
    if np.array_equal(grid, expected):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    blue_cells = np.argwhere(expected == BLUE)
    red_cells = np.argwhere(expected == RED)
    if blue_cells.size == 0 or red_cells.size == 0:
        return False

    if orientation == "rows":
        blue_rows = set(range(first_index if blue_first else first_index + separation, grid.shape[0], 2 * separation))
        red_rows = set(range(first_index + separation if blue_first else first_index, grid.shape[0], 2 * separation))
        if {int(r) for r, _c in blue_cells} != blue_rows:
            return False
        if {int(r) for r, _c in red_cells} != red_rows:
            return False
        for row in blue_rows:
            if not np.array_equal(expected[row], np.array([BLACK if c % 2 == 0 else BLUE for c in range(grid.shape[1])], dtype=int)):
                return False
        for row in red_rows:
            if not np.array_equal(expected[row], np.array([RED if c % 2 == 0 else BLACK for c in range(grid.shape[1])], dtype=int)):
                return False
        if len(blue_rows) + len(red_rows) < 3:
            return False
        return True

    blue_cols = set(range(first_index if blue_first else first_index + separation, grid.shape[1], 2 * separation))
    red_cols = set(range(first_index + separation if blue_first else first_index, grid.shape[1], 2 * separation))
    if {int(c) for _r, c in blue_cells} != blue_cols:
        return False
    if {int(c) for _r, c in red_cells} != red_cols:
        return False
    for col in blue_cols:
        if not np.array_equal(expected[:, col], np.array([BLACK if r % 2 == 0 else BLUE for r in range(grid.shape[0])], dtype=int)):
            return False
    for col in red_cols:
        if not np.array_equal(expected[:, col], np.array([RED if r % 2 == 0 else BLACK for r in range(grid.shape[0])], dtype=int)):
            return False
    if len(blue_cols) + len(red_cols) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        orientation = "rows" if bool(np.random.randint(0, 2)) else "cols"
        separation = int(np.random.choice(PERIOD_CHOICES))
        if orientation == "rows":
            grid, expected = _row_case(separation)
            blue_r, _blue_c = _seed_coord(grid, BLUE)
            red_r, _red_c = _seed_coord(grid, RED)
            first_index = min(blue_r, red_r)
            blue_first = blue_r < red_r
        else:
            grid, expected = _col_case(separation)
            _blue_r, blue_c = _seed_coord(grid, BLUE)
            _red_r, red_c = _seed_coord(grid, RED)
            first_index = min(blue_c, red_c)
            blue_first = blue_c < red_c

        if not _validate_case(grid, expected, orientation, separation, first_index, blue_first):
            continue

        out_sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if out_sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid unique input for puzzle 5")
