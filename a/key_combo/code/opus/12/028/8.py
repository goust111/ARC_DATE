from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
HEIGHT_RANGE = (10, 14)
HALF_WIDTH_RANGE = (8, 11)
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BACKGROUND)
    assert coords.shape == (2, 2), "expected exactly two non-black seed pixels"

    seeds = []
    for r, c in coords:
        seeds.append((int(r), int(c), int(grid[r, c])))

    seeds.sort(key=lambda item: (item[1], item[0], item[2]))
    return seeds


def _stripe_columns(base_col: int, stride: int, limit: int) -> list[int]:
    cols: set[int] = set()
    c = int(base_col)
    while c >= 0:
        cols.add(c)
        c -= stride

    c = int(base_col) + stride
    while c < limit:
        cols.add(c)
        c += stride

    return sorted(cols)


def _build_left_half(height: int, half_width: int, seeds: list[tuple[int, int, int]]) -> np.ndarray:
    (_, col_a, color_a), (_, col_b, color_b) = seeds
    d = abs(col_a - col_b)
    stride = 2 * d

    left = np.zeros((height, half_width), dtype=int)
    for col in _stripe_columns(col_a, stride, half_width):
        left[:, col] = color_a
    for col in _stripe_columns(col_b, stride, half_width):
        left[:, col] = color_b
    return left


def _swap_colors(grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    swapped = grid.copy()
    mask_a = swapped == color_a
    mask_b = swapped == color_b
    swapped[mask_a] = color_b
    swapped[mask_b] = color_a
    return swapped


def _solve_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    assert width % 2 == 0, "expected an even number of columns"
    half_width = width // 2

    seeds = _extract_seeds(g)
    (row_a, col_a, color_a), (row_b, col_b, color_b) = seeds

    assert row_a != row_b, "seed rows must differ"
    assert col_a != col_b, "seed columns must differ"
    assert color_a != color_b, "seed colors must differ"
    assert 0 < col_a < half_width - 1, "seed A must stay away from the left-half side edges"
    assert 0 < col_b < half_width - 1, "seed B must stay away from the left-half side edges"
    assert col_a < half_width and col_b < half_width, "both seeds must lie in the left half"

    left = _build_left_half(height, half_width, seeds)
    right = np.fliplr(left)

    if abs(col_a - col_b) % 2 == 1:
        right = _swap_colors(right, color_a, color_b)

    return np.concatenate((left, right), axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the left half with seed-driven vertical stripes, then mirror it into the right half with parity-dependent color behavior.

    Input structure:
    - The grid has a black background and exactly two non-black seed pixels.
    - The two seeds have different colors, lie on different rows and columns, and both sit strictly inside the left half.
    - The overall width is even so the canvas splits into equal left and right halves.

    Rule:
    - Compute the horizontal seed distance d from their column indices.
    - In the left half, turn each seed column into a full-height vertical stripe and repeat that stripe every 2d columns in both horizontal directions, clipping to the left half.
    - Form the right half as a left-right mirror of the left half; if d is odd, swap the two seed colors in the mirrored copy, and if d is even, keep the mirrored colors unchanged.

    Key insight:
    - The same seed spacing controls two independent effects: its magnitude sets the stripe period, while its parity decides whether the mirrored half preserves or swaps the two stripe colors.
    """
    return _solve_from_grid(grid)


def _choose_seed_columns(half_width: int, want_odd_distance: bool) -> tuple[int, int] | None:
    candidates = []
    valid_cols = range(1, half_width - 1)
    for col_a in valid_cols:
        for col_b in valid_cols:
            if col_a >= col_b:
                continue
            d = col_b - col_a
            if d < 2:
                continue
            if (d % 2 == 1) != want_odd_distance:
                continue

            stride = 2 * d
            repeated = any(
                (col + stride < half_width) or (col - stride >= 0)
                for col in (col_a, col_b)
            )
            if not repeated:
                continue

            candidates.append((col_a, col_b))

    if not candidates:
        return None
    return random.choice(candidates)


def _alternate_branch() -> bool:
    global _GEN_CALL_COUNT
    want_odd_distance = bool(_GEN_CALL_COUNT % 2)
    _GEN_CALL_COUNT += 1
    return want_odd_distance


def _input_is_valid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    height, width = map(int, grid.shape)
    if max(height, width) > 30 or width % 2 != 0:
        return False

    try:
        seeds = _extract_seeds(grid)
    except AssertionError:
        return False

    half_width = width // 2
    (row_a, col_a, color_a), (row_b, col_b, color_b) = seeds
    if row_a == row_b or col_a == col_b or color_a == color_b:
        return False
    if not (0 < col_a < half_width - 1 and 0 < col_b < half_width - 1):
        return False
    if not (col_a < half_width and col_b < half_width):
        return False
    return True


def _output_is_semantically_valid(inp: np.ndarray, out: np.ndarray) -> bool:
    if out.shape != inp.shape:
        return False

    height, width = map(int, inp.shape)
    half_width = width // 2
    left = out[:, :half_width]
    right = out[:, half_width:]

    seeds = _extract_seeds(inp)
    (_, col_a, color_a), (_, col_b, color_b) = seeds
    d = abs(col_a - col_b)
    stride = 2 * d

    expected_left = _build_left_half(height, half_width, seeds)
    if not np.array_equal(left, expected_left):
        return False

    preserve = np.fliplr(left)
    swapped = _swap_colors(preserve, color_a, color_b)
    expected_right = swapped if (d % 2 == 1) else preserve

    if not np.array_equal(right, expected_right):
        return False
    if np.array_equal(preserve, swapped):
        return False

    cols_a = set(_stripe_columns(col_a, stride, half_width))
    cols_b = set(_stripe_columns(col_b, stride, half_width))
    if not cols_a or not cols_b:
        return False
    if cols_a & cols_b:
        return False

    for c in range(half_width):
        column = left[:, c]
        values = set(map(int, np.unique(column)))
        if c in cols_a:
            if values != {color_a}:
                return False
        elif c in cols_b:
            if values != {color_b}:
                return False
        else:
            if values != {BACKGROUND}:
                return False

    if np.array_equal(left, np.fliplr(left)):
        return False

    return True


def _gen_attempt(want_odd_distance: bool) -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    half_width = random.randint(*HALF_WIDTH_RANGE)
    width = 2 * half_width

    cols = _choose_seed_columns(half_width, want_odd_distance)
    if cols is None:
        return None
    col_a, col_b = cols

    row_a, row_b = random.sample(range(height), 2)
    color_a, color_b = random.sample(PALETTE, 2)

    grid = np.zeros((height, width), dtype=int)
    grid[row_a, col_a] = color_a
    grid[row_b, col_b] = color_b

    if not _input_is_valid(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _output_is_semantically_valid(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    want_odd_distance = _alternate_branch()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(want_odd_distance)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
