from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
COUNTER_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400


def _fill_rect_preserving_existing(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> None:
    rr0, rr1 = sorted((int(r0), int(r1)))
    cc0, cc1 = sorted((int(c0), int(c1)))
    window = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
    window[window == BLACK] = int(color)


def _render_output(
    shape: tuple[int, int], shifted_row: int, shifted_col: int, cross_color: int = BLUE
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BLACK, dtype=int)
    out[int(shifted_row), :] = int(cross_color)
    out[:, int(shifted_col)] = int(cross_color)

    _fill_rect_preserving_existing(out, shifted_row, 0, 0, shifted_col, RED)
    _fill_rect_preserving_existing(out, shifted_row, w - 1, h - 1, shifted_col, GREEN)
    return out


def _cross_color(grid: np.ndarray) -> int:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY))
    assert len(colors) == 1, "expected exactly one non-black non-grey cross color"
    return int(colors[0])


def _cross_axes(grid: np.ndarray, color: int) -> tuple[int, int]:
    mask = grid == int(color)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    row_candidates = np.flatnonzero(row_counts == row_counts.max())
    col_candidates = np.flatnonzero(col_counts == col_counts.max())
    assert len(row_candidates) == 1, "expected a unique cross row"
    assert len(col_candidates) == 1, "expected a unique cross column"
    return int(row_candidates[0]), int(col_candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the counted cross, then use its shifted arm endpoints to define two filled corner rectangles.

    Input structure:
    - The grid has a black background, one full-row/full-column cross in a single non-black color, and 1-4 grey counter pixels at the top of the rightmost column.
    - The row and column of the cross intersect once and identify the cross center in `(r, c)` coordinates.
    - The grey pixels are separate from the cross and serve only as a count instruction.

    Rule:
    - Count the grey pixels in the rightmost column, erase them conceptually, and find the cross row `R` and column `C`.
    - Move the cross down by `K` rows and left by `K` columns to `(R + K, C - K)`, erasing the old position.
    - Keep the shifted cross in its original color, fill the rectangle between `(R + K, 0)` and `(0, C - K)` red, then fill the rectangle between `(R + K, W - 1)` and `(H - 1, C - K)` green.
    - Only black cells are recolored during rectangle filling, so blue cross pixels stay visible and earlier red cells keep priority over later green cells.

    Key insight:
    - The grey counter first determines a translation, and only after that translation do the shifted cross endpoints become the corner markers that parameterize the two filled rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    k = int(np.sum(g[:, -1] == GREY))
    color = _cross_color(g)
    row, col = _cross_axes(g, color)

    shifted_row = int(row + k)
    shifted_col = int(col - k)
    assert 0 <= shifted_row < h and 0 <= shifted_col < w, "shifted cross must stay in bounds"

    return _render_output((h, w), shifted_row, shifted_col, color)


def _position_buckets(
    h: int, w: int, k: int
) -> dict[str, list[tuple[int, int, int, int]]]:
    buckets = {"balanced": [], "red_smaller": [], "red_larger": []}
    area_cap = int(np.floor(0.60 * h * w))

    for shifted_row in range(2 * int(k), h - 1):
        for shifted_col in range(1, w - 1 - int(k)):
            red_area = int(shifted_row * shifted_col)
            green_area = int((h - 1 - shifted_row) * (w - 1 - shifted_col))
            if red_area < 4 or green_area < 4:
                continue
            if red_area + green_area > area_cap:
                continue

            imbalance = abs(red_area - green_area)
            if imbalance <= max(6, (h * w) // 10):
                buckets["balanced"].append((shifted_row, shifted_col, red_area, green_area))
            elif red_area < green_area:
                buckets["red_smaller"].append((shifted_row, shifted_col, red_area, green_area))
            else:
                buckets["red_larger"].append((shifted_row, shifted_col, red_area, green_area))

    return buckets


def _choose_shifted_position(h: int, w: int, k: int) -> tuple[int, int] | None:
    buckets = _position_buckets(h, w, k)
    available = [name for name, positions in buckets.items() if positions]
    if not available:
        return None

    bucket_name = random.choice(available)
    shifted_row, shifted_col, _red_area, _green_area = random.choice(buckets[bucket_name])
    return int(shifted_row), int(shifted_col)


def _build_input(h: int, w: int, k: int, shifted_row: int, shifted_col: int) -> np.ndarray:
    row = int(shifted_row - k)
    col = int(shifted_col + k)

    grid = np.full((int(h), int(w)), BLACK, dtype=int)
    grid[row, :] = BLUE
    grid[:, col] = BLUE
    grid[: int(k), -1] = GREY
    return grid


def _validate_generated(
    grid: np.ndarray, shifted_row: int, shifted_col: int, k: int
) -> bool:
    h, w = map(int, grid.shape)
    row = int(shifted_row - k)
    col = int(shifted_col + k)

    if row < k or col >= w - 1:
        return False
    if not np.all(grid[:k, -1] == GREY):
        return False
    if not np.all(grid[k:, -1] != GREY):
        return False
    if np.sum(grid == GREY) != k:
        return False
    if np.sum(grid == BLUE) != h + w - 1:
        return False
    if np.sum(grid[row] == BLUE) != w:
        return False
    if np.sum(grid[:, col] == BLUE) != h:
        return False

    expected = _render_output((h, w), shifted_row, shifted_col, BLUE)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    if np.sum(produced == RED) < 4 or np.sum(produced == GREEN) < 4:
        return False
    if not np.all(produced[shifted_row, :] == BLUE):
        return False
    if not np.all(produced[:, shifted_col] == BLUE):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(np.random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    k = int(np.random.randint(COUNTER_RANGE[0], COUNTER_RANGE[1] + 1))

    shifted = _choose_shifted_position(h, w, k)
    if shifted is None:
        return None

    shifted_row, shifted_col = shifted
    grid = _build_input(h, w, k, shifted_row, shifted_col)
    if not _validate_generated(grid, shifted_row, shifted_col, k):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
