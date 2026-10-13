from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_column(grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    candidates: list[int] = []
    for c in range(width):
        column = grid[:, c]
        if int(column[0]) == BLACK:
            continue
        if bool(np.all(column == column[0])):
            candidates.append(int(c))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    sep_col = candidates[0]
    assert 0 < sep_col < width - 1, "separator must split left and right regions"
    return sep_col, int(grid[0, sep_col])


def _single_run(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    start = int(indices[0])
    end = int(indices[-1])
    assert bool(np.all(mask[start : end + 1])), "foreground must form one contiguous run"
    return start, end


def _assert_alternating(values: np.ndarray, forbidden: int) -> None:
    assert values.size > 0
    assert bool(np.all(values != BLACK))
    assert bool(np.all(values != forbidden))
    if values.size == 1:
        return
    first = int(values[0])
    second = int(values[1])
    assert first != second, "data band must use at least two alternating colors"
    for i, value in enumerate(values):
        expected = first if i % 2 == 0 else second
        assert int(value) == expected, "data band must alternate with period two"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sample each right-hand data band using the aligned left-hand strip width as step size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    sep_col, sep_color = _find_separator_column(g)
    out = np.full((height, width - sep_col - 1), BLACK, dtype=int)

    active_rows = 0
    for r in range(height):
        left = g[r, :sep_col]
        right = g[r, sep_col + 1 :]

        left_run = _single_run(left != BLACK)
        right_run = _single_run(right != BLACK)
        assert (left_run is None) == (right_run is None), "rows must contain either both bands or neither"
        if left_run is None:
            continue

        active_rows += 1
        ls, le = left_run
        rs, re = right_run

        left_strip = left[ls : le + 1]
        assert bool(np.all(left_strip == left_strip[0])), "left strip must be monochrome"
        assert int(left_strip[0]) != sep_color, "left strip must not reuse the separator color"
        step = int(left_strip.size)

        data_band = right[rs : re + 1]
        _assert_alternating(data_band, sep_color)
        out[r, rs : re + 1 : step] = data_band[::step]

    assert active_rows > 0, "expected at least one aligned strip pair"
    return out


def _choose_rows(rng: np.random.Generator, height: int) -> list[int] | None:
    candidates = list(range(1, height - 1, 2))
    if len(candidates) < 2:
        return None
    count = int(rng.integers(2, min(6, len(candidates)) + 1))
    rows = rng.choice(np.array(candidates, dtype=int), size=count, replace=False)
    return sorted(int(v) for v in rows)


def _choose_band_length(rng: np.random.Generator, max_len: int, step: int) -> int:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        low, high = 2, min(max_len, max(2, step))
    elif mode == 1:
        low, high = max(3, step + 1), min(max_len, max(3, 2 * step))
    else:
        low, high = max(3, 2 * step + 1), max_len
    if low > high:
        low, high = 2, max_len
    return int(rng.integers(low, high + 1))


def _build_alternating_band(
    rng: np.random.Generator, length: int, palette: tuple[int, ...]
) -> np.ndarray:
    colors = rng.choice(np.array(palette, dtype=int), size=2, replace=False)
    phase = int(rng.integers(0, 2))
    return np.array([int(colors[(phase + i) % 2]) for i in range(length)], dtype=int)


def _validate_generated_grid(grid: np.ndarray) -> None:
    sep_col, sep_color = _find_separator_column(grid)
    active_rows = 0
    left_colors: set[int] = set()

    for r in range(grid.shape[0]):
        left = grid[r, :sep_col]
        right = grid[r, sep_col + 1 :]
        left_run = _single_run(left != BLACK)
        right_run = _single_run(right != BLACK)
        assert (left_run is None) == (right_run is None)
        if left_run is None:
            continue

        active_rows += 1
        ls, le = left_run
        rs, re = right_run
        left_strip = left[ls : le + 1]
        left_color = int(left_strip[0])
        assert bool(np.all(left_strip == left_color))
        assert left_color not in left_colors
        left_colors.add(left_color)
        _assert_alternating(right[rs : re + 1], sep_color)

    assert active_rows >= 2
    out = main(grid)
    assert out.shape == (grid.shape[0], grid.shape[1] - sep_col - 1)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(7, 18))
    left_width = int(rng.integers(4, 11))
    right_width = int(rng.integers(8, 19))

    rows = _choose_rows(rng, height)
    if rows is None:
        return None
    if len(rows) > len(PALETTE):
        return None

    grid = np.full((height, left_width + 1 + right_width), BLACK, dtype=int)
    sep_col = left_width
    grid[:, sep_col] = GREY

    left_colors = rng.choice(np.array(PALETTE, dtype=int), size=len(rows), replace=False)
    max_step = min(6, left_width)

    for idx, row in enumerate(rows):
        step = int(rng.integers(1, max_step + 1))
        left_start = int(rng.integers(0, left_width - step + 1))
        grid[row, left_start : left_start + step] = int(left_colors[idx])

        band_len = _choose_band_length(rng, right_width, step)
        band_start = int(rng.integers(0, right_width - band_len + 1))
        band = _build_alternating_band(rng, band_len, PALETTE)
        grid[row, sep_col + 1 + band_start : sep_col + 1 + band_start + band_len] = band

    _validate_generated_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
