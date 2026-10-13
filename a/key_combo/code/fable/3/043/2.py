from __future__ import annotations

import numpy as np


MAX_ATTEMPTS = 500
COLOR_POOL = np.arange(10, dtype=int)


def _collapse_rows(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]
    widths: list[int] = []
    run_start = 0
    for r in range(1, g.shape[0]):
        if not np.array_equal(g[r], g[r - 1]):
            widths.append(r - run_start)
            starts.append(r)
            run_start = r
    widths.append(g.shape[0] - run_start)
    return g[starts, :].copy(), np.array(widths, dtype=int)


def _collapse_cols(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]
    widths: list[int] = []
    run_start = 0
    for c in range(1, g.shape[1]):
        if not np.array_equal(g[:, c], g[:, c - 1]):
            widths.append(c - run_start)
            starts.append(c)
            run_start = c
    widths.append(g.shape[1] - run_start)
    return g[:, starts].copy(), np.array(widths, dtype=int)


def _expand(kernel: np.ndarray, row_widths: np.ndarray, col_widths: np.ndarray) -> np.ndarray:
    assert kernel.shape == (len(row_widths), len(col_widths))
    assert np.all(row_widths > 0) and np.all(col_widths > 0)
    return np.repeat(np.repeat(kernel, row_widths, axis=0), col_widths, axis=1)


def _adjacent_rows_and_cols_are_distinct(kernel: np.ndarray) -> bool:
    for r in range(1, kernel.shape[0]):
        if np.array_equal(kernel[r], kernel[r - 1]):
            return False
    for c in range(1, kernel.shape[1]):
        if np.array_equal(kernel[:, c], kernel[:, c - 1]):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Fold a stretched square kernel and reopen it with row and column stretch profiles exchanged.

    Input structure:
    - The grid is made from one k by k color kernel.
    - Kernel row i has been copied into a horizontal band of width r_i, and
      kernel column j has been copied into a vertical band of width c_j.
    - Adjacent kernel rows and adjacent kernel columns are visibly different, so
      adjacent equal bands determine the two run-length profiles uniquely.

    Rule:
    - Collapse adjacent identical rows and adjacent identical columns to recover
      the square kernel, the row-band widths, and the column-band widths.
    - Re-expand the same kernel without transposing or mirroring it.
    - In the output, kernel row i is copied c_i times and kernel column j is
      copied r_j times.

    Key insight:
    - The repeated bands are not disposable padding; their run lengths are data.
      The horizontal and vertical stretch profiles are swapped while the kernel's
      orientation stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_collapsed, row_widths = _collapse_rows(g)
    kernel, col_widths = _collapse_cols(row_collapsed)
    k_rows, k_cols = kernel.shape
    assert k_rows == k_cols, "collapsed core must be square"
    assert len(row_widths) == k_rows and len(col_widths) == k_cols
    assert _adjacent_rows_and_cols_are_distinct(kernel), "kernel must collapse uniquely"

    out = _expand(kernel, col_widths, row_widths)
    assert max(out.shape) <= 30, "ARC output dimensions must not exceed 30"
    return out


def _sample_widths(rng: np.random.Generator, k: int, max_width: int) -> np.ndarray:
    for _ in range(100):
        widths = rng.integers(1, max_width + 1, size=k, dtype=int)
        if np.max(widths) > 1 and len(set(map(int, widths))) >= 2:
            return widths
    raise RuntimeError("failed to sample non-degenerate widths")


def _sample_width_pair(rng: np.random.Generator, k: int) -> tuple[np.ndarray, np.ndarray]:
    max_width = max(2, min(5, 30 // k))
    for _ in range(200):
        row_widths = _sample_widths(rng, k, max_width)
        col_widths = _sample_widths(rng, k, max_width)
        if int(np.sum(row_widths)) == int(np.sum(col_widths)):
            continue
        if np.array_equal(row_widths, col_widths):
            continue
        return row_widths, col_widths
    raise RuntimeError("failed to sample distinct stretch profiles")


def _sample_kernel(rng: np.random.Generator, k: int) -> np.ndarray | None:
    palette_size = int(rng.integers(3, min(6, len(COLOR_POOL)) + 1))
    palette = rng.choice(COLOR_POOL, size=palette_size, replace=False)
    for _ in range(300):
        kernel = rng.choice(palette, size=(k, k), replace=True).astype(int)
        if len(np.unique(kernel)) < 3:
            continue
        if not _adjacent_rows_and_cols_are_distinct(kernel):
            continue
        if np.array_equal(kernel, kernel.T):
            continue
        return kernel
    return None


def _is_semantic_example(
    inp: np.ndarray,
    expected: np.ndarray,
    kernel: np.ndarray,
    row_widths: np.ndarray,
    col_widths: np.ndarray,
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if inp.shape[::-1] != expected.shape:
        return False
    if not np.array_equal(_expand(kernel, row_widths, col_widths), inp):
        return False
    if not np.array_equal(_expand(kernel, col_widths, row_widths), expected):
        return False
    if np.array_equal(expected, inp.T):
        return False

    row_collapsed, found_rows = _collapse_rows(inp)
    found_kernel, found_cols = _collapse_cols(row_collapsed)
    if not np.array_equal(found_kernel, kernel):
        return False
    if not np.array_equal(found_rows, row_widths):
        return False
    if not np.array_equal(found_cols, col_widths):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.integers(3, 8))
    kernel = _sample_kernel(rng, k)
    if kernel is None:
        return None

    row_widths, col_widths = _sample_width_pair(rng, k)
    inp = _expand(kernel, row_widths, col_widths)
    expected = _expand(kernel, col_widths, row_widths)

    if _is_semantic_example(inp, expected, kernel, row_widths, col_widths):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stretch-profile exchange input")
