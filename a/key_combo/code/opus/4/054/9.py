from __future__ import annotations

import numpy as np


BG = 0
SEP = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
_GEN_RNG = np.random.default_rng()


def _runs(mask: np.ndarray, value: bool = False) -> list[tuple[int, int]]:
    indices = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag) == bool(value)]
    runs: list[tuple[int, int]] = []
    start = None
    prev = None
    for idx in indices:
        if start is None:
            start = prev = int(idx)
        elif idx == prev + 1:
            prev = int(idx)
        else:
            runs.append((start, prev))
            start = prev = int(idx)
    if start is not None:
        runs.append((start, prev))
    return runs


def _read_color_bar(row: np.ndarray) -> list[int]:
    seq: list[int] = []
    for value in map(int, row):
        if value == BG:
            break
        assert value != SEP, "top strip cannot contain separator color"
        seq.append(value)
    assert len(seq) >= 2, "expected a nontrivial top strip"
    assert np.all(np.asarray(row[len(seq) :], dtype=int) == BG), "top strip must be contiguous"
    return seq


def _lattice_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    lattice = np.asarray(grid[2:, :], dtype=int)
    assert lattice.size > 0, "expected a lattice below the separator row"

    full_sep_rows = np.all(lattice == SEP, axis=1)
    full_sep_cols = np.all(lattice == SEP, axis=0)
    row_runs = _runs(full_sep_rows, value=False)
    col_runs = _runs(full_sep_cols, value=False)
    assert len(row_runs) >= 2 and len(col_runs) >= 2, "expected at least a 2x2 logical grid"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1 and len(col_widths) == 1, "expected uniform cell sizes"
    return row_runs, col_runs


def _cell_color_matrix(seq: list[int], rows: int, cols: int, snake: bool) -> np.ndarray:
    mat = np.zeros((rows, cols), dtype=int)
    idx = 0
    for br in range(rows):
        if snake and br % 2 == 1:
            order = range(cols - 1, -1, -1)
        else:
            order = range(cols)
        for bc in order:
            mat[br, bc] = int(seq[idx % len(seq)])
            idx += 1
    return mat


def _render(grid: np.ndarray, seq: list[int], snake: bool = True) -> np.ndarray:
    row_runs, col_runs = _lattice_bands(grid)
    out = np.asarray(grid, dtype=int).copy()
    colors = _cell_color_matrix(seq, len(row_runs), len(col_runs), snake=snake)
    for br, (r0, r1) in enumerate(row_runs):
        rr0 = r0 + 2
        rr1 = r1 + 2
        for bc, (c0, c1) in enumerate(col_runs):
            out[rr0 : rr1 + 1, c0 : c1 + 1] = int(colors[br, bc])
    return out


def _has_shorter_period(seq: list[int]) -> bool:
    k = len(seq)
    for step in range(1, k):
        if k % step == 0 and all(seq[i] == seq[i % step] for i in range(k)):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the logical cells of a separator grid by cycling the top strip along a snake traversal.

    Input structure:
    - Row 0 starts with a contiguous non-black, non-gray color strip and then black cells.
    - Row 1 is fully black.
    - From row 2 downward, full gray separator lines split the area into equal-sized logical cells whose interiors are initially non-separator.

    Rule:
    - Read the row-0 strip as an ordered color sequence C of length k.
    - Traverse logical cells in snake order: left-to-right on the first cell row, right-to-left on the second, then alternating.
    - Fill the i-th visited logical cell uniformly with C[i mod k], leaving the gray separators, the top strip, and row 1 unchanged.

    Key insight:
    - The gray lattice defines the objects, the top strip defines the repeating period, and the snake traversal determines which cell receives each period position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 5 and g.shape[1] >= 5, "grid is too small for the described structure"
    assert np.all(g[1] == BG), "row 1 must remain black"

    seq = _read_color_bar(g[0])
    return _render(g, seq, snake=True)


def _build_lattice(
    cell_rows: int,
    cell_cols: int,
    cell_size: int,
    outer_border: bool,
) -> np.ndarray:
    lattice_h = cell_rows * cell_size + (cell_rows + 1 if outer_border else cell_rows - 1)
    lattice_w = cell_cols * cell_size + (cell_cols + 1 if outer_border else cell_cols - 1)
    lattice = np.zeros((lattice_h, lattice_w), dtype=int)

    if outer_border:
        lattice[0, :] = SEP
        lattice[-1, :] = SEP
        lattice[:, 0] = SEP
        lattice[:, -1] = SEP

    r = 1 if outer_border else 0
    for br in range(cell_rows):
        r += cell_size
        if br < cell_rows - 1:
            lattice[r, :] = SEP
            r += 1

    c = 1 if outer_border else 0
    for bc in range(cell_cols):
        c += cell_size
        if bc < cell_cols - 1:
            lattice[:, c] = SEP
            c += 1

    return lattice


def _sample_bar_sequence(length: int) -> list[int]:
    distinct_count = int(_GEN_RNG.integers(2, min(4, length) + 1))
    palette = [int(x) for x in _GEN_RNG.choice(np.asarray(BAR_COLORS, dtype=int), size=distinct_count, replace=False)]
    while True:
        seq = [palette[0], palette[1]]
        while len(seq) < length:
            seq.append(int(_GEN_RNG.choice(np.asarray(palette, dtype=int))))
        _GEN_RNG.shuffle(seq)
        if len(set(seq)) < 2:
            continue
        if not all(color in seq for color in palette):
            continue
        if _has_shorter_period(seq):
            continue
        return [int(v) for v in seq]


def _sample_spec() -> tuple[int, int, int, bool]:
    specs = []
    for cell_rows in range(2, 7):
        for cell_cols in range(2, 7):
            for cell_size in (2, 3):
                for outer_border in (False, True):
                    lattice_h = cell_rows * cell_size + (cell_rows + 1 if outer_border else cell_rows - 1)
                    lattice_w = cell_cols * cell_size + (cell_cols + 1 if outer_border else cell_cols - 1)
                    if 2 + lattice_h <= 30 and lattice_w <= 30:
                        specs.append((cell_rows, cell_cols, cell_size, outer_border))
    return specs[int(_GEN_RNG.integers(len(specs)))]


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        seq = _read_color_bar(grid[0])
        row_runs, col_runs = _lattice_bands(grid)
        snake = _cell_color_matrix(seq, len(row_runs), len(col_runs), snake=True)
        row_major = _cell_color_matrix(seq, len(row_runs), len(col_runs), snake=False)
        out = main(grid)
    except Exception:
        return False

    if _has_shorter_period(seq):
        return False
    if len(row_runs) * len(col_runs) <= len(seq):
        return False
    if np.array_equal(snake, row_major):
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out, _render(grid, seq, snake=True)):
        return False
    if np.array_equal(out, _render(grid, seq, snake=False)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        cell_rows, cell_cols, cell_size, outer_border = _sample_spec()
        lattice = _build_lattice(cell_rows, cell_cols, cell_size, outer_border)
        height = 2 + lattice.shape[0]
        width = lattice.shape[1]

        max_k = min(6, width - 1, cell_rows * cell_cols - 1)
        if max_k < 2:
            continue
        bar_len = int(_GEN_RNG.integers(2, max_k + 1))
        seq = _sample_bar_sequence(bar_len)

        grid = np.zeros((height, width), dtype=int)
        grid[2:, :] = lattice
        grid[0, :bar_len] = np.asarray(seq, dtype=int)

        if _semantic_checks(grid):
            return grid

    raise RuntimeError("failed to generate a valid snake-period lattice input")
