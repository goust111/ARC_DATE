from __future__ import annotations

from collections import Counter
import random

import numpy as np


LATTICE_COLOR = 4
BG_COLOR = 0
SPRITE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _uniform_line_positions(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == int(color))]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == int(color))]
    return rows, cols


def _gap_slices(lines: list[int]) -> list[tuple[int, int]]:
    return [
        (int(lines[i]) + 1, int(lines[i + 1]))
        for i in range(len(lines) - 1)
        if int(lines[i + 1]) > int(lines[i]) + 1
    ]


def _extract_lattice_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    row_lines, col_lines = _uniform_line_positions(grid, LATTICE_COLOR)
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "expected visible lattice lines"

    row_slices = _gap_slices(row_lines)
    col_slices = _gap_slices(col_lines)
    assert row_slices and col_slices, "expected at least one lattice row and column"

    cell_hs = {rb - ra for ra, rb in row_slices}
    cell_ws = {cb - ca for ca, cb in col_slices}
    assert len(cell_hs) == 1 and len(cell_ws) == 1, "cells must have equal size"

    cells = []
    for ra, rb in row_slices:
        row = []
        for ca, cb in col_slices:
            cell = np.array(grid[ra:rb, ca:cb], dtype=int, copy=True)
            assert cell.size > 0
            row.append(cell)
        cells.append(row)
    return row_slices, col_slices, cells


def _row_outlier(row_cells: list[np.ndarray]) -> np.ndarray:
    sigs = [cell.tobytes() for cell in row_cells]
    counts = Counter(sigs)
    assert len(counts) == 2, "each row must contain one majority pattern and one outlier pattern"
    majorities = [sig for sig, cnt in counts.items() if int(cnt) == len(row_cells) - 1]
    uniques = [sig for sig, cnt in counts.items() if int(cnt) == 1]
    assert len(majorities) == 1 and len(uniques) == 1, "row must have exactly one unique cell"
    idx = sigs.index(uniques[0])
    return row_cells[idx]


def _build_seed(grid: np.ndarray) -> np.ndarray:
    _row_slices, _col_slices, cells = _extract_lattice_cells(grid)
    outliers = [_row_outlier(row_cells) for row_cells in cells]
    assert len({cell.tobytes() for cell in outliers}) == len(outliers), "row outliers must all differ"
    return np.vstack(outliers)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each lattice row's odd cell, stack those inner sprites vertically, then tile the composite seed across the output width.

    Input structure:
    - Black background with yellow lattice lines that partition the grid into equal square cells.
    - Every cell interior is a small colored sprite using non-black, non-yellow colors.
    - In each lattice row, all but one cell share the same interior sprite; one cell is the unique outlier for that row.

    Rule:
    - Ignore the yellow lattice lines and compare cell interiors row by row.
    - In each row, find the sprite pattern that appears exactly once and crop that cell interior.
    - Stack the row outliers from top to bottom without gaps to form one composite seed whose width equals one cell interior width.
    - Repeat that seed horizontally until the output reaches the original input width, truncating the final repetition on the right if needed.

    Key insight:
    - The output seed is not present as a single object in the input; it must be constructed by row-wise odd-one-out detection before the horizontal tiling step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    seed = _build_seed(g)
    out_h, seed_w = seed.shape
    out_w = int(g.shape[1])
    cols = np.arange(out_w) % int(seed_w)
    return seed[:, cols].copy()


def _random_pattern(size: int) -> np.ndarray:
    n_colors = random.randint(2, min(4, len(SPRITE_COLORS)))
    palette = random.sample(SPRITE_COLORS, n_colors)
    arr = np.array(random.choices(palette, k=size * size), dtype=int).reshape(size, size)

    if len(set(map(int, arr.flat))) < 2:
        arr[0, 0] = palette[0]
        arr[-1, -1] = palette[1]

    if np.all(arr == arr[0, 0]):
        arr[0, 0] = palette[0]
        arr[0, -1] = palette[1]
    return arr


def _make_distinct_patterns(count: int, size: int) -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()
    while len(patterns) < count:
        candidate = _random_pattern(size)
        sig = candidate.tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        patterns.append(candidate)
    return patterns


def _nontrivial_derangement(n: int) -> list[int]:
    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def _render_lattice(rows: list[list[np.ndarray]]) -> np.ndarray:
    n_rows = len(rows)
    n_cols = len(rows[0])
    cell_size = int(rows[0][0].shape[0])
    height = n_rows * (cell_size + 1) + 1
    width = n_cols * (cell_size + 1) + 1
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for r in range(0, height, cell_size + 1):
        grid[r, :] = LATTICE_COLOR
    for c in range(0, width, cell_size + 1):
        grid[:, c] = LATTICE_COLOR
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ra = 1 + r * (cell_size + 1)
            ca = 1 + c * (cell_size + 1)
            grid[ra : ra + cell_size, ca : ca + cell_size] = cell
    return grid


def _expected_output(outliers: list[np.ndarray], width: int) -> np.ndarray:
    seed = np.vstack(outliers)
    cols = np.arange(width) % int(seed.shape[1])
    return seed[:, cols].copy()


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, odd_cols: list[int], bases: list[np.ndarray]) -> bool:
    try:
        row_slices, col_slices, cells = _extract_lattice_cells(inp)
    except AssertionError:
        return False

    if len(row_slices) < 3 or len(col_slices) < 3:
        return False

    if len(set(odd_cols)) < 2:
        return False

    cell_size = row_slices[0][1] - row_slices[0][0]
    if cell_size != col_slices[0][1] - col_slices[0][0]:
        return False

    outliers = []
    global_counts = Counter(cell.tobytes() for row in cells for cell in row)
    for row_cells in cells:
        sigs = [cell.tobytes() for cell in row_cells]
        counts = Counter(sigs)
        if sorted(counts.values()) != [1, len(row_cells) - 1]:
            return False
        outlier = _row_outlier(row_cells)
        outliers.append(outlier)

    if len({cell.tobytes() for cell in outliers}) != len(outliers):
        return False

    # Anti-shortcut: no pattern should be globally unique; row-wise comparison must matter.
    if any(int(global_counts[cell.tobytes()]) == 1 for cell in outliers):
        return False

    # Anti-shortcut: the majority sprites vary across rows, so one fixed template is insufficient.
    if len({cell.tobytes() for cell in bases}) != len(bases):
        return False

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return False

    # The tiling step must do work: output must be wider than a single seed copy.
    if expected.shape[1] <= cell_size:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice([2, 3, 4])
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)

    grid_h = n_rows * (cell_size + 1) + 1
    grid_w = n_cols * (cell_size + 1) + 1
    if max(grid_h, grid_w, n_rows * cell_size) > 30:
        return None

    bases = _make_distinct_patterns(n_rows, cell_size)
    perm = _nontrivial_derangement(n_rows)
    outliers = [bases[j].copy() for j in perm]

    odd_cols = [random.randrange(n_cols) for _ in range(n_rows)]
    rows: list[list[np.ndarray]] = []
    for r in range(n_rows):
        row_cells = [bases[r].copy() for _ in range(n_cols)]
        row_cells[odd_cols[r]] = outliers[r].copy()
        rows.append(row_cells)

    inp = _render_lattice(rows)
    expected = _expected_output(outliers, inp.shape[1])

    if not _semantic_checks(inp, expected, odd_cols, bases):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
