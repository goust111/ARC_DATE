from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    _arc_utils = SourceFileLoader("_arc_utils_shared", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    bbox = _arc_utils.bbox
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


ALLOWED_BAND_COLORS = (Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL)
MAX_GENERATION_ATTEMPTS = 400


def _is_monotone(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _stack_bands(bands: np.ndarray, counts: np.ndarray) -> np.ndarray:
    pieces = [np.repeat(bands[i : i + 1, :], int(counts[i]), axis=0) for i in range(bands.shape[0])]
    return np.concatenate(pieces, axis=0)


def _parse_grid(grid: np.ndarray) -> dict[str, np.ndarray | int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    separator_rows = np.flatnonzero(np.all(g == Color.GREY, axis=1))
    assert separator_rows.size == 1, "expected exactly one full-width grey separator"
    sep = int(separator_rows[0])
    assert 3 <= sep < h - 3, "separator must split non-empty upper and lower regions"

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    n_bands = int(bottom.shape[0])
    assert 3 <= n_bands <= 5, "expected 3 to 5 row bands"

    band_colors: list[int] = []
    for row in bottom:
        uniques = np.unique(row)
        assert uniques.size == 1, "each row band must be monochrome"
        color = int(uniques[0])
        assert color in ALLOWED_BAND_COLORS, "row bands must use the allowed palette"
        band_colors.append(color)
    assert len(set(band_colors)) == n_bands, "row band colors must be distinct"

    green_mask = top == Color.GREEN
    matrix_bbox = bbox(green_mask)
    assert matrix_bbox is not None, "expected a green grid in the upper region"
    r0, c0, r1, c1 = matrix_bbox
    matrix = top[r0 : r1 + 1, c0 : c1 + 1]

    outside = top.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK
    assert np.all(outside == Color.BLACK), "upper region should only contain the embedded matrix"

    mh, mw = matrix.shape
    assert mh % 2 == 1 and mw % 2 == 1, "matrix bbox must alternate lines and editable cells"
    n_rows = (mh - 1) // 2
    n_cols = (mw - 1) // 2
    assert 2 <= n_rows <= 6, "matrix must have a small positive number of editable rows"
    assert n_cols == n_bands, "band count must match matrix column count"

    rr = np.arange(mh)[:, None]
    cc = np.arange(mw)[None, :]
    line_mask = (rr % 2 == 0) | (cc % 2 == 0)
    assert np.all(matrix[line_mask] == Color.GREEN), "grid lines must be green"

    cells = matrix[1::2, 1::2]
    assert set(map(int, np.unique(cells))).issubset({Color.BLACK, Color.BLUE}), "editable cells must be binary"

    counts = np.sum(cells == Color.BLUE, axis=0).astype(int)
    assert np.all(counts >= 1), "each column must contain at least one blue cell"

    return {
        "grid": g,
        "top": top,
        "bottom": bottom.copy(),
        "cells": cells.copy(),
        "counts": counts,
        "separator": sep,
        "matrix_bbox": np.array([r0, c0, r1, c1], dtype=int),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count blue cells in each embedded binary-matrix column and use those counts to repeat the matching row bands.

    Input structure:
    - A single full-width grey separator splits the grid into an upper matrix region and a lower row-band region.
    - The upper region contains one green lattice whose editable cells are black or blue, with columns read left to right.
    - The lower region contains 3 to 5 full-width monochrome row bands whose top-to-bottom order matches the matrix columns.

    Rule:
    - Extract the editable cells from the green lattice and count the number of blue cells in each matrix column.
    - Take the i-th row band below the separator and repeat that full-width row vertically exactly as many times as the i-th column count.
    - Stack the repeated row bands in column order and discard the separator and matrix region.

    Key insight:
    - The matrix is a frequency encoder: only each column's blue-cell count matters, and that count controls how often its paired row band appears in the output.
    """
    parsed = _parse_grid(grid)
    return _stack_bands(parsed["bottom"], parsed["counts"])


def _make_cells(n_rows: int, counts: list[int]) -> np.ndarray:
    n_cols = len(counts)
    for _ in range(200):
        cells = np.zeros((n_rows, n_cols), dtype=int)
        for c, count in enumerate(counts):
            chosen = sorted(random.sample(range(n_rows), count))
            cells[chosen, c] = Color.BLUE

        row_counts = np.sum(cells == Color.BLUE, axis=1).tolist()
        gap_column_exists = any(
            1 < counts[c] < n_rows and np.any(np.diff(np.flatnonzero(cells[:, c] == Color.BLUE)) > 1)
            for c in range(n_cols)
        )
        if not gap_column_exists and any(1 < count < n_rows for count in counts):
            continue
        if len({tuple(row) for row in cells.tolist()}) == 1:
            continue
        if n_rows == n_cols and (row_counts == counts or row_counts == counts[::-1]):
            continue
        return cells
    raise RuntimeError("failed to construct a non-degenerate binary matrix")


def _build_candidate() -> np.ndarray:
    n_cols = random.randint(3, 5)
    n_rows = random.randint(3, 6)

    for _ in range(200):
        counts = [random.randint(1, n_rows) for _ in range(n_cols)]
        if len(set(counts)) == 1:
            continue
        if max(counts) <= 1:
            continue
        if _is_monotone(counts):
            continue
        try:
            cells = _make_cells(n_rows, counts)
        except RuntimeError:
            continue
        break
    else:
        raise RuntimeError("failed to sample non-degenerate column counts")

    matrix_h = 2 * n_rows + 1
    matrix_w = 2 * n_cols + 1

    width = random.randint(max(matrix_w, 10), min(22, 30))
    left_pad = random.randint(0, width - matrix_w)
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    top_h = top_pad + matrix_h + bottom_pad
    total_h = top_h + 1 + n_cols

    grid = np.full((total_h, width), Color.BLACK, dtype=int)
    grid[top_h, :] = Color.GREY

    band_colors = random.sample(list(ALLOWED_BAND_COLORS), n_cols)
    for idx, color in enumerate(band_colors):
        grid[top_h + 1 + idx, :] = color

    for r in range(matrix_h):
        for c in range(matrix_w):
            rr = top_pad + r
            cc = left_pad + c
            if r % 2 == 0 or c % 2 == 0:
                grid[rr, cc] = Color.GREEN
            else:
                grid[rr, cc] = cells[(r - 1) // 2, (c - 1) // 2]

    return grid


def _validate_example(grid: np.ndarray) -> None:
    parsed = _parse_grid(grid)
    counts = parsed["counts"].tolist()
    cells = parsed["cells"]
    bands = parsed["bottom"]

    assert len(set(counts)) > 1, "uniform counts make the counting step replaceable"
    assert max(counts) > 1, "all-one counts make repetition degenerate"
    assert not _is_monotone(counts), "monotone counts weaken the hidden correspondence"

    row_counts = np.sum(cells == Color.BLUE, axis=1).tolist()
    if len(row_counts) == len(counts):
        assert row_counts != counts and row_counts != counts[::-1], "row counts must not mimic column counts"

    out = main(grid)
    expected = _stack_bands(bands, parsed["counts"])
    assert np.array_equal(out, expected)

    cursor = 0
    for idx, count in enumerate(counts):
        block = out[cursor : cursor + count, :]
        assert block.shape == (count, grid.shape[1])
        assert np.all(block == bands[idx : idx + 1, :])
        cursor += count
    assert cursor == out.shape[0]
    assert max(grid.shape) <= 30 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        try:
            _validate_example(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
