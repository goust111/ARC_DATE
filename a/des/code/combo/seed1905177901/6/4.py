from __future__ import annotations

import random

import numpy as np


def _compress_indices(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected at least one separator index"
    runs: list[tuple[int, int]] = []
    start = end = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == end + 1:
            end = idx
        else:
            runs.append((start, end))
            start = end = idx
    runs.append((start, end))
    return runs


def _regular_intervals(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in runs:
        interval = (prev_end + 1, start)
        assert interval[0] < interval[1], "separator runs must leave non-empty cells"
        intervals.append(interval)
        prev_end = end
    interval = (prev_end + 1, int(length))
    assert interval[0] < interval[1], "separator runs must leave a final non-empty cell"
    intervals.append(interval)

    sizes = [b - a for a, b in intervals]
    assert len(set(sizes)) == 1, "all cell intervals must have the same size"
    return intervals


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected exactly one separator color"
    separator, row_ids, col_ids = candidates[0]
    row_intervals = _regular_intervals(h, _compress_indices(row_ids))
    col_intervals = _regular_intervals(w, _compress_indices(col_ids))
    assert len(row_intervals) >= 2 and len(col_intervals) >= 2, "need legend row/col and target cells"
    return int(separator), row_intervals, col_intervals


def _background_color(grid: np.ndarray, separator: int) -> int:
    payload = grid[grid != separator]
    values, counts = np.unique(payload, return_counts=True)
    best = int(np.argmax(counts))
    return int(values[best])


def _row_legend_color(cell: np.ndarray, bg: int, separator: int) -> int:
    colors = [int(x) for x in np.unique(cell) if int(x) not in {bg, separator}]
    assert len(colors) == 1, "row legend cell must contain exactly one non-background color"
    color = colors[0]

    coords = np.argwhere(cell == color)
    assert len(coords) == 4, "row legend marker must be a 2x2 block"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    assert r1 - r0 == 1 and c1 - c0 == 1, "row legend marker must span 2x2"
    assert np.all(cell[r0 : r1 + 1, c0 : c1 + 1] == color), "row legend marker must be solid"
    return color


def _column_legend_mask(cell: np.ndarray, bg: int, separator: int) -> np.ndarray:
    colors = {int(x) for x in np.unique(cell)}
    assert colors.issubset({bg, separator}), "column legend cell must use only background and separator colors"
    mask = cell == separator
    assert np.any(mask), "column legend cell must contain a shape"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-defined cell matrix, fill each target cell with the column legend's shape
    recolored to the corresponding row legend's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_intervals, col_intervals = _separator_structure(g)
    bg = _background_color(g, separator)

    legend_r0, legend_r1 = row_intervals[0]
    legend_c0, legend_c1 = col_intervals[0]
    column_masks = [
        _column_legend_mask(g[legend_r0:legend_r1, c0:c1], bg, separator)
        for c0, c1 in col_intervals[1:]
    ]
    row_colors = [
        _row_legend_color(g[r0:r1, legend_c0:legend_c1], bg, separator)
        for r0, r1 in row_intervals[1:]
    ]

    assert len({mask.tobytes() for mask in column_masks}) == len(column_masks), "column legend shapes must be distinct"
    assert len(set(row_colors)) == len(row_colors), "row legend colors must be distinct"

    out = g.copy()
    for row_color, (r0, r1) in zip(row_colors, row_intervals[1:], strict=False):
        for mask, (c0, c1) in zip(column_masks, col_intervals[1:], strict=False):
            cell = np.full((r1 - r0, c1 - c0), bg, dtype=int)
            cell[mask] = row_color
            out[r0:r1, c0:c1] = cell
    return out


# --------------- generator ---------------

BACKGROUND = 0
SEPARATOR = 5
ROW_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

TARGET_ROW_RANGE = (2, 4)
TARGET_COL_RANGE = (2, 5)
CELL_HEIGHT_RANGE = (6, 8)
CELL_WIDTH_RANGE = (6, 8)
SEPARATOR_THICKNESS_CHOICES = [1, 2]
CELL_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 300


def _mask(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


BASE_SHAPES = [
    _mask([[1, 1, 1]]),
    _mask([[1], [1], [1]]),
    _mask([[1, 1], [1, 0]]),
    _mask([[1, 1], [0, 1]]),
    _mask([[1, 0], [1, 1]]),
    _mask([[0, 1], [1, 1]]),
    _mask([[1, 1], [1, 1]]),
    _mask([[1, 1, 1], [0, 1, 0]]),
    _mask([[1, 1, 0], [0, 1, 1]]),
    _mask([[0, 1, 1], [1, 1, 0]]),
    _mask([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    _mask([[1, 0, 0], [1, 1, 1]]),
    _mask([[1, 1, 1], [1, 0, 0]]),
    _mask([[1, 0, 1], [1, 1, 1]]),
    _mask([[1, 1, 1], [1, 0, 1]]),
]


def _unique_transforms(shape: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    transforms: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(shape, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = (candidate.shape, candidate.tobytes())
            if key in seen:
                continue
            seen.add(key)
            transforms.append(candidate.copy())
    return transforms


def _shape_pool(cell_h: int, cell_w: int) -> list[np.ndarray]:
    inner_h = cell_h - 2 * CELL_MARGIN
    inner_w = cell_w - 2 * CELL_MARGIN
    assert inner_h >= 2 and inner_w >= 2, "cells must leave room for legends"

    seen: set[bytes] = set()
    masks: list[np.ndarray] = []
    for base in BASE_SHAPES:
        for shape in _unique_transforms(base):
            sh, sw = shape.shape
            if sh > inner_h or sw > inner_w:
                continue
            full = np.zeros((cell_h, cell_w), dtype=bool)
            r0 = CELL_MARGIN + (inner_h - sh) // 2
            c0 = CELL_MARGIN + (inner_w - sw) // 2
            full[r0 : r0 + sh, c0 : c0 + sw] = shape
            key = full.tobytes()
            if key in seen:
                continue
            seen.add(key)
            masks.append(full)
    return masks


def _cell_origin(index: int, cell_size: int, sep: int) -> int:
    return index * (cell_size + sep)


def _row_block_position(cell_h: int, cell_w: int) -> tuple[int, int]:
    r0 = random.randint(CELL_MARGIN, cell_h - CELL_MARGIN - 2)
    c0 = random.randint(CELL_MARGIN, cell_w - CELL_MARGIN - 2)
    return r0, c0


def _render_grid(
    target_rows: int,
    target_cols: int,
    cell_h: int,
    cell_w: int,
    sep: int,
    row_colors: list[int],
    column_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = target_rows + 1
    total_cols = target_cols + 1
    height = total_rows * cell_h + (total_rows - 1) * sep
    width = total_cols * cell_w + (total_cols - 1) * sep

    inp = np.full((height, width), BACKGROUND, dtype=int)
    for k in range(1, total_rows):
        r0 = k * cell_h + (k - 1) * sep
        inp[r0 : r0 + sep, :] = SEPARATOR
    for k in range(1, total_cols):
        c0 = k * cell_w + (k - 1) * sep
        inp[:, c0 : c0 + sep] = SEPARATOR

    top_r0 = _cell_origin(0, cell_h, sep)
    for col_idx, mask in enumerate(column_masks, start=1):
        c0 = _cell_origin(col_idx, cell_w, sep)
        cell = inp[top_r0 : top_r0 + cell_h, c0 : c0 + cell_w]
        cell[mask] = SEPARATOR

    for row_idx, color in enumerate(row_colors, start=1):
        r0 = _cell_origin(row_idx, cell_h, sep)
        c0 = _cell_origin(0, cell_w, sep)
        dr, dc = _row_block_position(cell_h, cell_w)
        inp[r0 + dr : r0 + dr + 2, c0 + dc : c0 + dc + 2] = color

    out = inp.copy()
    for row_idx, color in enumerate(row_colors, start=1):
        r0 = _cell_origin(row_idx, cell_h, sep)
        for col_idx, mask in enumerate(column_masks, start=1):
            c0 = _cell_origin(col_idx, cell_w, sep)
            cell = np.full((cell_h, cell_w), BACKGROUND, dtype=int)
            cell[mask] = color
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    target_rows = random.randint(*TARGET_ROW_RANGE)
    target_cols = random.randint(*TARGET_COL_RANGE)
    cell_h = random.randint(*CELL_HEIGHT_RANGE)
    cell_w = random.randint(*CELL_WIDTH_RANGE)
    sep = random.choice(SEPARATOR_THICKNESS_CHOICES)

    shape_pool = _shape_pool(cell_h, cell_w)
    if len(shape_pool) < target_cols:
        return None

    row_colors = random.sample(ROW_COLOR_POOL, target_rows)
    random.shuffle(shape_pool)
    column_masks = [mask.copy() for mask in shape_pool[:target_cols]]

    inp, expected = _render_grid(target_rows, target_cols, cell_h, cell_w, sep, row_colors, column_masks)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
