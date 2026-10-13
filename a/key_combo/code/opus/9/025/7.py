from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg  # noqa: E402


BACKGROUND = 0
LATTICE = 5
FOREGROUND_COLORS = (1, 2, 3)
CELL_SIZE = 2
LINE_STEP = 3
MAX_ATTEMPTS = 400


def _find_lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.count_nonzero(grid[r, :] == LATTICE) > w // 2]
    cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == LATTICE) > h // 2]
    assert len(rows) >= 4 and len(cols) >= 4, "expected a visible lattice"
    assert len(set(np.diff(rows))) == 1 and rows[1] - rows[0] == LINE_STEP
    assert len(set(np.diff(cols))) == 1 and cols[1] - cols[0] == LINE_STEP
    return rows, cols


def _find_holes(grid: np.ndarray, rows: list[int], cols: list[int]) -> tuple[int, int, int, int]:
    holes = [(r, c) for r in rows for c in cols if int(grid[r, c]) == BACKGROUND]
    assert len(holes) == 4, "expected exactly four missing intersections"
    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2
    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return r0, r1, c0, c1


def _cell_bands(line_positions: list[int]) -> list[tuple[int, int]]:
    return [(line_positions[i] + 1, line_positions[i + 1] - 1) for i in range(len(line_positions) - 1)]


def _analyze_crop_columns(grid: np.ndarray) -> dict:
    rows, cols = _find_lattice_lines(grid)
    r0, r1, c0, c1 = _find_holes(grid, rows, cols)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    local_rows = [r for r in rows if r0 <= r <= r1]
    local_cols = [c for c in cols if c0 <= c <= c1]
    row_bands = _cell_bands(local_rows)
    col_bands = _cell_bands(local_cols)
    assert 2 <= len(col_bands) <= 4

    counts: list[int] = []
    colors: list[int] = []
    occupied_cells: list[int] = []
    partial_columns: list[bool] = []
    for c_band_idx, (ca, cb) in enumerate(col_bands):
        rel_ca = ca - c0
        rel_cb = cb - c0
        column_slice = crop[:, rel_ca : rel_cb + 1]
        fg_mask = (column_slice != BACKGROUND) & (column_slice != LATTICE)
        count = int(np.count_nonzero(fg_mask))
        assert count > 0
        level = math.isqrt(count)
        assert level * level == count and count in (1, 4, 9)
        values = sorted(set(int(v) for v in column_slice[fg_mask]))
        assert len(values) == 1 and values[0] in FOREGROUND_COLORS

        per_cell = []
        for ra, rb in row_bands:
            rel_ra = ra - r0
            rel_rb = rb - r0
            cell = crop[rel_ra : rel_rb + 1, rel_ca : rel_cb + 1]
            cell_fg = int(np.count_nonzero((cell != BACKGROUND) & (cell != LATTICE)))
            per_cell.append(cell_fg)

        counts.append(count)
        colors.append(values[0])
        occupied_cells.append(sum(v > 0 for v in per_cell))
        partial_columns.append(any(v not in (0, CELL_SIZE * CELL_SIZE) for v in per_cell))

    return {
        "rows": rows,
        "cols": cols,
        "crop_bbox": (r0, r1, c0, c1),
        "crop": crop,
        "local_rows": local_rows,
        "local_cols": local_cols,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "counts": counts,
        "colors": colors,
        "occupied_cells": occupied_cells,
        "partial_columns": partial_columns,
    }


def _build_pyramids(counts: list[int], colors: list[int]) -> np.ndarray:
    levels = [math.isqrt(v) for v in counts]
    assert all(level * level == count for level, count in zip(levels, counts))
    height = max(levels)
    widths = [2 * level - 1 for level in levels]
    total_width = sum(widths) + max(0, len(widths) - 1)
    out = np.full((height, total_width), BACKGROUND, dtype=int)

    start = 0
    for level, color in zip(levels, colors):
        base_width = 2 * level - 1
        for layer in range(level):
            width = 2 * layer + 1
            r = height - level + layer
            c0 = start + (base_width - width) // 2
            out[r, c0 : c0 + width] = color
        start += base_width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the crop marked by four missing lattice intersections, count foreground pixels by crop column, and rebuild them as bottom-aligned pyramids.

    Input structure:
    - A black background contains a full grey lattice made from 1-pixel lines with 2x2 cell interiors.
    - Exactly four lattice intersections are black instead of grey, and those holes are the corners of one axis-aligned rectangle.
    - Foreground pixels of colors blue, red, or green appear in cell interiors; within each crop column all foreground pixels share one color.

    Rule:
    - Detect the regular grey lattice and the four missing intersections, then take the inclusive crop bounded by those holes.
    - Inside that crop, scan the cell columns from left to right and count all non-black, non-grey pixels in each column.
    - Each count is a perfect square 1, 4, or 9; its square root is the pyramid height, and the shared column color becomes the pyramid color.
    - Output the resulting pyramids side by side with 1-column black gaps and their bases aligned on the bottom row.

    Key insight:
    - The missing intersections are not defects; they are the only locator for the data rectangle whose per-column pixel totals encode the pyramid sizes.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_crop_columns(g)
    return _build_pyramids(info["counts"], info["colors"])


def _grid_shape_from_lines(n_h_lines: int, n_v_lines: int) -> tuple[int, int]:
    return 3 * n_h_lines + 2, 3 * n_v_lines + 2


def _line_positions(n_lines: int) -> list[int]:
    return [2 + LINE_STEP * i for i in range(n_lines)]


def _sample_spans(rng: np.random.Generator, n_h_lines: int, n_v_lines: int) -> tuple[int, int]:
    row_span_options = [span for span in (2, 3, 4) if span < n_h_lines - 1]
    col_span_options = [span for span in (2, 3, 4) if span < n_v_lines]
    assert row_span_options and col_span_options
    row_span_weights = np.array([1 if span == 2 else 2 for span in row_span_options], dtype=float)
    row_span = int(rng.choice(row_span_options, p=row_span_weights / row_span_weights.sum()))
    col_span = int(rng.choice(col_span_options))
    return row_span, col_span


def _column_slots(line_rows: list[int], line_col_left: int, line_col_right: int) -> list[list[tuple[int, int]]]:
    slots: list[list[tuple[int, int]]] = []
    for i in range(len(line_rows) - 1):
        cell_slots = []
        for r in range(line_rows[i] + 1, line_rows[i + 1]):
            for c in range(line_col_left + 1, line_col_right):
                cell_slots.append((r, c))
        assert len(cell_slots) == CELL_SIZE * CELL_SIZE
        slots.append(cell_slots)
    return slots


def _sample_column_coords(
    rng: np.random.Generator,
    cell_slots: list[list[tuple[int, int]]],
    count: int,
    force_partial: bool,
) -> tuple[list[tuple[int, int]], list[int]]:
    all_slots = [coord for cell in cell_slots for coord in cell]
    for _ in range(200):
        choice_idx = rng.choice(len(all_slots), size=count, replace=False)
        chosen = [all_slots[int(i)] for i in choice_idx]
        per_cell_counts = []
        for cell in cell_slots:
            per_cell_counts.append(sum(coord in set(chosen) for coord in cell))
        has_partial = any(v not in (0, CELL_SIZE * CELL_SIZE) for v in per_cell_counts)
        if (not force_partial) or has_partial:
            return chosen, per_cell_counts
    raise RuntimeError("failed to sample non-degenerate column pixels")


def _full_height_counts(grid: np.ndarray, rows: list[int], target_col_bands: list[tuple[int, int]]) -> list[int]:
    counts = []
    for ca, cb in target_col_bands:
        total = 0
        for i in range(len(rows) - 1):
            cell = grid[rows[i] + 1 : rows[i + 1], ca : cb + 1]
            total += int(np.count_nonzero((cell != BACKGROUND) & (cell != LATTICE)))
        counts.append(total)
    return counts


def _expected_output(counts: list[int], colors: list[int]) -> np.ndarray:
    return _build_pyramids(counts, colors)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_h_lines = int(rng.integers(6, 10))
    n_v_lines = int(rng.integers(5, 10))
    row_span, col_span = _sample_spans(rng, n_h_lines, n_v_lines)
    h, w = _grid_shape_from_lines(n_h_lines, n_v_lines)
    line_rows = _line_positions(n_h_lines)
    line_cols = _line_positions(n_v_lines)

    row_start = int(rng.integers(0, n_h_lines - row_span - 1))
    col_start = int(rng.integers(0, n_v_lines - col_span))
    row_end = row_start + row_span
    col_end = col_start + col_span

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[line_rows, :] = LATTICE
    grid[:, line_cols] = LATTICE

    crop_line_rows = line_rows[row_start : row_end + 1]
    crop_line_cols = line_cols[col_start : col_end + 1]
    crop_counts_allowed = [1, 4] if row_span == 2 else [1, 4, 9]
    counts = [int(rng.choice(crop_counts_allowed)) for _ in range(col_span)]
    if len(set(counts)) == 1 and len(counts) > 1:
        flip_idx = int(rng.integers(0, len(counts)))
        alt = [v for v in crop_counts_allowed if v != counts[flip_idx]]
        counts[flip_idx] = int(rng.choice(alt))

    colors = [int(rng.choice(FOREGROUND_COLORS)) for _ in range(col_span)]
    if len({(counts[i], colors[i]) for i in range(col_span)}) == 1 and col_span > 1:
        colors[-1] = int(next(color for color in FOREGROUND_COLORS if color != colors[-1]))

    partial_target = None
    if not any(count in (1, 9) for count in counts):
        partial_target = int(rng.integers(0, col_span))

    column_partials: list[bool] = []
    for j in range(col_span):
        cell_slots = _column_slots(crop_line_rows, crop_line_cols[j], crop_line_cols[j + 1])
        coords, per_cell = _sample_column_coords(rng, cell_slots, counts[j], force_partial=j == partial_target)
        for r, c in coords:
            grid[r, c] = colors[j]
        column_partials.append(any(v not in (0, CELL_SIZE * CELL_SIZE) for v in per_cell))

    # Mark the crop rectangle by removing exactly four lattice intersections.
    for r in (crop_line_rows[0], crop_line_rows[-1]):
        for c in (crop_line_cols[0], crop_line_cols[-1]):
            grid[r, c] = BACKGROUND

    # Add sparse distractors in the same global columns but outside the crop rows.
    outside_slots: list[tuple[int, int]] = []
    for j in range(col_span):
        left = crop_line_cols[j]
        right = crop_line_cols[j + 1]
        for i in range(n_h_lines - 1):
            if row_start <= i < row_end:
                continue
            for r in range(line_rows[i] + 1, line_rows[i + 1]):
                for c in range(left + 1, right):
                    outside_slots.append((r, c))
    if len(outside_slots) < 2:
        return None
    distractor_n = int(rng.integers(1, min(5, len(outside_slots)) + 1))
    for idx in rng.choice(len(outside_slots), size=distractor_n, replace=False):
        r, c = outside_slots[int(idx)]
        grid[r, c] = int(rng.choice(FOREGROUND_COLORS))

    info = _analyze_crop_columns(grid)
    if info["counts"] != counts or info["colors"] != colors:
        return None
    if not any(info["partial_columns"]):
        return None
    if not any(int(a) != int(b) for a, b in zip(_full_height_counts(grid, line_rows, info["col_bands"]), counts)):
        return None
    if not np.array_equal(main(grid), _expected_output(counts, colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
