from __future__ import annotations

import random

import numpy as np


ROW_MARKER = 2
COL_MARKER = 1
TARGET_COLOR = 5
ARC_MAX = 30
GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.argmax(counts))
    assert int(np.sum(counts == counts[best])) == 1, "background must be uniquely dominant"
    return int(values[best])


def _find_target_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(grid == TARGET_COLOR)
    assert len(points) > 0, "missing target rectangle"
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0))
    target = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(target == TARGET_COLOR), "target must be a solid rectangle"
    assert len(points) == target.size, "target color must only appear inside the rectangle"
    return r0, c0, r1, c1


def _find_separator_row(grid: np.ndarray, target_top: int, bg: int) -> int:
    candidates: list[int] = []
    for r in range(target_top):
        if not np.all(grid[r] == grid[r, 0]):
            continue
        color = int(grid[r, 0])
        if color in {bg, TARGET_COLOR}:
            continue
        if np.any(np.isin(grid[:r], (ROW_MARKER, COL_MARKER))):
            candidates.append(r)
    assert candidates, "missing separator row"
    return max(candidates)


def _instruction_rows(grid: np.ndarray, sep_row: int, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    rows: list[tuple[int, int, int, int]] = []
    marker_cols: set[int] = set()

    for r in range(sep_row):
        nz = np.flatnonzero(grid[r] != bg)
        if len(nz) < 2:
            continue
        marker_col = int(nz[0])
        marker = int(grid[r, marker_col])
        if marker not in {ROW_MARKER, COL_MARKER}:
            continue

        stripe_cols = nz[nz > marker_col]
        assert len(stripe_cols) > 0, "instruction row needs a stripe"
        start = int(stripe_cols[0])
        end = int(stripe_cols[-1])
        assert end - start + 1 == len(stripe_cols), "stripe must be contiguous"
        stripe_color = int(grid[r, start])
        assert np.all(grid[r, start : end + 1] == stripe_color), "stripe must be monochrome"
        assert np.all(grid[r, marker_col + 1 : start] == bg), "gap between marker and stripe must be background"

        rows.append((marker, end - start + 1, stripe_color, marker_col))
        marker_cols.add(marker_col)

    assert rows, "no instructions found"
    assert len(marker_cols) == 1, "all markers must share one column"
    return next(iter(marker_cols)), rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the barcode rows into row/column paint instructions and apply them to the gray target rectangle.

    Input structure:
    - The grid has a dominant background color and is split horizontally into a barcode zone and a target zone.
    - The barcode zone contains one-pixel-tall horizontal color stripes, each preceded by a single marker pixel in a shared left-side marker column.
    - Marker color 2 means a row instruction, marker color 1 means a column instruction.
    - A full-width separator row sits between the barcode zone and the lower zone, which contains one solid gray (5) rectangle.

    Rule:
    - Find the gray rectangle and crop it as the output canvas.
    - Read barcode instruction rows from top to bottom.
    - For each instruction, use the stripe color as the paint color and the stripe length as a 1-based index into the target rectangle.
    - Marker 2 paints the indexed row; marker 1 paints the indexed column.
    - Later instructions overwrite earlier colors at intersections or repeated indices.

    Key insight:
    - The stripe length is not something to copy geometrically; it is a positional code telling which target row or column to repaint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    tr0, tc0, tr1, tc1 = _find_target_bbox(g)
    sep_row = _find_separator_row(g, tr0, bg)
    _, instructions = _instruction_rows(g, sep_row, bg)

    out = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    h, w = out.shape
    assert np.all(out == TARGET_COLOR), "input target must be unpainted"

    for marker, length, color, _ in instructions:
        if marker == ROW_MARKER:
            assert 1 <= length <= h, "row index out of range"
            out[length - 1, :] = color
        else:
            assert 1 <= length <= w, "column index out of range"
            out[:, length - 1] = color

    return out


def _sample_balanced_types(n: int) -> list[int]:
    n_row = n // 2
    n_col = n // 2
    if n % 2 == 1:
        if random.random() < 0.5:
            n_row += 1
        else:
            n_col += 1
    types = [ROW_MARKER] * n_row + [COL_MARKER] * n_col
    random.shuffle(types)
    return types


def _sample_lengths(count: int, limit: int) -> list[int]:
    assert count >= 1 and limit >= 1
    lengths = [random.randint(1, limit) for _ in range(count)]
    if count >= 2 and limit >= 2 and random.random() < 0.5:
        lengths[random.randrange(count)] = random.choice([1, limit])
    return lengths


def _bg_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.argmax(counts))
    if int(values[best]) != bg:
        return False
    return int(np.sum(counts == counts[best])) == 1


def _build_expected(target_h: int, target_w: int, instructions: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full((target_h, target_w), TARGET_COLOR, dtype=int)
    for marker, length, color in instructions:
        if marker == ROW_MARKER:
            out[length - 1, :] = color
        else:
            out[:, length - 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c not in {ROW_MARKER, COL_MARKER, TARGET_COLOR}]
    bg = random.choice(bg_choices)
    sep_choices = [c for c in range(10) if c not in {bg, ROW_MARKER, COL_MARKER, TARGET_COLOR}]
    sep = random.choice(sep_choices)
    stripe_choices = [c for c in range(10) if c not in {bg, sep, ROW_MARKER, COL_MARKER, TARGET_COLOR}]
    if not stripe_choices:
        return None

    target_h = random.randint(3, 9)
    target_w = random.randint(3, 9)

    n_instr = random.randint(2, min(8, target_h + target_w))
    types = _sample_balanced_types(n_instr)
    row_count = sum(t == ROW_MARKER for t in types)
    col_count = n_instr - row_count
    assert row_count >= 1 and col_count >= 1

    row_lengths = _sample_lengths(row_count, target_h)
    col_lengths = _sample_lengths(col_count, target_w)

    instructions: list[tuple[int, int, int]] = []
    row_i = 0
    col_i = 0
    for marker in types:
        if marker == ROW_MARKER:
            length = row_lengths[row_i]
            row_i += 1
        else:
            length = col_lengths[col_i]
            col_i += 1
        instructions.append((marker, length, random.choice(stripe_choices)))

    top_pad = random.randint(0, 2)
    gaps = [random.randint(0, 1) for _ in range(max(0, n_instr - 1))]
    barcode_h = top_pad + n_instr + sum(gaps)
    sep_row = barcode_h

    target_gap_top = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    total_h = barcode_h + 1 + target_gap_top + target_h + bottom_pad
    if total_h > ARC_MAX:
        return None

    marker_col = random.randint(0, 3)
    stripe_gap = random.randint(1, 2)
    longest = max(length for _, length, _ in instructions)
    barcode_min_w = marker_col + stripe_gap + longest
    total_w = max(barcode_min_w + random.randint(0, 4), target_w + random.randint(0, 6))
    if total_w > ARC_MAX:
        return None

    grid = np.full((total_h, total_w), bg, dtype=int)
    grid[sep_row, :] = sep

    r = top_pad
    for i, (marker, length, color) in enumerate(instructions):
        grid[r, marker_col] = marker
        start = marker_col + stripe_gap
        grid[r, start : start + length] = color
        if i < len(gaps):
            r += 1 + gaps[i]

    target_top = sep_row + 1 + target_gap_top
    target_left = random.randint(0, total_w - target_w)
    grid[target_top : target_top + target_h, target_left : target_left + target_w] = TARGET_COLOR

    if not _bg_is_unique_dominant(grid, bg):
        return None

    expected = _build_expected(target_h, target_w, instructions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
