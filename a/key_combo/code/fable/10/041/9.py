from __future__ import annotations

from collections import Counter

import numpy as np


MAX_ATTEMPTS = 400
ROW_RANGE = (6, 12)
COL_RANGE = (8, 15)
MIN_PIVOT_ROWS = 3
MAX_PIVOT_ROWS = 5
MIN_FILLER_COLORS = 3
ALL_COLORS = np.arange(10, dtype=int)


def _reverse_row(row: np.ndarray, pivot_col: int | None) -> np.ndarray:
    if pivot_col is None:
        return row[::-1].copy()
    return np.concatenate(
        [row[:pivot_col][::-1], row[pivot_col : pivot_col + 1], row[pivot_col + 1 :][::-1]]
    )


def _singleton_pivots(grid: np.ndarray) -> dict[int, int]:
    flat_counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    singleton_colors = {color for color, count in flat_counts.items() if count == 1}

    pivots: dict[int, int] = {}
    for r in range(grid.shape[0]):
        cols = [c for c, value in enumerate(grid[r]) if int(value) in singleton_colors]
        assert len(cols) <= 1, "expected at most one global-singleton pivot per row"
        if cols:
            pivots[int(r)] = int(cols[0])
    return pivots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each row around its own globally unique pivot pixel, or fully reverse the row if it has no pivot.

    Input structure:
    - The grid is fully filled with colored pixels and has no background area.
    - Some colors appear exactly once in the whole grid; each such singleton pixel lies on its own row.
    - Other rows contain only non-singleton colors.

    Rule:
    - Find every color whose total count in the entire grid is exactly one; that pixel is the pivot of its row.
    - In a pivot row, keep the pivot pixel fixed, reverse the segment strictly to its left, and reverse the segment strictly to its right.
    - In a row without a global singleton pivot, reverse the entire row left-to-right.

    Key insight:
    - The axis is defined by whole-grid frequency, not by position or row-local rarity, and the two sides reverse independently without crossing the pivot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pivots = _singleton_pivots(g)
    out = np.empty_like(g)
    for r in range(g.shape[0]):
        out[r] = _reverse_row(g[r], pivots.get(int(r)))
    return out


def _all_collinear(points: list[tuple[int, int]]) -> bool:
    if len(points) < 3:
        return True
    (r0, c0), (r1, c1) = points[:2]
    dr = r1 - r0
    dc = c1 - c0
    for r, c in points[2:]:
        if (r - r0) * dc != (c - c0) * dr:
            return False
    return True


def _choose_internal_col(width: int, rng: np.random.Generator) -> int:
    candidates = [c for c in range(1, width - 1) if c != width // 2]
    if not candidates:
        candidates = list(range(1, width - 1))
    return int(candidates[int(rng.integers(len(candidates)))])


def _make_asymmetric_sequence(
    length: int, colors: list[int], rng: np.random.Generator
) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=int)
    if length == 1:
        return np.array([int(colors[int(rng.integers(len(colors)))])], dtype=int)

    for _ in range(120):
        seq = rng.choice(colors, size=length, replace=True).astype(int)
        if len(set(map(int, seq))) < 2:
            continue
        if np.array_equal(seq, seq[::-1]):
            continue
        return seq
    return np.array(
        [int(colors[i % len(colors)]) for i in range(length)],
        dtype=int,
    )


def _make_pivot_row(
    width: int,
    pivot_col: int,
    pivot_color: int,
    filler_colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    for _ in range(120):
        left = _make_asymmetric_sequence(pivot_col, filler_colors, rng)
        right = _make_asymmetric_sequence(width - pivot_col - 1, filler_colors, rng)
        row = np.concatenate([left, np.array([int(pivot_color)], dtype=int), right])
        correct = _reverse_row(row, pivot_col)
        if np.array_equal(correct, row):
            continue
        if np.array_equal(correct, row[::-1]):
            continue
        return row.astype(int)
    return None


def _make_plain_row(
    width: int,
    filler_colors: list[int],
    rng: np.random.Generator,
    *,
    exact_one_row_singleton: bool,
) -> np.ndarray | None:
    if exact_one_row_singleton:
        if len(filler_colors) < 3:
            return None
        for _ in range(160):
            unique_color, color_a, color_b = map(
                int, rng.choice(filler_colors, size=3, replace=False)
            )
            pos_candidates = [c for c in range(1, width - 1) if c != width // 2]
            if not pos_candidates:
                pos_candidates = list(range(1, width - 1))
            unique_pos = int(pos_candidates[int(rng.integers(len(pos_candidates)))])
            remain = width - 1
            count_a = int(rng.integers(2, remain - 1))
            count_b = remain - count_a
            if count_b == 1:
                continue

            fill = np.array([color_a] * count_a + [color_b] * count_b, dtype=int)
            rng.shuffle(fill)
            row = np.empty(width, dtype=int)
            row[unique_pos] = unique_color
            row[np.arange(width) != unique_pos] = fill

            row_counts = Counter(map(int, row))
            if sum(count == 1 for count in row_counts.values()) != 1:
                continue
            if np.array_equal(row, row[::-1]):
                continue
            if np.array_equal(_reverse_row(row, unique_pos), row[::-1]):
                continue
            return row
        return None

    for _ in range(120):
        row = rng.choice(filler_colors, size=width, replace=True).astype(int)
        if len(set(map(int, row))) < 2:
            continue
        if np.array_equal(row, row[::-1]):
            continue
        return row
    return None


def _row_local_singleton_transform(grid: np.ndarray) -> np.ndarray:
    out = np.empty_like(grid)
    for r in range(grid.shape[0]):
        row = np.asarray(grid[r], dtype=int)
        counts = Counter(map(int, row))
        unique_cols = [c for c, value in enumerate(row) if counts[int(value)] == 1]
        pivot_col = unique_cols[0] if len(unique_cols) == 1 else None
        out[r] = _reverse_row(row, pivot_col)
    return out


def _validate_candidate(
    grid: np.ndarray,
    pivot_rows: list[int],
    pivot_cols: list[int],
    pivot_colors: list[int],
    filler_colors: list[int],
    distractor_row: int,
) -> bool:
    height, width = map(int, grid.shape)
    if max(height, width) > 30:
        return False

    counts = Counter(map(int, grid.ravel()))
    if any(counts[int(color)] != 1 for color in pivot_colors):
        return False
    if any(counts[int(color)] < 2 for color in filler_colors):
        return False

    expected_pivots = {int(r): int(c) for r, c in zip(pivot_rows, pivot_cols)}
    if _singleton_pivots(grid) != expected_pivots:
        return False

    pivot_points = [(int(r), int(c)) for r, c in expected_pivots.items()]
    if len(pivot_points) < 3 or _all_collinear(pivot_points):
        return False
    if not any(c == 0 for _r, c in pivot_points):
        return False
    if not any(c == width - 1 for _r, c in pivot_points):
        return False
    if not any(0 < c < width - 1 and c != width // 2 for _r, c in pivot_points):
        return False

    non_pivot_rows = [r for r in range(height) if r not in expected_pivots]
    if len(non_pivot_rows) < 2 or distractor_row not in non_pivot_rows:
        return False

    distractor_counts = Counter(map(int, grid[distractor_row]))
    if sum(count == 1 for count in distractor_counts.values()) != 1:
        return False
    unique_color = next(color for color, count in distractor_counts.items() if count == 1)
    if counts[int(unique_color)] <= 1:
        return False

    out = main(grid)
    if np.array_equal(out, grid[:, ::-1]):
        return False
    if np.array_equal(out, _row_local_singleton_transform(grid)):
        return False

    for r in range(height):
        if np.array_equal(out[r], grid[r]):
            return False
        if r in expected_pivots:
            if np.array_equal(out[r], grid[r][::-1]):
                return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(ROW_RANGE[0], ROW_RANGE[1] + 1))
    width = int(rng.integers(COL_RANGE[0], COL_RANGE[1] + 1))

    max_pivots = min(MAX_PIVOT_ROWS, height - 2, 10 - MIN_FILLER_COLORS)
    if max_pivots < MIN_PIVOT_ROWS:
        return None
    pivot_count = int(rng.integers(MIN_PIVOT_ROWS, max_pivots + 1))
    filler_max = min(5, 10 - pivot_count)
    if filler_max < MIN_FILLER_COLORS:
        return None
    filler_count = int(rng.integers(MIN_FILLER_COLORS, filler_max + 1))

    palette = rng.choice(ALL_COLORS, size=pivot_count + filler_count, replace=False).astype(int)
    pivot_colors = [int(v) for v in palette[:pivot_count]]
    filler_colors = [int(v) for v in palette[pivot_count:]]

    pivot_rows = sorted(map(int, rng.choice(height, size=pivot_count, replace=False)))
    mandatory_cols = [0, width - 1, _choose_internal_col(width, rng)]
    extra_cols = [int(rng.integers(width)) for _ in range(pivot_count - len(mandatory_cols))]
    pivot_cols = mandatory_cols + extra_cols
    rng.shuffle(pivot_cols)
    if _all_collinear(list(zip(pivot_rows, pivot_cols))):
        return None

    pivot_by_row = {row: (pivot_cols[i], pivot_colors[i]) for i, row in enumerate(pivot_rows)}
    non_pivot_rows = [r for r in range(height) if r not in pivot_by_row]
    distractor_row = int(non_pivot_rows[int(rng.integers(len(non_pivot_rows)))])

    grid = np.empty((height, width), dtype=int)
    for r in range(height):
        if r in pivot_by_row:
            pivot_col, pivot_color = pivot_by_row[r]
            row = _make_pivot_row(width, int(pivot_col), int(pivot_color), filler_colors, rng)
        else:
            row = _make_plain_row(
                width,
                filler_colors,
                rng,
                exact_one_row_singleton=(r == distractor_row),
            )
        if row is None:
            return None
        grid[r] = row

    if not _validate_candidate(
        grid,
        pivot_rows,
        pivot_cols,
        pivot_colors,
        filler_colors,
        distractor_row,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pivot-row reversal input")
