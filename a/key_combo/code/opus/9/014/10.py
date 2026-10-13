from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
LOGICAL_SHAPE = (6, 5)
MAX_GENERATION_ATTEMPTS = 200

# All width>=3 row intervals available in a 5-column logical matrix.
INTERVALS = ((0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4))


def _extract_logical(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return g[1::2, 1::2].copy()


def _embed_logical(frame: np.ndarray, logical: np.ndarray) -> np.ndarray:
    out = np.asarray(frame, dtype=int).copy()
    out[1::2, 1::2] = np.asarray(logical, dtype=int)
    return out


def _fill_scanlines(logical: np.ndarray) -> np.ndarray:
    filled = np.asarray(logical, dtype=int).copy()
    for r in range(filled.shape[0]):
        cols = np.flatnonzero(logical[r] == YELLOW)
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                filled[r, left + 1 : right] = YELLOW
    return filled


def _column_sort_order(logical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.count_nonzero(np.asarray(logical, dtype=int) == YELLOW, axis=0)
    order = np.argsort(counts, kind="stable")
    return counts.astype(int), order.astype(int)


def _sort_columns(logical: np.ndarray) -> np.ndarray:
    _counts, order = _column_sort_order(logical)
    return np.asarray(logical, dtype=int)[:, order]


def _mirror_intervals(intervals: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    width = LOGICAL_SHAPE[1] - 1
    return tuple((width - right, width - left) for left, right in intervals)


def _sparse_from_intervals(
    intervals: tuple[tuple[int, int], ...],
    extra_rows: tuple[int, ...] = (),
) -> np.ndarray:
    logical = np.zeros(LOGICAL_SHAPE, dtype=int)
    extras = set(extra_rows)
    for r, (left, right) in enumerate(intervals):
        logical[r, left] = YELLOW
        logical[r, right] = YELLOW
        if r in extras:
            logical[r, 2] = YELLOW
    return logical


def _identity_order(order: np.ndarray) -> bool:
    return np.array_equal(order, np.arange(len(order)))


def _is_valid_sparse_instance(logical: np.ndarray) -> bool:
    logical = np.asarray(logical, dtype=int)
    if logical.shape != LOGICAL_SHAPE:
        return False

    row_positions = [np.flatnonzero(logical[r] == YELLOW) for r in range(logical.shape[0])]
    if any(len(cols) < 2 for cols in row_positions):
        return False
    if any(np.any(np.diff(cols) <= 1) for cols in row_positions):
        return False

    filled = _fill_scanlines(logical)
    fill_counts, fill_order = _column_sort_order(filled)
    if len(set(map(int, fill_counts))) != logical.shape[1]:
        return False
    if _identity_order(fill_order):
        return False

    raw_counts, raw_order = _column_sort_order(logical)
    correct = filled[:, fill_order]
    wrong_skip_fill = logical[:, raw_order]
    if np.array_equal(correct, wrong_skip_fill):
        return False
    if np.array_equal(correct, filled):
        return False

    intervals = [(int(cols[0]), int(cols[-1])) for cols in row_positions]
    if not _has_diagonal_trend(intervals):
        return False
    if np.array_equal(raw_counts, fill_counts):
        return False

    return True


def _has_diagonal_trend(intervals: list[tuple[int, int]]) -> bool:
    left_steps = [b[0] - a[0] for a, b in zip(intervals, intervals[1:])]
    right_steps = [b[1] - a[1] for a, b in zip(intervals, intervals[1:])]
    if not left_steps or not right_steps:
        return False
    for direction in (-1, 1):
        allowed = {0, direction}
        all_steps = left_steps + right_steps
        if all(step in allowed for step in all_steps) and any(step == direction for step in all_steps):
            return True
    return False


def _enumerate_base_intervals() -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []

    def rec(prefix: list[tuple[int, int]]) -> None:
        if len(prefix) == LOGICAL_SHAPE[0]:
            if not _has_diagonal_trend(prefix):
                return
            logical = _sparse_from_intervals(tuple(prefix))
            if _is_valid_sparse_instance(logical):
                out.append(tuple(prefix))
            return

        if not prefix:
            choices = INTERVALS
        else:
            left, right = prefix[-1]
            choices = tuple(
                candidate
                for candidate in INTERVALS
                if left <= candidate[0] <= left + 1 and right <= candidate[1] <= right + 1
            )
        for candidate in choices:
            prefix.append(candidate)
            rec(prefix)
            prefix.pop()

    rec([])
    return out


BASE_INTERVALS = _enumerate_base_intervals()


def _instance_variants(
    intervals: tuple[tuple[int, int], ...],
) -> list[np.ndarray]:
    full_width_rows = [r for r, (left, right) in enumerate(intervals) if left == 0 and right == 4]
    extra_choices = [()]
    if full_width_rows:
        extra_choices.extend((r,) for r in full_width_rows)

    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for extra_rows in extra_choices:
        logical = _sparse_from_intervals(intervals, extra_rows)
        if not _is_valid_sparse_instance(logical):
            continue
        key = logical.tobytes()
        if key in seen:
            continue
        seen.add(key)
        variants.append(logical)
    return variants


ALL_LOGICAL_VARIANTS: list[np.ndarray] = []
for base in BASE_INTERVALS:
    ALL_LOGICAL_VARIANTS.extend(_instance_variants(base))
    ALL_LOGICAL_VARIANTS.extend(_instance_variants(_mirror_intervals(base)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the sparse binary matrix from a separator lattice, fill each row's enclosed span, then reorder columns by filled counts.

    Input structure:
    - The grid is a black separator lattice whose editable cells lie at odd `(r, c)` positions.
    - Those editable cells encode a 6x5 logical matrix with yellow foreground pixels on black.
    - Each logical row contains two or more non-adjacent yellow pixels that act as scanline span markers.

    Rule:
    - Read the logical matrix from the odd-row/odd-column editable cells.
    - In each logical row, fill every black cell strictly between consecutive yellow pixels with yellow.
    - Count yellow pixels in each filled column and sort the columns from the smallest count to the largest.
    - Write the sorted logical matrix back into the original separator lattice while leaving the lattice cells unchanged.

    Key insight:
    - The decisive column counts come from the scanline-filled spans, not from the sparse input markers, so extraction, filling, and sorting are all necessary.
    """
    logical = _extract_logical(grid)
    filled = _fill_scanlines(logical)
    sorted_logical = _sort_columns(filled)
    return _embed_logical(grid, sorted_logical)


def generate_input() -> np.ndarray:
    if not ALL_LOGICAL_VARIANTS:
        raise RuntimeError("no valid logical variants were generated")

    for _ in range(MAX_GENERATION_ATTEMPTS):
        logical = np.array(random.choice(ALL_LOGICAL_VARIANTS), dtype=int)
        if not _is_valid_sparse_instance(logical):
            continue

        grid = np.zeros((LOGICAL_SHAPE[0] * 2 + 1, LOGICAL_SHAPE[1] * 2 + 1), dtype=int)
        grid[1::2, 1::2] = logical

        expected = main(grid)
        filled = _fill_scanlines(logical)
        fill_counts, fill_order = _column_sort_order(filled)
        raw_counts, raw_order = _column_sort_order(logical)

        # Semantic checks: unique target, fill really matters, and sorting really matters.
        if len(set(map(int, fill_counts))) != LOGICAL_SHAPE[1]:
            continue
        if _identity_order(fill_order):
            continue
        if np.array_equal(expected, grid):
            continue
        if np.array_equal(expected[1::2, 1::2], logical[:, raw_order]):
            continue
        if np.array_equal(expected[1::2, 1::2], filled):
            continue
        if np.array_equal(raw_counts, fill_counts):
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
