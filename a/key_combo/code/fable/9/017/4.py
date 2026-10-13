from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_fable_9_017_4",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg


_MARKER_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

_GEN_MAX_ATTEMPTS = 400
_GEN_VALIDATE_ATTEMPTS = 120
_MAX_MARKERS_PER_ROW = 2
_MAX_MARKERS_PER_COLUMN = 3
_ADJ4_PENALTY = 5
_ADJ8_PENALTY = 1


def _find_separator(g: np.ndarray) -> int:
    h, _ = g.shape
    grey_cols = np.flatnonzero(np.all(g == Color.GREY, axis=0))
    assert len(grey_cols) == 1, "expected exactly one full-height grey separator"
    sep = int(grey_cols[0])
    assert int(np.count_nonzero(g == Color.GREY)) == h, "grey must appear only on the separator column"
    return sep


def _marker_mask(g: np.ndarray, bg: int) -> np.ndarray:
    return (g != bg) & (g != Color.GREY)


def _side_colors(g: np.ndarray, sep: int, bg: int) -> tuple[int, int]:
    left_cols = sorted(set(map(int, g[:, :sep].ravel())) - {bg})
    right_cols = sorted(set(map(int, g[:, sep + 1 :].ravel())) - {bg})
    assert left_cols and right_cols, "expected markers on both sides of the separator"
    assert left_cols != [Color.GREY] and right_cols != [Color.GREY]
    assert len(left_cols) == 1 and len(right_cols) == 1, "each side must contain exactly one marker color"
    left_color = int(left_cols[0])
    right_color = int(right_cols[0])
    assert left_color != right_color, "the two sides must use different colors"
    return left_color, right_color


def _moment_balance_column(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    coords = np.argwhere(_marker_mask(g, bg))
    assert coords.size > 0
    cols = coords[:, 1].astype(int)
    candidates = []
    for c in range(w):
        if np.any(cols == c):
            continue
        left = cols[cols < c]
        right = cols[cols > c]
        if left.size == 0 or right.size == 0:
            continue
        left_moment = int(np.sum(c - left))
        right_moment = int(np.sum(right - c))
        if left_moment == right_moment:
            candidates.append(c)
    assert len(candidates) == 1, f"expected a unique balance column, found {candidates}"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Move the grey separator onto the unique moment-balancing pivot column and recolor any swept markers to their new side.

    Input structure:
    - The background is black, with one full-height grey vertical separator column.
    - All markers strictly left of the separator share one color, and all markers strictly right of it share a different color.
    - Markers are isolated single pixels scattered on both sides.

    Rule:
    - For each empty candidate column c, compute the left moment as the sum of horizontal distances from all markers left of c to c, and the right moment analogously for all markers right of c.
    - Erase the old grey separator and redraw it at the unique column where the two moments are equal.
    - Any markers strictly between the old and new separator columns switch camps, so recolor them to the color of the side they move into; preserve all other pixels.

    Key insight:
    - The separator is not moved by marker counts alone: the correct pivot is the unique horizontal balance point where distance-weighted torque from both sides matches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected a black background"

    sep = _find_separator(g)
    left_color, right_color = _side_colors(g, sep, bg)
    marker_mask = _marker_mask(g, bg)
    assert not np.any(marker_mask[:, sep]), "markers may not occupy the separator column"

    new_sep = _moment_balance_column(g, bg)
    assert new_sep != sep, "generator should avoid already-balanced separators"
    assert not np.any(marker_mask[:, new_sep]), "the balance column must be empty before moving the separator"

    out = g.copy()
    out[:, sep] = bg
    out[:, new_sep] = Color.GREY

    cols = np.arange(g.shape[1])[None, :]
    if new_sep > sep:
        swept = marker_mask & (cols > sep) & (cols < new_sep)
        out[swept] = left_color
    else:
        swept = marker_mask & (cols > new_sep) & (cols < sep)
        out[swept] = right_color
    return out


def _random_bounded_sum(count: int, total: int, low: int, high: int) -> list[int] | None:
    if count <= 0:
        return [] if total == 0 else None
    if count * low > total or count * high < total:
        return None
    values = [low] * count
    extra = total - count * low
    caps = [high - low] * count
    while extra > 0:
        options = [i for i, cap in enumerate(caps) if cap > 0]
        if not options:
            return None
        idx = random.choice(options)
        values[idx] += 1
        caps[idx] -= 1
        extra -= 1
    random.shuffle(values)
    return values


def _sample_values(count: int, low: int, high: int, *, bias: str = "uniform") -> list[int]:
    values = []
    for _ in range(count):
        if bias == "small" and low < high:
            hi = low + (high - low) // 2
            values.append(random.randint(low, hi))
        elif bias == "large" and low < high:
            lo = high - (high - low) // 2
            values.append(random.randint(lo, high))
        else:
            values.append(random.randint(low, high))
    return values


def _count_relation_ok(direction: int, mode: str, left_count: int, right_count: int) -> bool:
    diff = right_count - left_count
    if mode == "balanced":
        return abs(diff) <= 1
    if mode == "conflict":
        return abs(diff) >= 2 and np.sign(diff) != np.sign(direction)
    if mode == "follow":
        return abs(diff) >= 2 and np.sign(diff) == np.sign(direction)
    raise ValueError(mode)


def _sample_counts(direction: int, mode: str, *, move_right: bool) -> tuple[int, int, int] | None:
    for _ in range(120):
        unswept_left = random.randint(1, 4)
        swept = random.randint(1, 3)
        far_side = random.randint(1, 4)
        total = unswept_left + swept + far_side
        if not 5 <= total <= 10:
            continue
        if move_right:
            left_count = unswept_left
            right_count = swept + far_side
        else:
            left_count = unswept_left + swept
            right_count = far_side
        if _count_relation_ok(direction, mode, left_count, right_count):
            return unswept_left, swept, far_side
    return None


def _sample_columns_move_right(width: int, pivot: int, old_sep: int, mode: str) -> list[int] | None:
    gap = pivot - old_sep
    max_left_dev = pivot
    max_right_dev = width - 1 - pivot
    if gap < 2 or max_right_dev < 1 or max_left_dev < gap + 1:
        return None

    if mode == "conflict":
        for _ in range(160):
            left_old, swept, far_right = 5, 1, 2
            left_devs = _sample_values(left_old, gap + 1, min(max_left_dev, gap + 4), bias="small")
            swept_devs = _sample_values(swept, 1, gap - 1, bias="small")
            total_left = sum(left_devs) + sum(swept_devs)
            right_devs = _random_bounded_sum(far_right, total_left, 1, max_right_dev)
            if right_devs is None:
                continue
            cols = [pivot - d for d in left_devs + swept_devs] + [pivot + d for d in right_devs]
            counts_by_col = Counter(cols)
            if max(counts_by_col.values()) > _MAX_MARKERS_PER_COLUMN:
                continue
            if len(set(cols)) < 4:
                continue
            if not _count_relation_ok(+1, mode, left_old, swept + far_right):
                continue
            return cols
        return None

    for _ in range(200):
        counts = _sample_counts(+1, mode, move_right=True)
        if counts is None:
            return None
        left_old, swept, far_right = counts
        left_bias = "small" if mode == "conflict" else "uniform"
        swept_bias = "small"
        left_devs = _sample_values(left_old, gap + 1, max_left_dev, bias=left_bias)
        swept_devs = _sample_values(swept, 1, gap - 1, bias=swept_bias)
        total_left = sum(left_devs) + sum(swept_devs)
        right_devs = _random_bounded_sum(far_right, total_left, 1, max_right_dev)
        if right_devs is None:
            continue
        cols = [pivot - d for d in left_devs + swept_devs] + [pivot + d for d in right_devs]
        counts_by_col = Counter(cols)
        if max(counts_by_col.values()) > _MAX_MARKERS_PER_COLUMN:
            continue
        if len(set(cols)) < 4:
            continue
        return cols
    return None


def _sample_columns_move_left(width: int, pivot: int, old_sep: int, mode: str) -> list[int] | None:
    gap = old_sep - pivot
    max_left_dev = pivot
    max_right_dev = width - 1 - pivot
    if gap < 2 or max_left_dev < 1 or max_right_dev < gap + 1:
        return None

    if mode == "conflict":
        for _ in range(160):
            left_old, swept, far_right = 2, 1, 5
            swept_devs = _sample_values(swept, 1, gap - 1, bias="small")
            right_devs = _sample_values(far_right, gap + 1, min(max_right_dev, gap + 4), bias="small")
            total_right = sum(swept_devs) + sum(right_devs)
            left_devs = _random_bounded_sum(left_old, total_right, 1, max_left_dev)
            if left_devs is None:
                continue
            cols = [pivot - d for d in left_devs] + [pivot + d for d in swept_devs + right_devs]
            counts_by_col = Counter(cols)
            if max(counts_by_col.values()) > _MAX_MARKERS_PER_COLUMN:
                continue
            if len(set(cols)) < 4:
                continue
            if not _count_relation_ok(-1, mode, left_old + swept, far_right):
                continue
            return cols
        return None

    for _ in range(200):
        counts = _sample_counts(-1, mode, move_right=False)
        if counts is None:
            return None
        left_old, swept, far_right = counts
        swept_devs = _sample_values(swept, 1, gap - 1, bias="small")
        right_bias = "small" if mode == "conflict" else "uniform"
        right_devs = _sample_values(far_right, gap + 1, max_right_dev, bias=right_bias)
        total_right = sum(swept_devs) + sum(right_devs)
        left_devs = _random_bounded_sum(left_old, total_right, 1, max_left_dev)
        if left_devs is None:
            continue
        cols = [pivot - d for d in left_devs] + [pivot + d for d in swept_devs + right_devs]
        counts_by_col = Counter(cols)
        if max(counts_by_col.values()) > _MAX_MARKERS_PER_COLUMN:
            continue
        if len(set(cols)) < 4:
            continue
        return cols
    return None


def _neighbor_score(cell: tuple[int, int], used: set[tuple[int, int]], row_count: int) -> int:
    r, c = cell
    score = row_count * 2
    for rr, cc in used:
        dr = abs(r - rr)
        dc = abs(c - cc)
        if dr + dc == 1:
            score += _ADJ4_PENALTY
        elif max(dr, dc) == 1:
            score += _ADJ8_PENALTY
    return score


def _place_marker_cells(height: int, columns: list[int]) -> list[tuple[int, int]] | None:
    row_counts = [0] * height
    used: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []
    for col in sorted(columns, key=lambda c: (Counter(columns)[c], random.random())):
        options = [(r, col) for r in range(height) if row_counts[r] < _MAX_MARKERS_PER_ROW and (r, col) not in used]
        if not options:
            return None
        scores = [_neighbor_score(cell, used, row_counts[cell[0]]) for cell in options]
        best = min(scores)
        pool = [cell for cell, score in zip(options, scores) if score == best]
        cell = random.choice(pool)
        used.add(cell)
        row_counts[cell[0]] += 1
        cells.append(cell)
    return cells


def _build_input(
    height: int,
    width: int,
    old_sep: int,
    pivot: int,
    columns: list[int],
    left_color: int,
    right_color: int,
) -> np.ndarray | None:
    cells = _place_marker_cells(height, columns)
    if cells is None:
        return None
    grid = np.zeros((height, width), dtype=int)
    grid[:, old_sep] = Color.GREY
    for r, c in cells:
        grid[r, c] = left_color if c < old_sep else right_color
    if np.any(grid[:, pivot] != 0):
        return None
    return grid


def _expected_output(g: np.ndarray) -> np.ndarray:
    bg = int(get_bg(g))
    sep = _find_separator(g)
    left_color, right_color = _side_colors(g, sep, bg)
    pivot = _moment_balance_column(g, bg)
    marker_mask = _marker_mask(g, bg)
    out = g.copy()
    out[:, sep] = bg
    out[:, pivot] = Color.GREY
    cols = np.arange(g.shape[1])[None, :]
    if pivot > sep:
        swept = marker_mask & (cols > sep) & (cols < pivot)
        out[swept] = left_color
    else:
        swept = marker_mask & (cols > pivot) & (cols < sep)
        out[swept] = right_color
    return out


def _validate_case(inp: np.ndarray, direction: int, mode: str) -> bool:
    try:
        g = np.asarray(inp, dtype=int)
        assert g.ndim == 2 and max(g.shape) <= 30
        bg = int(get_bg(g))
        assert bg == Color.BLACK
        sep = _find_separator(g)
        left_color, right_color = _side_colors(g, sep, bg)
        assert left_color != right_color
        pivot = _moment_balance_column(g, bg)
        assert np.sign(pivot - sep) == direction
        marker_mask = _marker_mask(g, bg)
        assert not np.any(marker_mask[:, pivot])
        cols = np.argwhere(marker_mask)[:, 1].astype(int)
        left_count = int(np.count_nonzero(cols < sep))
        right_count = int(np.count_nonzero(cols > sep))
        assert _count_relation_ok(direction, mode, left_count, right_count)
        if pivot > sep:
            swept_mask = marker_mask & (np.arange(g.shape[1])[None, :] > sep) & (np.arange(g.shape[1])[None, :] < pivot)
            assert np.any(swept_mask)
            assert np.all(g[swept_mask] == right_color)
            assert np.any(cols > pivot)
        else:
            swept_mask = marker_mask & (np.arange(g.shape[1])[None, :] > pivot) & (np.arange(g.shape[1])[None, :] < sep)
            assert np.any(swept_mask)
            assert np.all(g[swept_mask] == left_color)
            assert np.any(cols < pivot)
        out = main(g)
        assert np.array_equal(out, _expected_output(g))
        assert not np.array_equal(out, g)
        assert int(np.count_nonzero(out == Color.GREY)) == g.shape[0]
        return True
    except AssertionError:
        return False


def _attempt_case() -> np.ndarray | None:
    direction = random.choice([-1, 1])
    mode = random.choice(["conflict", "follow", "balanced"])
    if mode == "conflict":
        width = random.randint(20, 24)
    else:
        width = random.randint(14, 24)
    height = random.randint(8, 13)

    if direction > 0:
        if mode == "conflict":
            pivot = random.randint(8, min(10, width - 7))
            max_gap = min(3, pivot - 2)
        else:
            pivot = random.randint(5, width - 5)
            max_gap = min(5, pivot - 2)
        if max_gap < 2:
            return None
        gap = random.randint(2, max_gap)
        old_sep = pivot - gap
        columns = _sample_columns_move_right(width, pivot, old_sep, mode)
    else:
        if mode == "conflict":
            pivot = random.randint(max(12, width // 2), width - 8)
            max_gap = min(3, width - pivot - 3)
        else:
            pivot = random.randint(4, width - 6)
            max_gap = min(5, width - pivot - 3)
        if max_gap < 2:
            return None
        gap = random.randint(2, max_gap)
        old_sep = pivot + gap
        columns = _sample_columns_move_left(width, pivot, old_sep, mode)
    if columns is None:
        return None

    total_markers = len(columns)
    min_height = max(8, (total_markers + _MAX_MARKERS_PER_ROW - 1) // _MAX_MARKERS_PER_ROW + 1)
    if min_height > 30:
        return None
    height = max(height, min_height)
    if height > 30:
        return None

    left_color, right_color = random.sample(_MARKER_COLORS, 2)
    grid = _build_input(height, width, old_sep, pivot, columns, left_color, right_color)
    if grid is None:
        return None
    if not _validate_case(grid, direction, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid force-balance separator task")
