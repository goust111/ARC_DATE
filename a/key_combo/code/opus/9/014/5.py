from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_RETRIES = 400


def _find_separator_row(grid: np.ndarray) -> int:
    yellow_rows = np.flatnonzero(np.all(grid == YELLOW, axis=1))
    assert len(yellow_rows) == 1, "expected exactly one full-width yellow separator row"
    sep = int(yellow_rows[0])
    assert sep > 0 and sep < grid.shape[0] - 1, "separator must split the grid"
    assert sep * 2 + 1 == grid.shape[0], "top and bottom halves must have the same height"
    return sep


def _validate_half(region: np.ndarray, outline_color: int) -> None:
    colors = {int(x) for x in np.unique(region)}
    assert colors <= {BLACK, outline_color}, "half-region must contain only background and its outline color"
    for r in range(region.shape[0]):
        cols = np.flatnonzero(region[r] == outline_color)
        assert len(cols) >= 2, "every row must contain at least two outline pixels"


def _scan_fill_mask(region: np.ndarray, outline_color: int) -> np.ndarray:
    mask = np.zeros(region.shape, dtype=bool)
    for r in range(region.shape[0]):
        cols = np.flatnonzero(region[r] == outline_color)
        assert len(cols) >= 2, "scanline fill needs at least two outline pixels per row"
        for left, right in zip(cols, cols[1:]):
            mask[r, left : right + 1] = True
    return mask


def _classify_masks(top_mask: np.ndarray, bottom_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(top_mask.shape, dtype=int)
    both = top_mask & bottom_mask
    top_only = top_mask & ~bottom_mask
    bottom_only = bottom_mask & ~top_mask
    out[both] = GREEN
    out[top_only] = BLUE
    out[bottom_only] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare scanline-filled upper and lower diagonal outlines and color their overlap classes.

    Input structure:
    - The grid is split by one full-width yellow separator row into equal top and bottom halves.
    - The top half contains only black background and blue outline pixels.
    - The bottom half contains only black background and red outline pixels.
    - Each row in each half contains at least two outline pixels that define horizontal spans.

    Rule:
    - In the top half, fill every black cell lying between consecutive blue pixels on the same row.
    - In the bottom half, fill every black cell lying between consecutive red pixels on the same row.
    - Remove the yellow separator and align the two filled halves cell-by-cell.
    - Output green where both halves are filled, blue where only the top half is filled,
      red where only the bottom half is filled, and black where neither half is filled.

    Key insight:
    - The answer is a boolean comparison of two scanline-filled area masks, not a copy of either sparse outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape

    _validate_half(top, BLUE)
    _validate_half(bottom, RED)

    top_mask = _scan_fill_mask(top, BLUE)
    bottom_mask = _scan_fill_mask(bottom, RED)
    return _classify_masks(top_mask, bottom_mask)


def _walk_between(start: int, end: int, length: int, low: int, high: int) -> list[int]:
    vals = [int(start)]
    for idx in range(1, length):
        cur = vals[-1]
        remaining = length - 1 - idx
        choices = []
        for step in (-1, 0, 1):
            nxt = cur + step
            if not (low <= nxt <= high):
                continue
            if abs(end - nxt) > remaining:
                continue
            choices.append(step)
        assert choices, "walk must remain feasible"
        target_step = 0 if end == cur else (1 if end > cur else -1)
        weights = []
        for step in choices:
            if step == target_step:
                weights.append(5)
            elif step == 0:
                weights.append(3)
            else:
                weights.append(1)
        vals.append(cur + random.choices(choices, weights=weights)[0])
    assert vals[-1] == end
    return vals


def _sample_outline_intervals(half_h: int, width: int, sign: int) -> list[tuple[int, int]] | None:
    max_shift = min(3, half_h - 1)
    if max_shift < 1:
        return None
    for _ in range(120):
        left0 = random.randint(0, width - 3)
        delta = sign * random.randint(1, max_shift)
        left1 = left0 + delta
        if not (0 <= left1 <= width - 3):
            continue

        width0 = random.randint(2, min(5, width - left0))
        width1 = random.randint(2, min(5, width - left1))
        right0 = left0 + width0 - 1
        right1 = left1 + width1 - 1
        if right0 >= width or right1 >= width:
            continue
        if abs(left1 - left0) > half_h - 1 or abs(right1 - right0) > half_h - 1:
            continue

        lefts = _walk_between(left0, left1, half_h, 0, width - 2)
        rights = _walk_between(right0, right1, half_h, 1, width - 1)
        intervals = [(int(l), int(r)) for l, r in zip(lefts, rights)]

        if any(r - l < 1 for l, r in intervals):
            continue
        if max(r - l for l, r in intervals) < 2:
            continue
        if len(set(intervals)) < min(3, half_h):
            continue
        centers = [l + r for l, r in intervals]
        if abs(centers[-1] - centers[0]) < 2:
            continue
        return intervals
    return None


def _intervals_to_mask(intervals: list[tuple[int, int]], width: int) -> np.ndarray:
    mask = np.zeros((len(intervals), width), dtype=bool)
    for r, (left, right) in enumerate(intervals):
        mask[r, left : right + 1] = True
    return mask


def _render_input(
    top_intervals: list[tuple[int, int]],
    bottom_intervals: list[tuple[int, int]],
    width: int,
) -> np.ndarray:
    half_h = len(top_intervals)
    grid = np.zeros((half_h * 2 + 1, width), dtype=int)
    grid[half_h, :] = YELLOW
    for r, (left, right) in enumerate(top_intervals):
        grid[r, left] = BLUE
        grid[r, right] = BLUE
    for r, (left, right) in enumerate(bottom_intervals):
        rr = half_h + 1 + r
        grid[rr, left] = RED
        grid[rr, right] = RED
    return grid


def _expected_from_intervals(
    top_intervals: list[tuple[int, int]],
    bottom_intervals: list[tuple[int, int]],
    width: int,
) -> np.ndarray:
    top_mask = _intervals_to_mask(top_intervals, width)
    bottom_mask = _intervals_to_mask(bottom_intervals, width)
    return _classify_masks(top_mask, bottom_mask)


def _outline_only_output(grid: np.ndarray) -> np.ndarray:
    sep = _find_separator_row(grid)
    top_outline = grid[:sep] == BLUE
    bottom_outline = grid[sep + 1 :] == RED
    return _classify_masks(top_outline, bottom_outline)


def _has_row_overlap(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]]) -> bool:
    return any(max(a0, b0) <= min(a1, b1) for (a0, a1), (b0, b1) in zip(intervals_a, intervals_b))


def _has_top_protrusion(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]]) -> bool:
    return any(a0 < b0 or a1 > b1 for (a0, a1), (b0, b1) in zip(intervals_a, intervals_b))


def _has_bottom_protrusion(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]]) -> bool:
    return any(b0 < a0 or b1 > a1 for (a0, a1), (b0, b1) in zip(intervals_a, intervals_b))


def _semantic_checks(
    grid: np.ndarray,
    top_intervals: list[tuple[int, int]],
    bottom_intervals: list[tuple[int, int]],
) -> bool:
    width = grid.shape[1]
    expected = _expected_from_intervals(top_intervals, bottom_intervals, width)
    values = {int(x) for x in np.unique(expected)}
    if values != {BLACK, BLUE, RED, GREEN}:
        return False

    counts = {color: int(np.count_nonzero(expected == color)) for color in (BLACK, BLUE, RED, GREEN)}
    if min(counts[BLUE], counts[RED], counts[GREEN]) < 2:
        return False
    if counts[BLACK] < len(top_intervals):
        return False

    top_mask = _intervals_to_mask(top_intervals, width)
    bottom_mask = _intervals_to_mask(bottom_intervals, width)
    total = top_mask.size
    overlap = int(np.count_nonzero(top_mask & bottom_mask))
    top_only = int(np.count_nonzero(top_mask & ~bottom_mask))
    bottom_only = int(np.count_nonzero(bottom_mask & ~top_mask))
    if overlap >= total // 2:
        return False
    if max(np.count_nonzero(top_mask), np.count_nonzero(bottom_mask)) >= int(total * 0.75):
        return False
    if not _has_row_overlap(top_intervals, bottom_intervals):
        return False
    if not _has_top_protrusion(top_intervals, bottom_intervals):
        return False
    if not _has_bottom_protrusion(top_intervals, bottom_intervals):
        return False

    raw_only = _outline_only_output(grid)
    if np.array_equal(raw_only, expected):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(5, 8)
    width = random.randint(7, 10)

    top_intervals = _sample_outline_intervals(half_h, width, sign=1)
    if top_intervals is None:
        return None

    bottom_sign = random.choice([-1, -1, 1])
    bottom_intervals = _sample_outline_intervals(half_h, width, sign=bottom_sign)
    if bottom_intervals is None:
        return None

    if top_intervals == bottom_intervals:
        return None

    grid = _render_input(top_intervals, bottom_intervals, width)
    if not _semantic_checks(grid, top_intervals, bottom_intervals):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
