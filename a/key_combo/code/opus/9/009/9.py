from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
TARGET_HEIGHT = 9
ALLOWED_GAPS = (1, 3, 5, 7)
GENERATION_RETRIES = 300


def _row_pair(row: np.ndarray, color: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(row == color)
    if len(cols) == 0:
        return None
    assert len(cols) == 2, "each occupied row must contain exactly two markers"
    c0, c1 = map(int, cols)
    assert c0 + c1 == len(row) - 1, "the two markers must be mirrored around the vertical center"
    return c0, c1


def _row_span(row: np.ndarray, color: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(row == color)
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1])


def _occupied_prefix_height(grid: np.ndarray, color: int) -> int:
    occupied_rows = [r for r in range(grid.shape[0]) if np.any(grid[r] == color)]
    assert occupied_rows, "expected at least one occupied row"
    height = occupied_rows[-1] + 1
    assert occupied_rows == list(range(height)), "the visible pattern must occupy a top-aligned prefix"
    assert not np.any(grid[height:] == color), "rows below the visible prefix must be empty"
    return height


def _infer_vertical_period(grid: np.ndarray, color: int) -> int:
    visible_h = _occupied_prefix_height(grid, color)
    signatures = [_row_pair(grid[r], color) for r in range(visible_h)]
    for period in range(1, visible_h):
        if all(signatures[r] == signatures[r + period] for r in range(visible_h - period)):
            return period
    raise AssertionError("failed to infer a unique vertical translation period")


def _tile_pairs(base_pairs: list[tuple[int, int]], width: int, color: int) -> np.ndarray:
    out = np.full((TARGET_HEIGHT, width), BG, dtype=int)
    for r in range(TARGET_HEIGHT):
        c0, c1 = base_pairs[r % len(base_pairs)]
        out[r, c0] = color
        out[r, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the vertically periodic mirrored marker rows to height 9, recolor them red, and connect the closest pair rows.

    Input structure:
    - The grid has black background and blue foreground markers.
    - The occupied part is a top-aligned prefix of rows; each occupied row contains exactly two blue pixels mirrored across the vertical center.
    - The visible occupied prefix already exhibits a vertical translation period, while the remaining lower rows are empty.

    Rule:
    - Infer the smallest positive vertical period from repeated occupied rows.
    - Take the first one-period slice and tile it downward until the height becomes 9, keeping the original width.
    - Recolor every tiled blue marker red.
    - Compute the horizontal gap between the two red markers in each row; for every row attaining the minimum gap, fill the full span between the two markers with red.

    Key insight:
    - The nearest mirrored pair must be identified after completing the periodic stack to the full 9-row span, because the decisive rows recur in the extended pattern rather than only in the visible prefix.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    width = g.shape[1]
    assert width % 2 == 1, "mirrored rows require an odd width"

    visible_h = _occupied_prefix_height(g, BLUE)
    period = _infer_vertical_period(g, BLUE)
    base_pairs = []
    for r in range(period):
        pair = _row_pair(g[r], BLUE)
        assert pair is not None, "the period slice must contain occupied rows only"
        base_pairs.append(pair)

    out = _tile_pairs(base_pairs, width, RED)

    row_gaps: list[tuple[int, int, int, int]] = []
    for r in range(out.shape[0]):
        c0, c1 = _row_pair(out[r], RED)
        assert c0 is not None and c1 is not None
        row_gaps.append((c1 - c0 - 1, r, c0, c1))

    min_gap = min(gap for gap, _r, _c0, _c1 in row_gaps)
    for gap, r, c0, c1 in row_gaps:
        if gap == min_gap:
            out[r, c0 : c1 + 1] = RED

    # Guard against a degenerate implementation that ignores the extension step.
    assert out.shape[0] > visible_h and np.any(np.count_nonzero(out[visible_h:] == RED, axis=1) > 2)
    return out


def _gap_to_pair(width: int, gap: int) -> tuple[int, int]:
    assert width == 9, "the generator is designed for 9-column puzzles"
    assert gap in ALLOWED_GAPS
    c0 = (width - 2 - gap) // 2
    c1 = width - 1 - c0
    return c0, c1


def _sample_base_gaps(period: int) -> list[int]:
    if period == 2:
        return random.sample(ALLOWED_GAPS, 2)
    return random.sample(ALLOWED_GAPS, 3)


def _render_input(height: int, row_gaps: list[int], width: int = 9) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r, gap in enumerate(row_gaps):
        c0, c1 = _gap_to_pair(width, gap)
        grid[r, c0] = BLUE
        grid[r, c1] = BLUE
    return grid


def _visible_shortcut(grid: np.ndarray) -> np.ndarray:
    visible_h = _occupied_prefix_height(grid, BLUE)
    out = np.full((TARGET_HEIGHT, grid.shape[1]), BG, dtype=int)
    gaps: list[tuple[int, int, int, int]] = []
    for r in range(visible_h):
        c0, c1 = _row_pair(grid[r], BLUE)
        out[r, c0] = RED
        out[r, c1] = RED
        gaps.append((c1 - c0 - 1, r, c0, c1))
    min_gap = min(gap for gap, _r, _c0, _c1 in gaps)
    for gap, r, c0, c1 in gaps:
        if gap == min_gap:
            out[r, c0 : c1 + 1] = RED
    return out


def _validate_instance(grid: np.ndarray, period: int, row_gaps: list[int]) -> bool:
    if grid.ndim != 2 or grid.shape[1] != 9 or max(grid.shape) > 30:
        return False
    if len(set(row_gaps[:period])) < 2:
        return False

    try:
        if _infer_vertical_period(grid, BLUE) != period:
            return False
        out = main(grid)
    except Exception:
        return False

    if out.shape != (TARGET_HEIGHT, 9):
        return False
    if set(map(int, np.unique(out))) - {BG, RED}:
        return False
    if np.array_equal(out, _visible_shortcut(grid)):
        return False

    visible_h = _occupied_prefix_height(grid, BLUE)
    if not np.any(np.count_nonzero(out[visible_h:] == RED, axis=1) > 2):
        return False

    tiled_pairs = [_row_span(out[r], RED) for r in range(TARGET_HEIGHT)]
    gaps = [c1 - c0 - 1 for c0, c1 in tiled_pairs]
    min_gap = min(gaps)
    for r, (c0, c1) in enumerate(tiled_pairs):
        filled = np.flatnonzero(out[r] == RED)
        if gaps[r] == min_gap:
            if not np.array_equal(filled, np.arange(c0, c1 + 1)):
                return False
        else:
            if not np.array_equal(filled, np.array([c0, c1])):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        period = random.choice((2, 3))
        if period == 2:
            visible_h = random.choice((3, 4))
            base_gaps = _sample_base_gaps(period)
            row_gaps = [base_gaps[r % period] for r in range(visible_h)]
        else:
            visible_h = 4
            base_gaps = _sample_base_gaps(period)
            row_gaps = [base_gaps[0], base_gaps[1], base_gaps[2], base_gaps[0]]

        total_h = random.randint(visible_h + 1, 7)
        grid = _render_input(total_h, row_gaps)
        if _validate_instance(grid, period, row_gaps):
            return grid

    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
