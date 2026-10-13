from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg


PANEL_SIZE_RANGE = (3, 5)
PANEL_DENSITY_RANGE = (0.28, 0.62)
MAX_INTERSECTION_FRACTION = 0.34
GENERATION_ATTEMPTS = 400
NON_BLACK_COLORS = np.array(Color.NOT_BLACK, dtype=int)


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    height, width = map(int, grid.shape)
    separator_rows = []
    for r in range(height):
        row_color = int(grid[r, 0])
        if row_color == bg:
            continue
        if np.all(grid[r] == row_color) and int(np.count_nonzero(grid == row_color)) == width:
            separator_rows.append(r)
    assert len(separator_rows) == 1, "expected exactly one unique solid separator row"
    return int(separator_rows[0])


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    separator_row = _find_separator_row(g, bg)
    top = g[:separator_row]
    bottom = g[separator_row + 1 :]
    assert top.shape == bottom.shape, "expected equal-size panels above and below the separator"
    assert top.size > 0, "expected non-empty panels"
    rows, cols = map(int, top.shape)
    assert rows == cols, "expected square panels"

    separator_color = int(g[separator_row, 0])
    top_colors = sorted(int(c) for c in np.unique(top) if int(c) != bg)
    bottom_colors = sorted(int(c) for c in np.unique(bottom) if int(c) != bg)
    assert len(top_colors) == 1, "expected one foreground color in the top panel"
    assert len(bottom_colors) == 1, "expected one foreground color in the bottom panel"
    top_color = int(top_colors[0])
    bottom_color = int(bottom_colors[0])
    assert len({separator_color, top_color, bottom_color}) == 3, "panel and separator colors must be distinct"

    return top, bottom, separator_color, top_color, bottom_color


def _render_output(top: np.ndarray, bottom: np.ndarray, separator_color: int) -> np.ndarray:
    n = int(top.shape[0])
    top_color = int(np.max(top))
    bottom_color = int(np.max(bottom))
    top_mask = top == top_color
    bottom_mask = bottom == bottom_color

    top_stamp = top
    bottom_stamp = bottom
    intersection_stamp = np.zeros_like(top, dtype=int)
    intersection_stamp[top_mask & bottom_mask] = int(separator_color)

    out = np.zeros((n * n, n * n), dtype=int)
    for r in range(n):
        for c in range(n):
            rs = slice(r * n, (r + 1) * n)
            cs = slice(c * n, (c + 1) * n)
            top_on = bool(top_mask[r, c])
            bottom_on = bool(bottom_mask[r, c])
            if top_on and not bottom_on:
                out[rs, cs] = top_stamp
            elif bottom_on and not top_on:
                out[rs, cs] = bottom_stamp
            elif top_on and bottom_on:
                out[rs, cs] = intersection_stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand two stacked square panels into a block collage whose block stamp depends on joint panel activity.

    Input structure:
    - The grid has black background and one solid non-black separator row.
    - The separator row splits the grid into two equal square panels of size n x n.
    - The top and bottom panels each contain exactly one distinct foreground color on black.

    Rule:
    - Build a black output of size (n*n) x (n*n), partitioned into n x n blocks of size n x n.
    - For panel position (r, c), inspect whether the top panel, bottom panel, both, or neither are foreground there.
    - Stamp the whole top panel into that block for top-only, the whole bottom panel for bottom-only, their foreground intersection recolored to the separator color for both, and leave the block black for neither.

    Key insight:
    - The same two panels serve simultaneously as placement masks and as stamp templates, while the separator color is reused only for the overlap stamp.
    """
    top, bottom, separator_color, _top_color, _bottom_color = _split_input(grid)
    return _render_output(top, bottom, separator_color)


def _density_bounds(total: int) -> tuple[int, int]:
    lo = max(3, int(np.ceil(total * PANEL_DENSITY_RANGE[0])))
    hi = int(np.floor(total * PANEL_DENSITY_RANGE[1]))
    return lo, hi


def _category_count_options(n: int) -> list[tuple[int, int, int, int]]:
    total = n * n
    min_fg, max_fg = _density_bounds(total)
    max_both = max(1, int(np.floor(total * MAX_INTERSECTION_FRACTION)))
    options = []
    for top_only in range(1, total - 2):
        for bottom_only in range(1, total - top_only - 1):
            for both in range(1, min(max_both, total - top_only - bottom_only - 1) + 1):
                none = total - top_only - bottom_only - both
                if none < 1:
                    continue
                top_fg = top_only + both
                bottom_fg = bottom_only + both
                if not (min_fg <= top_fg <= max_fg and min_fg <= bottom_fg <= max_fg):
                    continue
                if none > max_fg:
                    continue
                options.append((top_only, bottom_only, both, none))
    return options


def _build_status_grid(n: int) -> np.ndarray:
    counts = _category_count_options(n)
    assert counts, f"no valid category counts for n={n}"
    top_only, bottom_only, both, none = counts[np.random.randint(0, len(counts))]
    values = np.array([1] * top_only + [2] * bottom_only + [3] * both + [0] * none, dtype=int)
    np.random.shuffle(values)
    return values.reshape(n, n)


def _panel_masks_from_status(status: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top_mask = (status == 1) | (status == 3)
    bottom_mask = (status == 2) | (status == 3)
    return top_mask, bottom_mask


def _mask_is_rich(mask: np.ndarray) -> bool:
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    if np.count_nonzero(row_counts) < 2 or np.count_nonzero(col_counts) < 2:
        return False
    if not np.any((row_counts > 0) & (row_counts < mask.shape[1])):
        return False
    if not np.any((col_counts > 0) & (col_counts < mask.shape[0])):
        return False
    return True


def _status_is_semantic(status: np.ndarray) -> bool:
    counts = np.bincount(status.ravel(), minlength=4)
    if np.any(counts == 0):
        return False
    top_mask, bottom_mask = _panel_masks_from_status(status)
    if not _mask_is_rich(top_mask) or not _mask_is_rich(bottom_mask):
        return False
    if np.array_equal(top_mask, bottom_mask):
        return False
    if np.array_equal(top_mask, ~bottom_mask):
        return False
    if counts[3] == 1 and status.shape[0] >= 4:
        return False
    return True


def _build_panels(status: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    separator_color, top_color, bottom_color = map(int, colors)
    top_mask, bottom_mask = _panel_masks_from_status(status)
    top = np.zeros(status.shape, dtype=int)
    bottom = np.zeros(status.shape, dtype=int)
    top[top_mask] = top_color
    bottom[bottom_mask] = bottom_color
    return top, bottom, separator_color


def _semantic_validation(grid: np.ndarray) -> bool:
    try:
        top, bottom, separator_color, top_color, bottom_color = _split_input(grid)
    except AssertionError:
        return False

    n = int(top.shape[0])
    top_mask = top == top_color
    bottom_mask = bottom == bottom_color
    status = np.zeros((n, n), dtype=int)
    status[top_mask & ~bottom_mask] = 1
    status[~top_mask & bottom_mask] = 2
    status[top_mask & bottom_mask] = 3

    if not _status_is_semantic(status):
        return False

    expected = _render_output(top, bottom, separator_color)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    intersection_stamp = np.zeros_like(top, dtype=int)
    intersection_stamp[top_mask & bottom_mask] = separator_color
    block_map = {
        0: np.zeros_like(top, dtype=int),
        1: top,
        2: bottom,
        3: intersection_stamp,
    }
    seen = set()
    for r in range(n):
        for c in range(n):
            block = actual[r * n : (r + 1) * n, c * n : (c + 1) * n]
            status_value = int(status[r, c])
            if not np.array_equal(block, block_map[status_value]):
                return False
            seen.add(status_value)
    return seen == {0, 1, 2, 3}


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(PANEL_SIZE_RANGE[0], PANEL_SIZE_RANGE[1] + 1))
    colors = np.random.choice(NON_BLACK_COLORS, size=3, replace=False)

    for _ in range(80):
        status = _build_status_grid(n)
        if not _status_is_semantic(status):
            continue
        top, bottom, separator_color = _build_panels(status, colors)
        separator = np.full((1, n), separator_color, dtype=int)
        grid = np.vstack([top, separator, bottom])
        if _semantic_validation(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 052/1")
