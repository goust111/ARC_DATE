from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint every completely blank row and column red while preserving the foreground pattern.

    Input structure
    - The grid contains one background color and one foreground color.
    - The background color is the only color that forms at least one full row or full column.
    - Red is absent from the input and is reserved for the output guide lines.

    Rule
    - Find all rows and columns made entirely of the background color.
    - Recolor every cell in those full blank rows and columns red.
    - Leave all other cells unchanged.

    Key insight
    - The blank rows and columns are missing guide lines; because they contain no foreground pixels, they can be detected directly as full same-color lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == Color.RED), "red must be absent from the input"

    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in colors_present(g):
        row_mask = np.all(g == color, axis=1)
        col_mask = np.all(g == color, axis=0)
        if np.any(row_mask) or np.any(col_mask):
            candidates.append((color, row_mask, col_mask))

    assert len(candidates) == 1, "expected exactly one color to form complete rows or columns"
    bg, blank_rows, blank_cols = candidates[0]
    assert len(colors_present(g, exclude={bg})) == 1, "expected one foreground color"
    assert np.any(blank_rows) or np.any(blank_cols), "expected at least one blank row or column"

    target = blank_rows[:, None] | blank_cols[None, :]
    assert np.all(g[target] == bg), "blank-line mask must contain only background cells"

    out = g.copy()
    out[target] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and retry cap; randint upper bounds are exclusive.
GRID_H_RANGE = (12, 28)
GRID_W_RANGE = (14, 26)
GENERATION_ATTEMPTS = 300

# Input palette.
BACKGROUND = Color.BLACK
GUIDE = Color.RED
FG_COLORS = [c for c in Color.NOT_BLACK if c != Color.RED]

# Blank-line counts.
BLANK_ROW_RANGE = (0, 4)
BLANK_COL_RANGE = (1, 3)
BLANK_COLS_WITHOUT_ROWS = 2

# Foreground density over nonblank row/column intersections.
FG_DENSITY_RANGE = (0.68, 0.72)

def _gen_positive_composition(total: int, parts: int) -> list[int]:
    """Split total into positive random parts."""
    if parts == 1:
        return [total]
    cuts = np.sort(np.random.choice(np.arange(1, total), parts - 1, replace=False))
    return np.diff(np.r_[0, cuts, total]).astype(int).tolist()


def _gen_blank_positions(length: int, count: int) -> list[int]:
    """Return blank indices with positive nonblank gaps between edges and lines."""
    if count == 0:
        return []
    segments = np.array(_gen_positive_composition(length - count, count + 1), dtype=int)
    return (np.cumsum(segments[:-1]) + np.arange(count)).tolist()


def _sample_layout():
    """Choose canvas, color, blank lines, and the surviving row/column indices."""
    h = int(np.random.randint(*GRID_H_RANGE))
    w = int(np.random.randint(*GRID_W_RANGE))
    fg = int(np.random.choice(FG_COLORS))
    blank_row_count = int(np.random.randint(*BLANK_ROW_RANGE))
    blank_col_count = BLANK_COLS_WITHOUT_ROWS
    if blank_row_count != 0:
        blank_col_count = int(np.random.randint(*BLANK_COL_RANGE))
    blank_rows = set(_gen_blank_positions(h, blank_row_count))
    blank_cols = set(_gen_blank_positions(w, blank_col_count))
    non_blank_rows = [r for r in range(h) if r not in blank_rows]
    non_blank_cols = [c for c in range(w) if c not in blank_cols]
    return h, w, fg, blank_rows, blank_cols, non_blank_rows, non_blank_cols


def _build_foreground_mask(nr: int, nc: int, target_fg: int) -> np.ndarray | None:
    """Create a foreground mask with at least one foreground/background per row/column."""
    reserved_bg = np.zeros((nr, nc), dtype=bool)
    reserved_bg[np.arange(nr), np.random.randint(nc, size=nr)] = True
    row_order = np.random.permutation(nr)
    reserved_bg[row_order[np.arange(nc) % nr], np.arange(nc)] = True

    available = ~reserved_bg
    if target_fg > int(np.sum(available)):
        return None

    fg_mask = np.zeros((nr, nc), dtype=bool)
    for r in range(nr):
        choices = np.flatnonzero(available[r])
        if choices.size == 0:
            return None
        fg_mask[r, int(np.random.choice(choices))] = True

    for c in range(nc):
        if not np.any(fg_mask[:, c]):
            choices = np.flatnonzero(available[:, c])
            if choices.size == 0:
                return None
            fg_mask[int(np.random.choice(choices)), c] = True

    current_fg = int(np.sum(fg_mask))
    if current_fg > target_fg:
        return None
    remaining = np.argwhere(available & ~fg_mask)
    need = target_fg - current_fg
    if need:
        if need > len(remaining):
            return None
        chosen = np.random.choice(len(remaining), need, replace=False)
        coords = remaining[chosen]
        fg_mask[coords[:, 0], coords[:, 1]] = True
    return fg_mask


def _gen_attempt() -> np.ndarray | None:
    """Assemble one candidate and reject it if the solver oracle disagrees."""
    h, w, fg, blank_rows, blank_cols, non_blank_rows, non_blank_cols = _sample_layout()
    nr, nc = len(non_blank_rows), len(non_blank_cols)
    area = nr * nc
    lo = int(np.ceil(FG_DENSITY_RANGE[0] * area))
    hi = int(np.floor(FG_DENSITY_RANGE[1] * area))
    if lo > hi:
        return None
    target_fg = int(np.random.randint(lo, hi + 1))

    fg_mask = _build_foreground_mask(nr, nc, target_fg)
    if fg_mask is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[np.ix_(non_blank_rows, non_blank_cols)] = np.where(fg_mask, fg, BACKGROUND)
    expected = grid.copy()
    line_mask = np.zeros((h, w), dtype=bool)
    if blank_rows:
        line_mask[list(blank_rows), :] = True
    if blank_cols:
        line_mask[:, list(blank_cols)] = True
    expected[line_mask] = GUIDE

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c1d99e64 input")
