from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the top palette row and redraw it below as a widening wedge centered on its missing cell.

    Input structure
    - The top row is a palette row with exactly one background gap.
    - Below it, the only foreground is one vertical marker bar of a single color.
    - The marker bar is either in the gap column or immediately beside it.

    Rule
    - Clear the grid to background.
    - Starting at the top of the marker bar, each next row copies one more
      column of the top-row palette on both sides of the gap.
    - Fill the gap column with the marker color, and keep the vertical marker bar.

    Key insight
    - The lone background cell in the top row is the center of the expanding
      copy; the lower vertical bar provides the center color and the row where
      the expansion begins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    bg = get_bg(g)
    top = g[0, :]
    gap_cols = np.flatnonzero(top == bg)
    assert gap_cols.size == 1, "expected exactly one background gap in the top row"
    gap_col = int(gap_cols[0])

    lower_coords = np.argwhere(g[1:, :] != bg)
    assert lower_coords.size > 0, "expected a lower marker bar"
    lower_coords[:, 0] += 1

    marker_colors = {int(g[r, c]) for r, c in lower_coords}
    marker_cols = {int(c) for _r, c in lower_coords}
    assert len(marker_colors) == 1, "expected one lower marker color"
    assert len(marker_cols) == 1, "expected one lower marker column"
    marker = marker_colors.pop()
    line_col = marker_cols.pop()
    line_start = int(lower_coords[:, 0].min())

    assert np.all(g[1:line_start, :] == bg), "rows before the marker bar must be background"
    assert np.all(g[line_start:, line_col] == marker), "marker bar must be solid"
    assert lower_coords.shape[0] == height - line_start, "marker bar must be the only lower foreground"
    assert line_col == gap_col or abs(line_col - gap_col) == 1
    if line_col != gap_col:
        assert top[line_col] == marker
    assert np.all(top[np.arange(width) != gap_col] != bg), "only the gap may be background in the palette row"

    out = np.full_like(g, bg)
    out[line_start:, line_col] = marker

    for r in range(line_start, height):
        radius = r - line_start
        c0 = max(0, gap_col - radius)
        c1 = min(width - 1, gap_col + radius)
        for c in range(c0, c1 + 1):
            if radius == 0 and c == gap_col and line_col != gap_col:
                continue
            out[r, c] = marker if c == gap_col else int(top[c])
        out[r, line_col] = marker

    return out

# --------------- generator (att2) ---------------
# Canvas, structure, and retries.
_CANVAS_SIZES = (9, 11, 13)
_LINE_START_MIN = 2
_MAX_ATTEMPTS = 300
# Color roles.
_BACKGROUND, _MARKER = 5, 8
_PALETTE_COLORS = (0, 1, 2, 3, 4, 6, 7, 8, 9)


def _gen_expected_output(inp: np.ndarray, line_start: int) -> np.ndarray:
    n = inp.shape[0]
    gap_col = n // 2
    out = np.full_like(inp, _BACKGROUND)
    top = inp[0]

    for r in range(line_start, n):
        radius = r - line_start
        c0 = max(0, gap_col - radius)
        c1 = min(n - 1, gap_col + radius)
        for c in range(c0, c1 + 1):
            out[r, c] = _MARKER if c == gap_col else int(top[c])

    return out


def _make_input_grid(n: int, line_start: int) -> np.ndarray:
    gap_col = n // 2
    inp = np.full((n, n), _BACKGROUND, dtype=int)
    for c in range(n):
        if c != gap_col:
            inp[0, c] = random.choice(_PALETTE_COLORS)
    inp[line_start:, gap_col] = _MARKER
    return inp


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_CANVAS_SIZES)
    line_start = random.randint(_LINE_START_MIN, n // 2 - 1)
    inp = _make_input_grid(n, line_start)
    expected = _gen_expected_output(inp, line_start)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bae5c565 input")
