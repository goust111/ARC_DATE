from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift alternating two-color row bars so each row aligns to the topmost bar's color columns.

    Input structure:
    - The dominant background surrounds horizontal two-pixel bars.
    - Every foreground row has exactly one adjacent bar using the same two colors.
    - The topmost bar gives the reference left/right column for each color.

    Rule:
    - Keep bars whose left color is already in its reference column.
    - For a reversed bar, shift the whole two-pixel bar horizontally until its
      left color lands in that color's reference column.
    - Background cells vacated by shifted bars become background.

    Key insight:
    - The rows are not recolored; each two-color bar is translated so equal
      colors line up with the topmost reference bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 2, "expected exactly two foreground colors"

    foreground_rows: list[tuple[int, np.ndarray]] = []
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] != bg)
        if cols.size:
            foreground_rows.append((int(r), cols))
    assert foreground_rows, "expected at least one foreground row"

    ref_r, ref_cols = foreground_rows[0]
    assert ref_cols.size == 2, "reference row must contain a two-pixel bar"
    assert int(ref_cols[1]) == int(ref_cols[0]) + 1, "reference bar must be adjacent"
    ref_values = tuple(map(int, g[ref_r, ref_cols]))
    assert ref_values[0] != ref_values[1], "reference bar must use two colors"
    assert set(ref_values) == set(foreground_colors), "reference bar must use all foreground colors"
    ref_left = int(ref_cols[0])
    ref_color_cols = {ref_values[0]: ref_left, ref_values[1]: int(ref_cols[1])}

    out = g.copy()
    for r, cols in foreground_rows:
        assert cols.size == 2, "each foreground row must contain one two-pixel bar"
        assert int(cols[1]) == int(cols[0]) + 1, "each row bar must be adjacent"
        assert int(cols[0]) == ref_left, "input bars must start in the same column"

        values = tuple(map(int, g[r, cols]))
        assert set(values) == set(ref_values), "each row bar must use the reference colors"
        shift = ref_color_cols[values[0]] - int(cols[0])
        dst_cols = cols + shift
        assert int(dst_cols[0]) >= 0 and int(dst_cols[-1]) < g.shape[1], "shifted bar must fit"

        out[r, cols] = bg
        out[r, dst_cols] = values

    return out

# --------------- generator (att2) ---------------
# Canvas, palette, geometry, and retry bounds.
GRID_SIZE_RANGE = (4, 6)
ARC_COLOR_COUNT = 10
BAR_WIDTH = 2
SHIFTED_BAR_WIDTH = 3
MIN_PADDED_BAR_COUNT = 3
MAX_GENERATION_ATTEMPTS = 300

def _sample_layout() -> tuple[int, int, int, int]:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    possible_counts = [h]
    if h - 2 >= MIN_PADDED_BAR_COUNT:
        possible_counts.append(h - 2)
    bar_count = random.choice(possible_counts)
    top_pad = (h - bar_count) // 2
    return h, w, bar_count, top_pad


def _gen_attempt() -> np.ndarray | None:
    h, w, bar_count, top_pad = _sample_layout()
    colors = random.sample(range(ARC_COLOR_COUNT), 3)
    bg, color_a, color_b = colors
    bar_col = random.randint(0, w - SHIFTED_BAR_WIDTH)
    out = np.full((h, w), bg, dtype=int)
    inp = out.copy()
    for i in range(bar_count):
        r = top_pad + i
        if i % 2 == 0:
            out[r, bar_col : bar_col + BAR_WIDTH] = [color_a, color_b]
            inp[r, bar_col : bar_col + BAR_WIDTH] = [color_a, color_b]
        else:
            out[r, bar_col + 1 : bar_col + SHIFTED_BAR_WIDTH] = [color_b, color_a]
            inp[r, bar_col : bar_col + BAR_WIDTH] = [color_b, color_a]
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
