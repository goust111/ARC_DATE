# Variant of 1ae2feb7
# Dimension: D9 — 分隔条结构 (obstacle)
# Change: 单条垂直分隔列——一整列（允许缺1个格子）填充同色像素，将网格分为bar_side和ext_side... → 分隔条有缺口/洞

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# -----------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# variant_spec:
#   core_change:
#     original: a nearly solid separator column gates all bar rows equally
#     variant: the separator column is sparse, with multiple obvious holes; only
#       rows that actually contain a separator pixel are allowed to emit onto the
#       extension side. Rows whose bar faces a separator hole keep the target
#       side blank.
#   concrete_shape:
#     - All bars still live on one side of a vertical separator position.
#     - The extension side is still completely blank in the input.
#     - Some bar rows contain the separator color at the separator column
#       ("emitting rows"), and those rows paint periodic dots/stripes exactly as
#       in the original task.
#     - Other bar rows have a hole at that same separator column ("blocked
#       rows"). Those rows still contain bars, but produce no extension at all.
#     - Because bar rows are spaced apart, the separator column visibly appears
#       as a broken vertical dotted line with several gaps.
#   behavior_trace:
#     - Original logic on a bar row of length 3: paint target cells at offsets
#       0, 3, 6, ...
#     - Variant logic on the same row:
#         * if separator pixel exists in that row: same as original
#         * if separator cell is a hole: paint nothing on the target side
#     - Therefore the changed control point is not the periodic rule itself, but
#       the per-row permission to apply it.
#   complexity_calibration:
#     - To make the new rule visible, generation must include both emitting rows
#       and blocked rows; otherwise holes would only appear on already-empty rows
#       and the task would visually collapse back to the original.
#     - The number of bar rows stays small (4-5), with 2-4 emitting rows. This
#       keeps examples readable while guaranteeing repeated support for the new
#       gating rule.
#   implicit_assumptions_review:
#     - Old `_find_sep_col` assumed the separator occupied almost the whole
#       column (`vals.size >= h - 1`). That fails immediately here.
#     - Old row loop processed every row containing bars. That is no longer
#       valid; we must process only rows where `grid[r, sep_col]` is non-bg.
#     - Side detection by empty extension side still holds, so we preserve the
#       original left-vs-right behavior once the separator location is found.
#   legality:
#     - Bars remain confined to one side of the separator position.
#     - The extension side remains fully blank in the input.
#     - Rows are spaced by at least one empty row, so the broken separator is
#       legible and bars do not vertically merge into clutter.
#   observability:
#     - Humans can compare two similar bar rows and notice that only the rows
#       aligned with separator pixels project outward.
#     - No invisible ordering or color-id arithmetic is introduced.
#   new_solver_assumptions:
#     - The separator color is the only non-bg color that appears in multiple
#       rows while staying in a single column.
#     - The extension side is completely blank in the input.
#     - `generate_input()` guarantees both conditions by assigning every bar
#       segment a globally unique color and clearing the full extension side.
# -----------------------------------------------------------------------------


def _find_sep_info(g: np.ndarray, bg: int) -> tuple[int | None, int | None, bool | None]:
    """Find the sparse separator using repeated single-column color plus a blank ext side."""
    h, w = g.shape
    for color in colors_present(g, exclude=[bg]):
        coords = np.argwhere(g == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) < 2 or len(cols) != 1:
            continue
        c = int(cols[0])
        if c + 1 < w and np.all(g[:, c + 1 :] == bg):
            return c, int(color), True
        if c > 0 and np.all(g[:, :c] == bg):
            return c, int(color), False
    return None, None, None


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if row[c] == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and row[c] == col:
            c += 1
        yield c0, c - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    sep_col, sep_color, bars_on_left = _find_sep_info(g, bg)
    if sep_col is None or sep_color is None or bars_on_left is None:
        return g.copy()

    out = g.copy()

    bars = []
    sep_rows = np.flatnonzero(g[:, sep_col] == sep_color)
    for r in map(int, sep_rows):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

BAR_ROWS_MIN, BAR_ROWS_MAX = 4, 5
EMIT_ROWS_MIN = 2

P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_colors(rng: np.random.Generator, seg_count: int) -> tuple[int, list[int]]:
    """Pick separator color and globally unique bar colors."""
    sep_color = int(rng.integers(1, 10))
    pool = [c for c in range(1, 10) if c != sep_color]
    seg_colors = list(map(int, rng.choice(pool, size=seg_count, replace=False)))
    return sep_color, seg_colors


def _sample_row_shapes(
    rng: np.random.Generator,
    active_count: int,
    bar_side_len: int,
) -> list[tuple[list[int], int]]:
    """
    Return per-row bar lengths (near->far) and separator gap.
    Total segment count is capped at 8 so every segment can have a unique color.
    """
    extra_budget = min(active_count, 8 - active_count)
    two_bar_rows = set(map(int, rng.choice(active_count, size=int(rng.integers(0, extra_budget + 1)), replace=False)))
    shapes: list[tuple[list[int], int]] = []
    for idx in range(active_count):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        if idx not in two_bar_rows:
            shapes.append(([int(rng.integers(1, capacity + 1))], sep_gap))
            continue
        near_len = int(rng.integers(1, capacity))
        far_len = int(rng.integers(1, capacity - near_len + 1))
        lengths = [near_len, far_len]
        if rng.random() < P_SWAP_COLORS:
            lengths = lengths[::-1]
        shapes.append((lengths, sep_gap))
    return shapes


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            out[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = color
            cursor += length


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    max_rows = min(BAR_ROWS_MAX, (h + 1) // 2)
    active_count = int(rng.integers(BAR_ROWS_MIN, max_rows + 1))
    active_rows = _choose_active_rows(rng, h, active_count)

    emit_count = int(rng.integers(EMIT_ROWS_MIN, active_count))
    emit_rows = set(map(int, rng.choice(active_rows, size=emit_count, replace=False)))
    blocked_rows = [r for r in active_rows if r not in emit_rows]
    if not blocked_rows:
        return None

    row_shapes = _sample_row_shapes(rng, active_count, bar_side_len)
    seg_count = sum(len(lengths) for lengths, _ in row_shapes)
    sep_color, seg_colors = _sample_colors(rng, seg_count)
    color_cursor = 0

    out = np.zeros((h, w), dtype=int)
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for row, (lengths, sep_gap) in zip(active_rows, row_shapes):
        colors = seg_colors[color_cursor : color_cursor + len(lengths)]
        color_cursor += len(lengths)
        bars = list(zip(lengths, colors))
        row_bars_map[int(row)] = bars
        _place_bars(out, int(row), sep_col, bars_on_left, bars, sep_gap)
        if row in emit_rows:
            out[int(row), sep_col] = sep_color

    for row in emit_rows:
        _paint_extension(out, int(row), sep_col, bars_on_left, row_bars_map[int(row)])

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if int(np.count_nonzero(inp[:, sep_col] == sep_color)) < EMIT_ROWS_MIN:
        return None
    if not all(inp[r, sep_col] == 0 for r in blocked_rows):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
