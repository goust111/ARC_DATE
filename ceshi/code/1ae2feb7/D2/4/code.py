# Variant of 1ae2feb7
# Dimension: D2 — 轨迹涂色模式 (trajectory)
# Change: periodic_pattern——以间隔=色条像素数在ext_side上涂单色点（位置0, length, 2*len... → 递减间隔模式

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D2 variant
# a) Concrete behavior:
#    The separator is still a mostly solid vertical column, and the bar side still contains 1-2
#    monochrome horizontal bars per active row. The only changed object is the dot trail on the
#    extension side. Ext-side coordinates are measured from the first cell past the separator:
#    old rule = 0, L, 2L, ... ; new rule = L, L+(L-1), L+(L-1)+(L-2), ... until the next addend
#    would be 0 or the point would leave the grid. Example with L=4 on the left-bar variant:
#    old absolute columns = sep+1, sep+5, sep+9, ... ; new absolute columns = sep+5, sep+8, sep+10.
#    This creates a visibly tightening trail rather than a constant-period comb.
# b) Complexity calibration:
#    The core objects and ordering stay unchanged, so the task is still recognizable as the same
#    family. To make the new trail visible, generation rejects cases where nothing is painted or
#    where every row behaves too similarly to the old periodic rule.
# c) Main() assumption audit:
#    - _find_sep_col still assumes one mostly solid single-color separator column; unchanged.
#    - Mass-based side detection still assumes the extension side is empty in the input; generator
#      preserves that.
#    - Run detection on the bar side still assumes horizontal monochrome runs; unchanged.
#    - The old solver's `range(..., step=length)` depended on two old facts: first paint at ext
#      position 0 and constant stride `length`. Both are false now, so that block must be replaced
#      by an explicit decreasing-gap iterator. The no-overwrite rule remains valid.
# d) Placement legality:
#    Bars are still placed fully inside one side of the separator, optionally with a 1-cell gap to
#    the separator, and adjacent bars on the same row remain contiguous. Trail points are produced
#    only if they stay within the extension side. Generator validates `main(input) == output`.
# e) Visual observability:
#    The new rule is directly visible because the first painted cell is no longer adjacent to the
#    separator and later painted cells get closer together. Samples are filtered so at least one bar
#    creates a multi-point decreasing trail and the overall output differs from the original periodic
#    extension on the same input.
# f) Visual cleanliness:
#    Only one overlay system exists on the extension side, and the existing nearest-first no-overwrite
#    order prevents ambiguous mixing. Active rows are spaced apart exactly as in the base task.
# g) Color inference:
#    Colors are unchanged: each trail keeps the source bar color, with no hidden mapping or color-id
#    dependent behavior. The only learned change is geometric.


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


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


def _iter_decreasing_abs_positions(
    sep_col: int,
    ext_len: int,
    bars_on_left: bool,
    length: int,
):
    """Yield absolute trail cells for ext positions L, L+(L-1), ... ."""
    pos = length
    next_gap = length - 1
    while pos < ext_len:
        if bars_on_left:
            yield sep_col + 1 + pos
        else:
            yield sep_col - 1 - pos
        if next_gap <= 0:
            break
        pos += next_gap
        next_gap -= 1


def _paint_decreasing(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
) -> int:
    """Paint one bar's decreasing-gap trail; return number of newly painted cells."""
    ext_len = out.shape[1] - sep_col - 1 if bars_on_left else sep_col
    painted = 0
    for cc in _iter_decreasing_abs_positions(sep_col, ext_len, bars_on_left, length):
        if int(out[row, cc]) == bg:
            out[row, cc] = color
            painted += 1
    return painted


def _paint_periodic(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
) -> int:
    """Old-rule helper used only to reject degenerate generator outputs."""
    painted = 0
    if bars_on_left:
        positions = range(sep_col + 1, out.shape[1], length)
    else:
        positions = range(sep_col - 1, -1, -length)
    for cc in positions:
        if int(out[row, cc]) == bg:
            out[row, cc] = color
            painted += 1
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = []
    for r in range(h):
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
        if length <= 0:
            continue
        _paint_decreasing(out, r, sep_col, bars_on_left, length, col, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MIN_TOTAL_TRAIL_CELLS = 3
MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _count_decreasing_positions(ext_len: int, length: int) -> int:
    return sum(1 for _ in _iter_decreasing_abs_positions(0, ext_len, True, length))


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    need_multi_point: bool,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (length, color) ordered near->far."""
    for _ in range(80):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        if capacity <= 0:
            continue
        two_bars = rng.random() < P_TWO_BARS and capacity >= 2

        if not two_bars:
            length = int(rng.integers(1, capacity + 1))
            bars = [(length, primary)]
        else:
            near_len = int(rng.integers(1, capacity))
            far_len = int(rng.integers(1, capacity - near_len + 1))
            others = [c for c in palette if c != primary]
            colors = [primary, int(rng.choice(others))]
            if rng.random() < P_SWAP_COLORS:
                colors.reverse()
            bars = [(near_len, colors[0]), (far_len, colors[1])]

        if not need_multi_point:
            return bars, sep_gap
        if any(_count_decreasing_positions(ext_side_len, length) >= 2 for length, _ in bars):
            return bars, sep_gap

    sep_gap = 0
    capacity = bar_side_len
    forced_len = min(capacity, max(2, ext_side_len - 1))
    return [(forced_len, primary)], sep_gap


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
    bg: int,
) -> int:
    painted = 0
    for length, color in bars:
        painted += _paint_decreasing(out, row, sep_col, bars_on_left, length, color, bg)
    return painted


def _paint_extension_periodic(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
) -> int:
    painted = 0
    for length, color in bars:
        painted += _paint_periodic(out, row, sep_col, bars_on_left, length, color, bg)
    return painted


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - bar_side_len

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    total_painted = 0
    has_multi_point_trail = False

    forced_row = active_rows[int(rng.integers(0, len(active_rows)))]
    for idx, row in enumerate(active_rows):
        need_multi_point = row == forced_row
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            need_multi_point=need_multi_point,
        )
        row_bars_map[row] = bars
        if any(_count_decreasing_positions(ext_side_len, length) >= 2 for length, _ in bars):
            has_multi_point_trail = True
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        total_painted += _paint_extension(out, row, sep_col, bars_on_left, bars, bg=0)

    if total_painted < MIN_TOTAL_TRAIL_CELLS or not has_multi_point_trail:
        return None

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    periodic_out = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension_periodic(periodic_out, row, sep_col, bars_on_left, bars, bg=0)
    if np.array_equal(periodic_out, out):
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
