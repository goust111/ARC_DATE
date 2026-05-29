# Variant of 1ae2feb7
# Dimension: D1 — 射线路径方向 (path)
# Change: straight_ray——从分隔条出发，沿水平方向向对侧网格边缘延伸（垂直于分隔条方向的直线射线）... → 对角线射线

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D1 variant
# a) Concrete behavior:
#    The separator is still a mostly full vertical column. Bars still live entirely on one side
#    and remain horizontal monochrome runs. The only semantic change is where their periodic marks
#    are written on the opposite side. Instead of painting along the same row, each bar now emits
#    a 45-degree ray that moves one row downward and one column toward the ext side before the
#    first paint, then repeats every `bar_length` diagonal steps. Example with bars on the left:
#    a length-3 bar at row r paints (r+1, sep+1), (r+4, sep+4), (r+7, sep+7), ...
#    The original task would have painted (r, sep+1), (r, sep+4), (r, sep+7), ...
#    The nearest bar in a row is still processed first, so a far bar can only fill holes left on
#    that same diagonal and never overwrites earlier paint.
# b) Complexity calibration:
#    The change is large enough to be visually obvious because output paint leaves the source row
#    and forms slanted traces. It is still controlled: all rays on one side share the same slope
#    (down-right if bars are on the left, down-left if bars are on the right), so there are only
#    two global cases and no hidden per-row branching.
# c) Implicit assumptions from the original solver:
#    - `_find_sep_col` assumes the separator is still a mostly filled uniform vertical column.
#      That remains true.
#    - Side detection by non-bg mass assumes the target side is empty in the input. The generator
#      still clears the entire ext side, so this remains true.
#    - Bar extraction assumes source objects are horizontal runs on a single row. That remains true.
#    - The old painting loop assumed "same row" continuation. That is the exact place that breaks
#      under the variant, so it is replaced by diagonal stepping with bounds checks.
#    - Non-overwrite behavior used to be harmless because all collisions were row-local. With the
#      chosen fixed slope, rays from different source rows are parallel and cannot intersect, so
#      only same-row bars compete, matching the original nearest-first intent.
# d) Placement legality:
#    The generator restricts active rows to `0..h-2`, guaranteeing that every ray has at least the
#    first diagonal step inside the grid. Bar placement on the source side is unchanged, so bars do
#    not overlap and stay within bounds. The ext side is built output-first and then cleared to
#    recover the input, so `main(input) == output` remains the final legality check.
# e) Visual observability:
#    Humans can infer the rule from a few examples because newly colored cells are no longer on the
#    source row; they appear one row lower and continue on a clear 45-degree line. The step size is
#    still readable from repeated gaps along that diagonal.
# f) Visual cleanliness:
#    Using one common slope per side keeps different rows as parallel diagonals, avoiding cluttered
#    crossings. Same-row competition is limited to the original 1-2 adjacent bars.
# g) Color inferability:
#    Colors are not remapped. Each diagonal ray inherits its bar color exactly as before, and the
#    nearest-first overwrite rule is supported by rows that sometimes contain two differently colored
#    adjacent bars.


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


def _iter_diagonal_positions(
    row: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    h: int,
    w: int,
):
    if step <= 0:
        return
    rr = row + 1
    cc = sep_col + 1 if bars_on_left else sep_col - 1
    dc = 1 if bars_on_left else -1
    while 0 <= rr < h and 0 <= cc < w:
        yield rr, cc
        rr += step
        cc += dc * step


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
        for rr, cc in _iter_diagonal_positions(r, sep_col, bars_on_left, length, h, w):
            if int(out[rr, cc]) == bg:
                out[rr, cc] = col

    return out


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

MIN_EXTRA_PAINT = 3
MIN_REPEAT_ROWS = 1
MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, usable_rows: int, count: int) -> list[int]:
    """Pick `count` start rows in [0, usable_rows) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(usable_rows - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        return [], sep_gap

    max_single = min(capacity, 5)
    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS

    if not two_bars:
        length = int(rng.integers(1, max_single + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, min(capacity - 1, 4) + 1))
    far_cap = min(capacity - near_len, 5)
    far_len = int(rng.integers(1, far_cap + 1))
    others = [c for c in palette if c != primary]
    colors = [primary, int(rng.choice(others))]
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])], sep_gap


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
) -> int:
    h, w = out.shape
    painted = 0
    for length, color in bars:
        for rr, cc in _iter_diagonal_positions(row, sep_col, bars_on_left, length, h, w):
            if out[rr, cc] == 0:
                out[rr, cc] = color
                painted += 1
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

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    usable_rows = h - 1
    max_active = min(ACTIVE_MAX, usable_rows // 2 + usable_rows % 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, usable_rows, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_candidates = [r for r in inactive_rows if r == 0 or r == h - 1]
        if not gap_candidates:
            gap_candidates = inactive_rows
        gap_row = int(gap_candidates[int(rng.integers(0, len(gap_candidates)))])
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_paint_counts: dict[int, int] = {}
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        if not bars:
            return None
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        row_paint_counts[row] = _paint_extension(out, row, sep_col, bars_on_left, bars)

    if sum(row_paint_counts.values()) < active_count + MIN_EXTRA_PAINT:
        return None
    if sum(count >= 2 for count in row_paint_counts.values()) < MIN_REPEAT_ROWS:
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
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
