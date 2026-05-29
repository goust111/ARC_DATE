# Variant of 1ae2feb7
# Dimension: D7 — 多步处理顺序 (multi_step)
# Change: ordered_processing——按色条距分隔条的距离从近到远排序处理，先涂的不被后涂的覆盖（no-overwrite规则） → 从远到近处理（反转顺序）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D7 variant
# a) Concrete behavior and trace
#    The grid still has one vertical separator column, bars on exactly one side, and an empty
#    extension side in the input. Each row's bars still project periodic dots starting at the
#    cell immediately beyond the separator. The only change is priority:
#    - original: nearest bar paints first, farther bars can only fill remaining zeros
#    - variant: farthest bar paints first, nearer bars can only fill remaining zeros
#    Example on the left-bar variant, with one active row "... AA BBB | 0 0 0 0 0 0":
#    near bar = BBB (length 3), far bar = AA (length 2). Both try to paint the first extension
#    cell. Original order gives positions 1,4,... to BBB before AA runs; variant gives that
#    first cell to AA, and BBB only fills still-empty positions on its own period. The visible
#    difference is therefore exactly on overlap cells, especially the first extension cell.
# b) Complexity calibration
#    The object vocabulary is unchanged: 1 separator, 2-5 active rows, 1-2 adjacent bars per
#    row. To make the D7 change observable, the generator guarantees at least one witness row
#    with two bars of distinct colors. That keeps the task close to the source while ensuring
#    at least one row where the priority reversal changes the output.
# c) Hidden-assumption audit on the original main()
#    - _find_sep_col: only assumes one mostly-filled monochrome separator column; still valid.
#    - bars_on_left via non-bg mass: still valid because the input extension side remains empty.
#    - run extraction: still valid because bars remain horizontal monochrome runs.
#    - original ascending sort by distance encoded the old rule; this is the one line of logic
#      that directly depends on the old invariant and must be reversed.
#    - no-overwrite checks remain correct and are exactly what makes the order change visible.
# d) Placement legality
#    Bars are still placed contiguously on one side of the separator, with total occupied width
#    <= bar-side width and optional separator gap <= 1. Extension dots are only painted on the
#    opposite side, so no overlaps with bars or separator occur. Output-first generation remains
#    valid because the input is obtained by zeroing the extension side.
# e) Visual observability
#    A two-bar witness row always reveals the change because both bars paint the first extension
#    cell and their colors differ. When lengths differ, more overlap cells inside the periodic
#    pattern reinforce the far-first rule. This avoids invisible behavior changes.
# f) Visual cleanliness
#    Only one row's own bars interact with that row's extension side. Rows are separated by at
#    least one blank row, so overlaps never mix across rows. The grid stays visually partitioned
#    into bar-side, separator, and extension-side regions.
# g) Color inference
#    No sorting or logic depends on color ids. Colors only identify bars visually; the rule is
#    inferred from distance to the separator and overwrite blocking.
# 9) New assumptions introduced by this variant
#    The solver assumes the same spatial layout as the source task. The generator now additionally
#    guarantees at least one two-bar witness row so the reversed priority is exercised, and it
#    rejects samples where far-first and near-first would accidentally agree everywhere.


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


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
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
    return bars


def _solve_with_order(grid: np.ndarray, *, far_first: bool) -> np.ndarray:
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

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    if far_first:
        bars.sort(key=lambda t: (-t[0], t[1], t[2]))
    else:
        bars.sort(key=lambda t: (t[0], t[1], t[2]))

    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[r, cc]) == bg:
                out[r, cc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_order(grid, far_first=True)


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

MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_two_bar_lengths(
    rng: np.random.Generator,
    capacity: int,
    *,
    prefer_distinct: bool,
) -> tuple[int, int]:
    if capacity < 2:
        raise ValueError("capacity must be at least 2 for a two-bar row")
    if prefer_distinct and capacity >= 3:
        for _ in range(16):
            near_len = int(rng.integers(1, capacity))
            far_len = int(rng.integers(1, capacity - near_len + 1))
            if near_len != far_len:
                return near_len, far_len
    near_len = int(rng.integers(1, capacity))
    far_len = int(rng.integers(1, capacity - near_len + 1))
    return near_len, far_len


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    *,
    force_two_bars: bool = False,
    prefer_distinct_lengths: bool = False,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar-side capacity must stay positive")

    two_bars = force_two_bars or (capacity >= 2 and rng.random() < P_TWO_BARS)
    if not two_bars:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len, far_len = _sample_two_bar_lengths(
        rng,
        capacity,
        prefer_distinct=prefer_distinct_lengths,
    )
    others = [c for c in palette if c != primary]
    far_color = int(rng.choice(others))
    colors = [primary, far_color]
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
    *,
    far_first: bool,
) -> None:
    w = out.shape[1]
    ordered = list(reversed(bars)) if far_first else list(bars)
    for length, color in ordered:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _has_two_bar_witness(row_bars_map: dict[int, list[tuple[int, int]]]) -> bool:
    for bars in row_bars_map.values():
        if len(bars) >= 2 and bars[0][1] != bars[1][1]:
            return True
    return False


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

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    witness_row = int(rng.choice(active_rows))

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        force_two_bars = row == witness_row
        try:
            bars, sep_gap = _sample_row_bars(
                rng,
                palette,
                row_primary[idx],
                bar_side_len,
                force_two_bars=force_two_bars,
                prefer_distinct_lengths=force_two_bars,
            )
        except ValueError:
            return None
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    if not _has_two_bar_witness(row_bars_map):
        return None

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars, far_first=True)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    solved_far = main(inp)
    if not np.array_equal(solved_far, out):
        return None

    solved_near = _solve_with_order(inp, far_first=False)
    if np.array_equal(solved_near, solved_far):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
