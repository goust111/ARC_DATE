# Variant of 1ae2feb7
# Dimension: D11 — 涂色起始偏移 (trajectory)
# Change: 从位置0开始涂色（ext_side紧邻分隔条的第一个格子），序列为0, length, 2*length, ...... → 从位置length开始（跳过第一个位置）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D11 variant
# -----------------------------------
# a) Concrete behavior and trace
#    Objects stay the same as the base task: one vertical separator column, 2-5
#    active rows, and 1-2 horizontal monochrome bars on the bar side of each
#    active row. The only change is how those bars project onto the empty
#    extension side.
#
#    Small example with separator at column 4, bars on the left, and one bar of
#    length 3 ending next to the separator:
#      input row: 111120000000
#      base rule: paint ext offsets 0,3,6,...  -> cols 5,8,11,...
#      D11 rule:  paint ext offsets 3,6,9,...  -> cols 8,11,...
#    So the first cell after the separator must remain blank, and more generally
#    the first painted cell is length cells away from the separator.
#
#    With two bars, the processing order still matters exactly as before:
#    nearest-to-separator first, then farther ones, never overwriting existing
#    non-background cells. Only the starting offset changes from 0 to `length`.
#
# b) Complexity calibration
#    If we allowed arbitrary bar lengths relative to the extension side, many
#    rows would produce no visible ext-side paint under the new rule (e.g. a bar
#    length equal to the whole ext-side width). That would make the variant hard
#    to infer and too close to the unchanged input. The generator therefore keeps
#    the original geometry but constrains the nearest bar on every active row to
#    satisfy `length < ext_side_len`, guaranteeing at least one visible D11 dot.
#
# c) Implicit-assumption audit for the original solver
#    `_find_sep_col` and `_iter_runs_1d` do not depend on the old offset, so
#    they remain valid.
#    `bars.sort(...)` also remains valid because the variant does not change the
#    "nearest bar wins" rule.
#    The offset-sensitive lines are the original `range(sep_col + 1, w, step)`
#    and `range(sep_col - 1, -1, -step)`: they assume the first ext cell is
#    paintable. Under D11 that assumption is false, so both solver and generator
#    must start at one full bar-length farther away.
#
# d) Placement legality
#    Bars are still placed wholly on one side of the separator, optionally with
#    a 1-cell gap to the separator, and a total width not exceeding the bar-side
#    capacity. The new generator additionally guarantees `nearest_length <
#    ext_side_len`, so at least one ext-side target position exists and all
#    generated rows remain in-bounds.
#
# e) Visual observability
#    The intended visible cue is a blank moat between the separator and the first
#    repeated color on every active row. Because every active row now has at
#    least one visible ext-side placement, that cue appears repeatedly in each
#    generated puzzle instead of only in rare rows.
#
# f) Visual cleanliness
#    The variant does not add new object types or overlays. It only shifts the
#    periodic paint locations outward, so the ext side remains easy to parse:
#    empty separator-adjacent cells, then repeated dots/stripes.
#
# g) Color inference
#    No color remapping is introduced. Colors still directly identify source bars,
#    which keeps the human inference burden focused on the spatial rule change.
#
# New solver assumptions introduced by D11
# ----------------------------------------
# 1. The input still contains exactly one dominant separator column with >= H-1
#    cells of a single non-background color.
# 2. All bars still lie on the same side of that separator.
# 3. Bars are still processed nearest-first with no overwriting.
# 4. The generator guarantees every active row has at least one reachable
#    position under the shifted offset, so the D11 behavior is actually visible.


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


def _extension_positions(sep_col: int, w: int, step: int, bars_on_left: bool) -> range:
    """Return D11 extension positions: start one full step away from the separator."""
    if step <= 0:
        return range(0, 0)
    if bars_on_left:
        return range(sep_col + 1 + step, w, step)
    return range(sep_col - 1 - step, -1, -step)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    # Find the vertical separator column: a column whose non-bg cells are all the same color.
    # (May not span full height; some rows can be bg.)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    # Decide which side contains the bars (the other side is the painting target).
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = []
    for r in range(h):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1  # 0 if bar touches left of separator
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)  # 0 if bar touches right of separator
                bars.append((dist, r, c0, c1, length, col))

    # Process bars from closest to separator outward; do not overwrite.
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        for cc in _extension_positions(sep_col, w, length, bars_on_left):
            if int(out[r, cc]) == bg:
                out[r, cc] = col

    return out


# --------------- generator (D11 variant) ---------------

# Canvas size bounds (inclusive ranges).
H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

# Bar-side width bounds (clamped so ext_side stays in [EXT_SIDE_MIN, EXT_SIDE_MAX]).
BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

# Palette sizes.
PALETTE_MIN, PALETTE_MAX = 5, 8

# Active-row counts.
ACTIVE_MIN, ACTIVE_MAX = 2, 5

# Per-row bar structure probabilities.
P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

# Separator gap (a single terminal-row cell omitted from separator column).
P_SEP_COLUMN_GAP = 0.2

# Retry budget for generation.
MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    """Pick separator color and a disjoint bar-color palette."""
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
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is ordered near->far and the nearest bar can visibly project under D11."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0 or ext_side_len <= 1:
        raise ValueError("invalid geometry for D11 sampling")

    visible_near_max = min(capacity, ext_side_len - 1)
    if visible_near_max <= 0:
        raise ValueError("nearest bar cannot reach ext side under D11")

    two_bars = rng.random() < P_TWO_BARS and capacity >= 2
    if two_bars:
        near_max = min(capacity - 1, ext_side_len - 1)
        if near_max <= 0:
            two_bars = False

    if not two_bars:
        length = int(rng.integers(1, visible_near_max + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, near_max + 1))
    far_len = int(rng.integers(1, capacity - near_len + 1))
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
    """Paint bars on the bar side, stacking from nearest-to-separator outward."""
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
    """Simulate the D11 periodic extension onto the ext side (nearest-first, no overwrite)."""
    w = out.shape[1]
    for length, color in bars:
        for cc in _extension_positions(sep_col, w, length, bars_on_left):
            if out[row, cc] == 0:
                out[row, cc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    # bar_side_len constrained so ext_side_len = w - 1 - bar_side_len is in [EXT_SIDE_MIN, EXT_SIDE_MAX].
    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - bar_side_len
    if ext_side_len <= 1:
        return None

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len, ext_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars)

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
