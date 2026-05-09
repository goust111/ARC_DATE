# Combo variant of 1ae2feb7 (index 15)
# combo_spec: D11/0+D14/2
# Name: 延迟起始且面积递增

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for combo D11/0 + D14/2
# ----------------------------------------
# a) Concrete behavior and trace
#    Base task: each bar of length L paints single pixels at ext offsets
#    0, L, 2L, 3L, ... from the separator.
#    D11 changes the anchor sequence to L, 2L, 3L, ... so the separator-adjacent
#    region stays blank. D14 changes the paint action so the 1st anchor paints
#    1 cell, the 2nd paints 2 cells, the 3rd paints 3 cells, etc., always
#    growing away from the separator. The fusion must apply both changes to the
#    same anchor sequence.
#
#    Small left-side example with separator at column 5 and one bar of length 2:
#      input row: 112220000000
#      base:      anchors at 6,8,10,...     -> [6], [8], [10], ...
#      D11 only:  anchors at 8,10,12,...    -> [8], [10], [12], ...
#      D14 only:  anchors at 6,8,10,...     -> [6], [8,9], [10,11,12], ...
#      combo:     anchors at 8,10,12,...    -> [8], [10,11], [12,13,14], ...
#    Difference from base: there is an initial blank moat of width L, and the
#    later marks widen progressively instead of staying single-pixel dots.
#
# b) Complexity calibration
#    If rows were allowed to use arbitrary nearest-bar lengths, many would show
#    too little of the fused rule: long bars could yield only one clipped anchor,
#    and length-1 bars could weaken the "delayed start" cue. The generator
#    therefore keeps two showcase rows with a single nearest bar of length 2 and
#    requires ext_side_len >= 8. That produces repeated evidence of:
#      - two blank cells right past the separator, then
#      - a 1-cell mark, then a 2-cell mark, then a 3-cell mark.
#    Other rows retain the original variety of one- and two-bar interactions.
#
# c) Implicit-assumption audit for the original solver
#    `_find_sep_col`, `_iter_runs_1d`, and the row-wise near-first sorting are
#    still valid because the object vocabulary and priority rule do not change.
#    The assumptions broken by the combo are:
#      - `range(sep_col + 1, ..., step)` / `range(sep_col - 1, ..., -step)`
#        assumed painting starts immediately beyond the separator.
#      - `out[r, cc] = col` assumed each action touches exactly one cell.
#    The fused helper must therefore:
#      - shift the first anchor by one full bar length, and
#      - paint a segment whose length depends on the anchor index,
#        updating only background cells inside that segment.
#
# d) Placement legality
#    Bars stay on one side of the separator, optionally with a 1-cell gap, and
#    never exceed the available bar-side width. The generator uses output-first
#    construction and clips every painted segment to grid bounds, so no write can
#    go out of range. Showcase rows use a single near bar, which also prevents
#    bar-side ambiguity around the intended blank moat.
#
# e) Visual observability
#    A human solver should be able to spot the fusion from 2-3 examples because
#    the same row pattern repeats: blank cells immediately beyond the separator,
#    then widening same-color streaks at a fixed periodic spacing. This is
#    qualitatively different from both single-change variants:
#      - D11 alone keeps isolated dots.
#      - D14 alone begins painting adjacent to the separator.
#
# f) Visual cleanliness
#    Only active rows receive ext-side paint, as in the base task. The growing
#    segments stay horizontal and row-local, so even when two bars share a row,
#    the near-first no-overwrite rule keeps the layering interpretable instead
#    of mixing shapes across rows.
#
# g) Change inferability
#    No color remapping is introduced. Colors still point directly back to their
#    source bars, so the inference burden is purely geometric: infer the delayed
#    anchors and the increasing paint size.
#
# New solver assumptions introduced by the combo
# ----------------------------------------------
# 1. The separator is still the unique dominant monochrome column with >= H-1
#    non-background cells.
# 2. All source bars still lie on exactly one side of the separator.
# 3. Bars are still processed nearest-first and cannot overwrite earlier paint.
# 4. The generator guarantees showcase rows where the fused rule differs from
#    the base task, D11 alone, and D14 alone, so the combo is visible in data.


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


def _paint_periodic(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    color: int,
    bg: int,
    *,
    start_delay: bool,
    growing: bool,
) -> None:
    """Paint periodic anchors with optional delayed start and growing segment size."""
    if step <= 0:
        return

    w = out.shape[1]
    anchor_shift = step if start_delay else 0
    if bars_on_left:
        anchor = sep_col + 1 + anchor_shift
        seg_len = 1
        while anchor < w:
            paint_len = seg_len if growing else 1
            end = min(w, anchor + paint_len)
            seg = out[row, anchor:end]
            seg[seg == bg] = color
            anchor += step
            seg_len += 1
    else:
        anchor = sep_col - 1 - anchor_shift
        seg_len = 1
        while anchor >= 0:
            paint_len = seg_len if growing else 1
            start = max(0, anchor - paint_len + 1)
            seg = out[row, start : anchor + 1]
            seg[seg == bg] = color
            anchor -= step
            seg_len += 1


def _simulate(
    grid: np.ndarray,
    *,
    start_delay: bool,
    growing: bool,
) -> np.ndarray:
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
                bars.append((dist, r, c0, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, length, col in bars:
        _paint_periodic(
            out,
            r,
            sep_col,
            bars_on_left,
            length,
            col,
            bg,
            start_delay=start_delay,
            growing=growing,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the fused periodic projection rule for combo D11/0 + D14/2.

    Input structure:
    A grid contains one dominant vertical separator column, 2-5 active rows with
    one or two horizontal monochrome bars on exactly one side of that separator,
    and an empty extension side on the other side.

    Rule:
    Process bars row-wise from nearest to the separator outward, as in the base
    task. For a bar of length L, paint on the extension side starting at distance
    L from the separator, then every additional L cells. The 1st anchor paints a
    1-cell segment, the 2nd anchor paints a 2-cell segment, the 3rd paints a
    3-cell segment, and so on, always growing away from the separator and never
    overwriting non-background cells.

    Key insight:
    The fusion is not two independent passes. Each bar uses one shared periodic
    anchor sequence that is both delayed by its own length and indexed to decide
    how wide each painted segment should be.
    """
    return _simulate(grid, start_delay=True, growing=True)


H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 8, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400
SHOWCASE_COUNT = 2
SHOWCASE_LENGTH = 2


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
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
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Sample non-showcase rows while keeping the nearest bar visible under D11."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("invalid row capacity")

    visible_near_max = min(capacity, ext_side_len - 1)
    if visible_near_max <= 0:
        raise ValueError("nearest bar cannot reach extension side")

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


def _paint_extension_combo(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    for length, color in bars:
        _paint_periodic(
            out,
            row,
            sep_col,
            bars_on_left,
            length,
            color,
            bg=0,
            start_delay=True,
            growing=True,
        )


def _ext_mask(shape: tuple[int, int], sep_col: int, bars_on_left: bool) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if bars_on_left:
        mask[:, sep_col + 1 :] = True
    else:
        mask[:, :sep_col] = True
    return mask


def _variant_metrics(
    inp: np.ndarray,
    combo_out: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    showcase_rows: list[int],
) -> dict[str, int]:
    ext_mask = _ext_mask(inp.shape, sep_col, bars_on_left)
    base_out = _simulate(inp, start_delay=False, growing=False)
    d11_out = _simulate(inp, start_delay=True, growing=False)
    d14_out = _simulate(inp, start_delay=False, growing=True)

    metrics = {
        "diff_vs_base": int(np.sum((combo_out != base_out) & ext_mask)),
        "diff_vs_d11": int(np.sum((combo_out != d11_out) & ext_mask)),
        "diff_vs_d14": int(np.sum((combo_out != d14_out) & ext_mask)),
        "showcase_blank_moat_rows": 0,
        "showcase_rows_diff_all": 0,
    }

    for row in showcase_rows:
        combo_row = combo_out[row]
        if bars_on_left:
            moat = combo_row[sep_col + 1 : sep_col + 1 + SHOWCASE_LENGTH]
        else:
            moat = combo_row[sep_col - SHOWCASE_LENGTH : sep_col]
        if moat.size == SHOWCASE_LENGTH and np.all(moat == 0):
            metrics["showcase_blank_moat_rows"] += 1

        row_diff = (
            np.any(combo_out[row] != base_out[row])
            and np.any(combo_out[row] != d11_out[row])
            and np.any(combo_out[row] != d14_out[row])
        )
        if row_diff:
            metrics["showcase_rows_diff_all"] += 1

    return metrics


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]] | None:
    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - bar_side_len
    if ext_side_len < EXT_SIDE_MIN:
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
    showcase_rows = active_rows[: min(SHOWCASE_COUNT, active_count)]

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        if row in showcase_rows:
            bars = [(SHOWCASE_LENGTH, row_primary[idx])]
            sep_gap = 0
        else:
            bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len, ext_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension_combo(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    metrics = _variant_metrics(inp, out, sep_col, bars_on_left, showcase_rows)
    if metrics["showcase_blank_moat_rows"] < len(showcase_rows):
        return None
    if metrics["showcase_rows_diff_all"] < len(showcase_rows):
        return None
    if metrics["diff_vs_base"] < len(showcase_rows) * 4:
        return None
    if metrics["diff_vs_d11"] < len(showcase_rows) * 3:
        return None
    if metrics["diff_vs_d14"] < len(showcase_rows) * 2:
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, metrics


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        attempt = _gen_attempt(rng)
        if attempt is not None:
            grid, _metrics = attempt
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
