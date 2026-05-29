# Variant of 1ae2feb7
# Dimension: D2 — 轨迹涂色模式 (trajectory)
# Change: periodic_pattern——以间隔=色条像素数在ext_side上涂单色点（位置0, length, 2*len...）→ 颜色循环轨迹

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D2 variant
#
# variant_spec:
#   preserved_structure:
#     - Grid size, separator geometry, active-row spacing, and bar placement stay
#       the same as the source task.
#     - The separator is still a mostly-full vertical monochrome column.
#     - All bars still live on one side of the separator; ext_side is empty in input.
#     - A row may still have 1 or 2 adjacent monochrome bars.
#   changed_mechanism:
#     - Single-bar rows behave exactly like the original task.
#     - Multi-bar rows keep the same candidate dot positions as the original:
#       each bar of length L contributes periodic positions 0, L, 2L, ... on ext_side.
#     - The old rule colored these positions bar-by-bar, nearest-first, with no overwrite.
#     - The new rule first takes the UNION of those periodic positions on that row,
#       ordered from the separator outward, then colors them with a repeating cycle
#       derived from the row's bar colors ordered nearest-to-farthest from the separator.
#   concrete_trace:
#     - Example row on the left side: `333442` where separator color is `2`.
#       Bars are near=`44` (length 2, color 4) and far=`333` (length 3, color 3).
#     - Original output dots on ext_side:
#       near bar => positions 0,2,4,6,8,... color 4
#       far bar  => positions 0,3,6,9,... color 3, but blocked where color 4 already exists
#       result   => positions [0,2,4,6,8]=4 and [3,9]=3.
#     - Variant output dots on ext_side:
#       candidate positions = union{0,2,3,4,6,8,9,...}
#       color cycle = [4,3] because 4 is nearer to the separator than 3
#       assign outward => 0->4, 2->3, 3->4, 4->3, 6->4, 8->3, 9->4, ...
#     - Difference:
#       geometry of dotted positions is unchanged, but ownership-by-bar disappears.
#       Color now depends on cycle index along the trajectory, not on which bar hit first.
#   complexity_calibration:
#     - The cycle order is purely spatial and visible in the input, avoiding hidden
#       color-id sorting.
#     - Keeping the original periodic geometry prevents the task from becoming a
#       completely different puzzle; only the D2 trajectory/color progression changes.
#     - The generator biases toward 2-bar rows and rejects samples where every 2-bar
#       row degenerates to the original output, so the variant remains observable.
#   implicit_assumption_review:
#     - `_find_sep_col` still assumes a mostly full monochrome separator column.
#       Generator preserves that.
#     - Bar detection still assumes all meaningful horizontal runs are on one side
#       of the separator. Generator preserves that.
#     - Original `bars.sort(...); for each bar paint if bg` relied on "a bar owns its
#       own periodic dots" and on overwrite blocking to encode priority. That premise
#       is false in this variant, so solver must group bars by row before painting.
#     - New solver assumes cycle order is nearest-to-farthest within a row. Generator
#       places multi-bar rows contiguously so this order is visually explicit.
#     - New solver assumes ext_side in the input is background; generator clears it.
#   placement_legality_and_visual_cleanliness:
#     - Bars remain within the bar side and are adjacent when there are two of them.
#     - Active rows still have at least one empty row between them, so trajectories do
#       not visually merge across rows.
#     - ext_side is untouched except for the dotted trajectory, keeping outputs legible.
#   color_inference:
#     - The cycle source is the visible bar list in that row.
#     - The cycle order is spatial (nearest to separator outward), not numeric.
#     - Palette size stays moderate and multi-bar rows are repeated across samples,
#       giving enough evidence for humans to infer the rule from 2-4 examples.
# ---------------------------------------------------------------------------


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


def _ext_positions(sep_col: int, width: int, bars_on_left: bool, step: int) -> list[int]:
    if step <= 0:
        return []
    if bars_on_left:
        return list(range(sep_col + 1, width, step))
    return list(range(sep_col - 1, -1, -step))


def _collect_row_bars(
    g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int
) -> dict[int, list[tuple[int, int, int, int, int, int]]]:
    h, w = g.shape
    row_bars: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for r in range(h):
        found: list[tuple[int, int, int, int, int, int]] = []
        if bars_on_left:
            runs = _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                found.append((dist, r, c0, c1, length, col))
        else:
            runs = _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                found.append((dist, r, c0, c1, length, col))
        if found:
            row_bars[r] = sorted(found, key=lambda t: (t[0], t[2]))
    return row_bars


def _paint_original_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> None:
    for _, _, _, _, length, col in bars:
        for cc in _ext_positions(sep_col, out.shape[1], bars_on_left, length):
            if int(out[row, cc]) == bg:
                out[row, cc] = col


def _paint_cycle_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> None:
    if len(bars) <= 1:
        _paint_original_row(out, row, sep_col, bars_on_left, bars, bg)
        return

    cycle = [int(col) for _, _, _, _, _, col in bars]
    positions = sorted(
        {
            cc
            for _, _, _, _, length, _ in bars
            for cc in _ext_positions(sep_col, out.shape[1], bars_on_left, length)
        },
        reverse=not bars_on_left,
    )
    for idx, cc in enumerate(positions):
        if int(out[row, cc]) == bg:
            out[row, cc] = cycle[idx % len(cycle)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    del h, w

    bg = get_bg(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    for row, bars in row_bars.items():
        _paint_cycle_row(out, row, sep_col, bars_on_left, bars, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.72
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_MATCH_LENGTHS = 0.35
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _row_ext_signature(
    width: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> tuple[int, ...]:
    positions = sorted(
        {
            cc
            for length, _ in bars
            for cc in _ext_positions(sep_col, width, bars_on_left, length)
        },
        reverse=not bars_on_left,
    )
    return tuple(map(int, positions))


def _paint_extension_original(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    for length, color in bars:
        for cc in _ext_positions(sep_col, out.shape[1], bars_on_left, length):
            if out[row, cc] == 0:
                out[row, cc] = color


def _paint_extension_cycle(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    if len(bars) <= 1:
        _paint_extension_original(out, row, sep_col, bars_on_left, bars)
        return

    positions = _row_ext_signature(out.shape[1], sep_col, bars_on_left, bars)
    cycle = [int(color) for _, color in bars]
    for idx, cc in enumerate(positions):
        if out[row, cc] == 0:
            out[row, cc] = cycle[idx % len(cycle)]


def _row_variant_signal(
    width: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> bool:
    if len(bars) <= 1:
        return False
    positions = _row_ext_signature(width, sep_col, bars_on_left, bars)
    if len(positions) < 4:
        return False

    cycle = [int(color) for _, color in bars]
    cycle_vals = [cycle[idx % len(cycle)] for idx in range(len(positions))]

    original = np.zeros(width, dtype=int)
    tmp = original.copy()
    for length, color in bars:
        for cc in _ext_positions(sep_col, width, bars_on_left, length):
            if tmp[cc] == 0:
                tmp[cc] = color
    original_vals = [int(tmp[cc]) for cc in positions]

    return cycle_vals != original_vals


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    sep_col: int,
    bars_on_left: bool,
) -> tuple[list[tuple[int, int]], int]:
    for _ in range(80):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        if capacity <= 0:
            continue

        two_bars = rng.random() < P_TWO_BARS and capacity >= 2
        if not two_bars:
            length = int(rng.integers(1, capacity + 1))
            return [(length, primary)], sep_gap

        if rng.random() < P_MATCH_LENGTHS:
            max_len = capacity // 2
            if max_len <= 0:
                continue
            near_len = far_len = int(rng.integers(1, max_len + 1))
        else:
            near_len = int(rng.integers(1, capacity))
            far_len = int(rng.integers(1, capacity - near_len + 1))

        others = [c for c in palette if c != primary]
        colors = [primary, int(rng.choice(others))]
        if rng.random() < P_SWAP_COLORS:
            colors.reverse()
        bars = [(near_len, colors[0]), (far_len, colors[1])]

        positions = _row_ext_signature(ext_side_len + 1 + bar_side_len, sep_col, bars_on_left, bars)
        if len(positions) < 4:
            continue
        if not _row_variant_signal(ext_side_len + 1 + bar_side_len, sep_col, bars_on_left, bars):
            continue
        return bars, sep_gap

    sep_gap = 0
    capacity = max(2, bar_side_len)
    near_len = far_len = 1 if capacity >= 2 else capacity
    others = [c for c in palette if c != primary]
    return [(near_len, primary), (far_len, int(others[0]))], sep_gap


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
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    cycle_rows = 0

    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng, palette, row_primary[idx], bar_side_len, ext_side_len, sep_col, bars_on_left
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)
        if _row_variant_signal(w, sep_col, bars_on_left, bars):
            cycle_rows += 1

    min_cycle_rows = 1 if active_count <= 3 else 2
    if cycle_rows < min_cycle_rows:
        return None

    for row, bars in row_bars_map.items():
        _paint_extension_cycle(out, row, sep_col, bars_on_left, bars)

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
