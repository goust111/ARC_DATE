# Combo variant of 1ae2feb7 (index 28)
# combo_spec: D3/1+D9/0
# Name: 水平分隔垂直色条垂直射线

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


# Design analysis
# ===============
# a) Concrete behavior and trace
#    Original task:
#    - A near-full vertical separator column splits the grid into left/right halves.
#    - Each active row on one side contains 1-2 horizontal bars.
#    - A bar of width k paints same-row dots across the other side at horizontal step k, with
#      nearer-to-separator bars processed first and protected by a no-overwrite rule.
#    Combo D3/1 + D9/0:
#    - The whole geometry is rotated: the separator is now a horizontal row, and all source objects
#      become vertical bars stacked above or below that row.
#    - Each active column contains 1-2 vertical monochrome bars ordered from the separator outward.
#    - A bar of height k paints on the opposite side of the separator within the same column at
#      rows sep+1, sep+1+k, sep+1+2k, ... (or symmetrically upward when bars are below).
#    Tiny trace:
#    - Suppose bars lie above the separator. In one active column, a 2-cell bar touches the separator
#      and a 3-cell bar sits directly above it.
#    - The original row-wise logic would be wrong because there is no meaningful horizontal width to
#      measure and no left/right target side to paint.
#    - The combo logic scans the column, extracts heights [2, 3], sorts them by separator distance
#      [0, 2], paints the lower side every 2 rows first, then fills only remaining background cells
#      every 3 rows. This makes both the rotation and the "height controls period" rule visible.
#
# b) Complexity calibration
#    - Keep the original 1-2 source segments per active lane and 2-5 active lanes, but transpose the
#      lanes from rows to columns so the task still has repeated evidence without becoming crowded.
#    - Use taller grids (15-20 rows by 8-13 cols) so vertical stepping is visually obvious and does
#      not collapse into only the cell adjacent to the separator.
#    - Require at least one sample with extension dots beyond the row directly next to the separator,
#      and usually require either a two-bar column or several active columns, so the nearest-first
#      interaction is observable across examples.
#
# c) Implicit-assumption audit of original main()
#    - `_find_sep_col` assumed the obstacle is a vertical column; after fusion the correct primitive is
#      `_find_sep_row`.
#    - Row scanning assumed sources were horizontal runs. After fusion, the only meaningful runs are
#      vertical, so the same 1-D run finder must be applied to column vectors.
#    - `left_mass/right_mass` assumed the populated side is lateral. After rotation that becomes
#      `top_mass/bottom_mass`.
#    - The original tuple `(dist, row, c0, c1, width, color)` depended on horizontal extents. After
#      fusion it must become `(dist, col, r0, r1, height, color)` because the step size is bar height.
#    - The original painting loops wrote to `out[row, cc]`. After fusion they must write to `out[rr, col]`.
#    - New combined assumption: all source bars live on exactly one side of the separator row and the
#      opposite side is empty in the input. The generator enforces this by construction.
#
# d) Placement legality
#    - The separator row position is chosen so both bar_side and ext_side heights stay inside the same
#      numeric bounds as the original problem's side widths/heights.
#    - Active columns are sampled with pairwise spacing >= 2, preventing different source stacks from
#      visually merging into wider shapes.
#    - Within one active column, the 1-2 bars are adjacent to each other, never overlap, and their
#      total height fits within the available bar-side capacity after an optional 1-cell separator gap.
#    - Any optional separator-row gap is placed only in an inactive column, so it never interferes with
#      source stacks or with identifying the separator.
#
# e) Visual observability
#    - A human sees a horizontal divider, vertical bars on one side, and vertical dotted propagation on
#      the other side. That is already enough to distinguish the combo from the original task.
#    - Heights are directly countable from the input, and repeated dot spacing on the opposite side
#      reflects those counts with no hidden arithmetic or color coding.
#
# f) Visual cleanliness
#    - Only one half-plane contains sources and the other half-plane contains the generated dots, so the
#      semantic regions remain separate.
#    - Column spacing plus no-overwrite keep the ext side readable even when a column has two bars.
#    - The palette excludes the separator color, keeping the separator identifiable despite one optional gap.
#
# g) Inferability / anti-degeneracy
#    - No color remapping is introduced: source colors are copied directly to their dots.
#    - Generator acceptance requires visible far propagation (`far_cells >= 2` in rows beyond the first
#      ext row) and at least one source bar of height >= 2, so the sample cannot collapse into a trivial
#      "paint only the adjacent row" case.
#    - Because the input uses a horizontal separator row and vertical source bars, the defining control
#      variables are the transposed geometry and bar heights, not any hidden numeric property of colors.


def _find_sep_row(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        vals = row[row != bg]
        if vals.size >= w - 1 and (vals == vals[0]).all():
            return r, int(vals[0])
    return None, None


def _iter_runs_1d(line: np.ndarray, i_lo: int, i_hi: int, bg: int):
    i = i_lo
    while i < i_hi:
        if int(line[i]) == bg:
            i += 1
            continue
        color = int(line[i])
        i0 = i
        while i < i_hi and int(line[i]) == color:
            i += 1
        yield i0, i - 1, color


# Overview:
# Detect the horizontal separator row, find all vertical source bars on the populated side,
# then project each bar's color vertically across the opposite side with period equal to its height.
#
# Input structure:
# - Exactly one near-full horizontal separator row of a single non-background color.
# - All non-separator source objects are vertical monochrome bars entirely above or entirely below it.
# - Each active column contains 1-2 bars stacked from the separator outward; the opposite side is empty.
#
# Rule:
# - Determine whether the bars are above or below the separator by comparing non-background mass.
# - In each active column, read every vertical run on the bar side, compute its height, and sort all runs
#   by distance from the separator so the nearest run is processed first.
# - For a run of height k and color c, paint color c on the opposite side of the same column at every k-th
#   cell starting from the cell adjacent to the separator, but never overwrite a non-background cell.
#
# Key insight:
# The original "bar length is the step size" rule survives unchanged after rotation; only the geometry
# changes. Width becomes height, left/right becomes up/down, and row-wise scanning becomes column-wise scanning.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    sep_row, _sep_color = _find_sep_row(g, bg)
    if sep_row is None:
        return g.copy()

    out = g.copy()

    top_mass = int(np.sum(g[:sep_row, :] != bg)) if sep_row > 0 else 0
    bottom_mass = int(np.sum(g[sep_row + 1 :, :] != bg)) if sep_row + 1 < h else 0
    bars_on_top = top_mass >= bottom_mass

    bars: list[tuple[int, int, int, int, int, int]] = []
    for c in range(w):
        col = g[:, c]
        if bars_on_top:
            for r0, r1, color in _iter_runs_1d(col, 0, sep_row, bg=bg):
                height = r1 - r0 + 1
                dist = (sep_row - 1) - r1
                bars.append((dist, c, r0, r1, height, color))
        else:
            for r0, r1, color in _iter_runs_1d(col, sep_row + 1, h, bg=bg):
                height = r1 - r0 + 1
                dist = r0 - (sep_row + 1)
                bars.append((dist, c, r0, r1, height, color))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, c, _r0, _r1, height, color in bars:
        if height <= 0:
            continue
        if bars_on_top:
            for rr in range(sep_row + 1, h, height):
                if int(out[rr, c]) == bg:
                    out[rr, c] = color
        else:
            for rr in range(sep_row - 1, -1, -height):
                if int(out[rr, c]) == bg:
                    out[rr, c] = color

    return out


# --------------- generator ---------------

# Taller canvases make the rotated vertical-period behavior plainly visible.
H_MIN, H_MAX = 15, 20
W_MIN, W_MAX = 8, 13

# Heights of the source side and target side relative to the separator row.
BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_ROW_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_cols(rng: np.random.Generator, w: int, count: int) -> list[int]:
    """Pick `count` columns in [0, w) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(w - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_col_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) with bars ordered near->far as (height, color)."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS

    if not two_bars:
        height = int(rng.integers(1, capacity + 1))
        return [(height, primary)], sep_gap

    near_height = int(rng.integers(1, capacity))
    far_height = int(rng.integers(1, capacity - near_height + 1))
    others = [c for c in palette if c != primary]
    colors = [primary, int(rng.choice(others))]
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_height, colors[0]), (far_height, colors[1])], sep_gap


def _place_bars(
    out: np.ndarray,
    col: int,
    sep_row: int,
    bars_on_top: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    """Place source bars on the populated side, stacked from the separator outward."""
    if bars_on_top:
        cursor = sep_row - sep_gap
        for height, color in bars:
            out[cursor - height : cursor, col] = color
            cursor -= height
    else:
        cursor = sep_row + 1 + sep_gap
        for height, color in bars:
            out[cursor : cursor + height, col] = color
            cursor += height


def _paint_extension(
    out: np.ndarray,
    col: int,
    sep_row: int,
    bars_on_top: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the target-side periodic painting in nearest-first order."""
    h = out.shape[0]
    for height, color in bars:
        if bars_on_top:
            positions = range(sep_row + 1, h, height)
        else:
            positions = range(sep_row - 1, -1, -height)
        for rr in positions:
            if int(out[rr, col]) == 0:
                out[rr, col] = color


def _bar_heights(bar_map: dict[int, list[tuple[int, int]]]) -> list[int]:
    return [height for bars in bar_map.values() for height, _color in bars]


def _far_extension_stats(out: np.ndarray, sep_row: int, bars_on_top: bool) -> tuple[int, int]:
    """
    Measure visible propagation beyond the cell adjacent to the separator.
    Returns (far_nonzero_cells, columns_with_far_extension).
    """
    h, w = out.shape
    far_cells = 0
    far_cols = 0
    for c in range(w):
        if bars_on_top:
            band = out[sep_row + 2 :, c] if sep_row + 2 < h else np.empty(0, dtype=int)
        else:
            band = out[: sep_row - 1, c] if sep_row - 1 > 0 else np.empty(0, dtype=int)
        if band.size > 0 and bool(np.any(band != 0)):
            far_cells += int(np.sum(band != 0))
            far_cols += 1
    return far_cells, far_cols


def _fusion_metrics(
    sep_row: int,
    bars_on_top: bool,
    bar_map: dict[int, list[tuple[int, int]]],
    out: np.ndarray,
) -> dict[str, int]:
    far_cells, far_cols = _far_extension_stats(out, sep_row, bars_on_top)
    heights = _bar_heights(bar_map)
    return {
        "active_cols": len(bar_map),
        "two_bar_cols": sum(1 for bars in bar_map.values() if len(bars) == 2),
        "max_bar_height": max(heights, default=0),
        "far_extension_cells": far_cells,
        "far_extension_cols": far_cols,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, h - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, h - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_top = bool(rng.integers(0, 2) == 0)
    sep_row = bar_side_len if bars_on_top else (h - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (w + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_cols = _choose_active_cols(rng, w, active_count)
    active_set = set(active_cols)
    inactive_cols = [c for c in range(w) if c not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[sep_row, :] = sep_color
    if inactive_cols and rng.random() < P_SEP_ROW_GAP:
        gap_col = inactive_cols[int(rng.integers(0, len(inactive_cols)))]
        out[sep_row, gap_col] = 0

    col_primary = list(map(int, rng.permutation(palette)[:active_count]))
    col_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, col in enumerate(active_cols):
        bars, sep_gap = _sample_col_bars(rng, palette, col_primary[idx], bar_side_len)
        col_bars_map[col] = bars
        _place_bars(out, col, sep_row, bars_on_top, bars, sep_gap)

    for col, bars in col_bars_map.items():
        _paint_extension(out, col, sep_row, bars_on_top, bars)

    metrics = _fusion_metrics(sep_row, bars_on_top, col_bars_map, out)
    if metrics["max_bar_height"] < 2:
        return None
    if metrics["far_extension_cols"] < 1 or metrics["far_extension_cells"] < 2:
        return None
    if metrics["two_bar_cols"] == 0 and metrics["active_cols"] < 3:
        return None

    inp = out.copy()
    if bars_on_top:
        inp[sep_row + 1 :, :] = 0
    else:
        inp[:sep_row, :] = 0

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
