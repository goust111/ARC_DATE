# Variant of 1ae2feb7
# Dimension: D13 — 颜色约束与关系 (meta)
# Change: 色条颜色互不相同且与分隔条颜色不同，颜色数5-8种；涂点颜色等于源色条颜色... → 涂点颜色由叠加顺序递变

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# -----------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# spec_variant:
#   preserved_geometry:
#     - Grid shape, separator detection, bar placement, active-row spacing, and
#       "nearest bar first / never overwrite" all stay the same as 1ae2feb7.
#     - Bars still live on exactly one side of a mostly solid vertical separator.
#     - The extension side is still empty in the input and filled only in output.
#   changed_color_rule:
#     - Original: every painted dot inherited the source bar color forever.
#     - Variant: each bar still paints at the same periodic positions, but the
#       color of successive painted dots advances through a visible global cycle.
#     - The first painted dot from a bar uses that bar's own color. Each later
#       successfully painted dot uses the next color in the cycle.
#   visible_cycle_definition:
#     - The cycle is not tied to hidden color ids.
#     - Instead, read the input top-to-bottom; within each active row, read bars
#       from the separator outward. The distinct bar colors encountered in that
#       visible order form the palette cycle.
#     - Because generator enforces unique bar colors, the cycle is directly
#       visible as an ordered list of bar colors already present in the input.
#   concrete_example:
#     - Suppose a left-side input has visible bar order [A, B, C, D, E].
#     - A length-2 bar paints ext positions 1,3,5,...
#       Original output colors there: A, A, A, ...
#       Variant output colors there:  A, B, C, D, ...
#     - If a farther bar of length 3 starts at color C, its painted cells use
#       C, D, E, A, ... but only on cells still background after nearer bars.
#   complexity_calibration:
#     - The periodic geometry is unchanged, so the task is still recognizably
#       1ae2feb7 rather than a new object-detection puzzle.
#     - To make D13 visible, generation guarantees 5-8 unique bar colors and at
#       least one row with a short single bar, so humans can see a long color
#       run instead of a trivial one-color extension.
#     - Rows may still have 1-2 bars, preserving the original overlap logic.
#   implicit_assumption_audit_for_main:
#     - _find_sep_col still assumes a mostly solid monochrome separator column.
#       The variant does not change this, and generator preserves it.
#     - left_mass/right_mass still infers the bar side from the input because
#       the extension side remains all background in the input.
#     - _iter_runs_1d still identifies monochrome bars. Variant changes only the
#       output colors, not the bar encoding, so this remains valid.
#     - Original painting loop used `out[r, cc] = col`, which silently assumed
#       the color for every dot from one bar is constant. That is false now.
#       The new loop must track per-bar paint count and map it through the cycle.
#     - Color advancement is keyed to successful paints, not blocked attempts:
#       "first/second/third dot" refers to actual new dots, which keeps the rule
#       local and learnable even when farther bars lose cells to nearer bars.
#   legality_and_cleanliness:
#     - Bars are still laid out contiguously on the bar side, never overlap, and
#       stay within the allocated side width.
#     - Active rows keep >=1 empty row between them, matching the original spec.
#     - Extension-side clutter is controlled by preserving the original no-
#       overwrite rule, so rows stay readable even with color cycling.
#   color_inference_check:
#     - The cycle order is recoverable from visible bar order, not from numeric
#       ids. Multiple rows begin from different phases, so the examples show the
#       same cycle under different starting offsets, which is enough to infer the
#       rule from a few ARC examples.
# -----------------------------------------------------------------------------


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


def _extract_bars(
    g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int
) -> tuple[list[tuple[int, int, int, int, int, int]], list[int]]:
    """Return painting bars and the visible global color cycle."""
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
    cycle: list[int] = []
    seen: set[int] = set()

    for r in range(h):
        runs = list(
            _iter_runs_1d(
                g[r, :],
                0 if bars_on_left else sep_col + 1,
                sep_col if bars_on_left else w,
                bg=bg,
            )
        )
        ordered_runs = list(reversed(runs)) if bars_on_left else runs
        for c0, c1, col in ordered_runs:
            length = c1 - c0 + 1
            dist = (sep_col - 1) - c1 if bars_on_left else c0 - (sep_col + 1)
            bars.append((dist, r, c0, c1, length, col))
            if col not in seen:
                seen.add(col)
                cycle.append(col)

    return bars, cycle


def _solve_core(grid: np.ndarray, *, use_cycle: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    del h

    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars, cycle = _extract_bars(g, sep_col, bars_on_left, bg)
    bars.sort(key=lambda t: (t[0], t[1], t[2]))

    cycle_index = {col: idx for idx, col in enumerate(cycle)}

    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)

        painted = 0
        for cc in positions:
            if int(out[r, cc]) != bg:
                continue
            if use_cycle and cycle:
                idx = (cycle_index[col] + painted) % len(cycle)
                out[r, cc] = cycle[idx]
            else:
                out[r, cc] = col
            painted += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, use_cycle=True)


# --------------- generator ---------------

H_MIN, H_MAX = 9, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 6, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400
MIN_VARIANT_DIFF = 4


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(
    rng: np.random.Generator, total_bars: int
) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    palette = list(map(int, rng.choice(bar_pool, size=total_bars, replace=False)))
    return sep_color, palette


def _sample_row_bar_counts(
    rng: np.random.Generator, total_bars: int, active_count: int
) -> list[int]:
    """Partition total bars into active_count bins, each 1 or 2, with >=1 single-bar row."""
    extra = total_bars - active_count
    if extra < 0 or extra >= active_count:
        raise ValueError("invalid bar-count partition")
    counts = [1] * active_count
    for idx in rng.choice(active_count, size=extra, replace=False):
        counts[int(idx)] = 2
    return counts


def _sample_lengths_for_row(
    rng: np.random.Generator,
    bar_count: int,
    bar_side_len: int,
    prefer_showcase: bool,
) -> tuple[list[int], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("no capacity for bars")

    if bar_count == 1:
        if prefer_showcase:
            length = int(rng.integers(1, min(3, capacity) + 1))
        else:
            small_cap = min(4, capacity)
            if rng.random() < 0.75:
                length = int(rng.integers(1, small_cap + 1))
            else:
                length = int(rng.integers(1, capacity + 1))
        return [length], sep_gap

    if capacity < 2:
        raise ValueError("not enough capacity for two bars")

    near_cap = min(3, capacity - 1)
    near_len = int(rng.integers(1, near_cap + 1))
    far_cap = capacity - near_len
    far_len = int(rng.integers(1, far_cap + 1))
    return [near_len, far_len], sep_gap


def _place_bars(
    grid: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            grid[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            grid[row, cursor : cursor + length] = color
            cursor += length


def _variant_metric(inp: np.ndarray) -> tuple[int, int]:
    """Return (cells differing from legacy, rows with >=2 ext colors)."""
    out = main(inp)
    legacy = _solve_core(inp, use_cycle=False)

    diff = int(np.sum(out != legacy))

    bg = get_bg(inp)
    sep_col, _ = _find_sep_col(inp, bg)
    if sep_col is None:
        return diff, 0

    left_mass = int(np.sum(inp[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(inp[:, sep_col + 1 :] != bg)) if sep_col + 1 < inp.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    rows_with_multi = 0
    for r in range(inp.shape[0]):
        ext = out[r, sep_col + 1 :] if bars_on_left else out[r, :sep_col]
        cols = set(map(int, ext[ext != bg]))
        if len(cols) >= 2:
            rows_with_multi += 1

    return diff, rows_with_multi


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    max_total_bars = min(PALETTE_MAX, 2 * max_active - 1)
    if max_total_bars < PALETTE_MIN:
        return None
    total_bars = int(rng.integers(PALETTE_MIN, max_total_bars + 1))

    active_lo = max(ACTIVE_MIN, total_bars // 2 + 1)
    active_hi = min(max_active, total_bars)
    if active_lo > active_hi:
        return None
    active_count = int(rng.integers(active_lo, active_hi + 1))

    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    row_bar_counts = _sample_row_bar_counts(rng, total_bars, active_count)

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng, total_bars)

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    showcase_candidates = [idx for idx, cnt in enumerate(row_bar_counts) if cnt == 1]
    showcase_idx = int(rng.choice(showcase_candidates)) if showcase_candidates else 0

    color_idx = 0
    for row_idx, row in enumerate(active_rows):
        bar_count = row_bar_counts[row_idx]
        lengths, sep_gap = _sample_lengths_for_row(
            rng,
            bar_count=bar_count,
            bar_side_len=bar_side_len,
            prefer_showcase=row_idx == showcase_idx,
        )
        row_colors = palette[color_idx : color_idx + bar_count]
        color_idx += bar_count
        bars = list(zip(lengths, row_colors))
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    diff, rows_with_multi = _variant_metric(inp)
    if diff < MIN_VARIANT_DIFF or rows_with_multi < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
