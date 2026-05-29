# Variant of 1ae2feb7
# Dimension: D4 — 终止条件/目标 (dst)
# Change: null（网格边缘）——射线延伸直到到达网格的对侧边缘，没有其他终止条件... → 到达目标线停止

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

"""
Design analysis for the D4 variant
==================================

a) Concretization + behavior trace
----------------------------------
- Original visual structure:
  one vertical separator splits the grid into a bar side and an empty extension side.
  Each active row contains 1-2 horizontal monochrome bars on the bar side.
  In the output, each bar paints periodic dots/segments on the extension side until the grid edge.
- Variant visual structure:
  keep the original separator and bars, but add one more full-height vertical line on the extension side.
  This second line is the visible target line. Periodic painting stops before that line; the cells beyond it stay empty.
- Concrete example:
  suppose bars are on the left, separator at column 5, target line at column 11.
  A length-4 bar paints columns 6, 10 in the variant, then stops because the next hit would be column 14, which is past the target line.
  In the original task the same row would continue to column 14 as well.
- Affected pipeline stage:
  the only semantic change is the destination/termination condition of the extension process.
  Input construction must therefore add a target line, and solver painting must replace "until border" with "until target line".

b) Complexity calibration
-------------------------
- The new control variable is a single visible geometric distance: separator-to-target-line gap.
- This stays within ARC-scale complexity: one extra object, one extra distance, no hidden lookup table.
- To make the variant observable, generator always leaves at least one empty column beyond the target line,
  and rejects samples where stopping early would produce the same output as the old edge-based rule.
- No extra symbolic encoding object is added; the distance itself is directly visible in the grid.

c) Implicit-assumption audit of the original solver
---------------------------------------------------
- Original `_find_sep_col` assumes there is only one separator-like vertical column.
  This is no longer safe because the target line is intentionally another separator-like column.
- Original bar-side detection compares total non-bg mass on the two sides of the separator.
  This breaks in the variant because the target line contributes a full column of mass and can outweigh the bars.
- Original paint loop uses `range(..., w, step)` / `range(..., -1, -step)`, which hard-codes the border as the stop.
  In the variant that would leak paint beyond the target line.
- The revised solver therefore:
  1. finds all separator-like vertical columns,
  2. picks as the primary separator the unique such column that has non-bg content on both sides,
  3. uses the nearest other separator-like column as the target line,
  4. infers the bar side as the side opposite that target line,
  5. paints only in the open interval between separator and target.

d) Placement legality
---------------------
- Width is partitioned into:
  `bar_side_len + 1(main separator) + corridor_len + 1(target line) + tail_margin`.
- Generator samples these widths so all pieces are non-overlapping and in-bounds.
- Bars are placed only on the bar side exactly as in the original task.
- The corridor between separator and target line starts empty in the input and receives the periodic fill in the output.
- The tail beyond the target line remains empty in both input and output, which keeps the target line visually isolated.

e) Visual observability
-----------------------
- Humans can see the second vertical line in the input and can also see that output paint halts at that line.
- Because the generator enforces a non-empty tail beyond the target line, the stop effect is visible instead of collapsing back to the border case.
- The target line position varies across samples, so the stopping distance is not a fixed constant.

f) Visual cleanliness
---------------------
- Only one extra object is introduced: a clean vertical target line.
- No dots, markers, or legends are added inside the corridor, so the extension pattern stays easy to read.
- The target line color is distinct from separator and bar colors to avoid merging the two roles.

g) Color-rule audit
-------------------
- There is no color remapping.
- Colors are purely object labels: background, separator, target line, and bar colors.
- Since the new rule is geometric rather than chromatic, humans do not need to infer any hidden color transformation.

9) New assumptions introduced by the revised solver
---------------------------------------------------
- There is exactly one target line on the extension side and it is separator-like: a mostly full monochrome vertical column.
- The primary separator is the only separator-like column with non-bg structure on both sides.
- The nearest other separator-like column to that primary separator is the target line.
- Bars occupy only the side opposite the target line.
- Generator guarantees these assumptions by constructing exactly two vertical lines, keeping the tail empty,
  and never placing other non-bg objects on the target side.
"""


def _find_line_candidates(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    cands: list[tuple[int, int]] = []
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            cands.append((c, int(vals[0])))
    return cands


def _pick_separator_and_target(
    g: np.ndarray, bg: int, cands: list[tuple[int, int]]
) -> tuple[int, int | None]:
    scored: list[tuple[int, int, int]] = []
    for c, _color in cands:
        left_has = int(np.any(g[:, :c] != bg)) if c > 0 else 0
        right_has = int(np.any(g[:, c + 1 :] != bg)) if c + 1 < g.shape[1] else 0
        if left_has and right_has:
            balance = min(
                int(np.sum(g[:, :c] != bg)) if c > 0 else 0,
                int(np.sum(g[:, c + 1 :] != bg)) if c + 1 < g.shape[1] else 0,
            )
            scored.append((balance, c, _color))

    if scored:
        scored.sort(reverse=True)
        sep_col = int(scored[0][1])
    else:
        sep_col = int(cands[0][0])

    other_cols = [int(c) for c, _color in cands if int(c) != sep_col]
    target_col = min(other_cols, key=lambda c: abs(c - sep_col)) if other_cols else None
    return sep_col, target_col


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


def _paint_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    color: int,
    target_col: int | None,
) -> None:
    if step <= 0:
        return
    w = out.shape[1]
    if bars_on_left:
        stop = w if target_col is None else target_col
        positions = range(sep_col + 1, stop, step)
    else:
        stop = -1 if target_col is None else target_col
        positions = range(sep_col - 1, stop, -step)
    for cc in positions:
        if int(out[row, cc]) == 0:
            out[row, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _ = h, w

    bg = _bg_color(g)
    cands = _find_line_candidates(g, bg)
    if not cands:
        return g.copy()

    sep_col, target_col = _pick_separator_and_target(g, bg, cands)
    out = g.copy()

    if target_col is None:
        left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
        right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
        bars_on_left = left_mass >= right_mass
    else:
        bars_on_left = bool(target_col > sep_col)

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
        _paint_row(out, r, sep_col, bars_on_left, length, col, target_col)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 10
CORRIDOR_MIN, CORRIDOR_MAX = 3, 9
TAIL_MIN, TAIL_MAX = 1, 4

PALETTE_MIN, PALETTE_MAX = 4, 7

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_colors(
    rng: np.random.Generator, min_palette_size: int
) -> tuple[int, int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    target_pool = [c for c in range(1, 10) if c != sep_color]
    target_color = int(rng.choice(target_pool))
    bar_pool = [c for c in range(1, 10) if c not in (sep_color, target_color)]
    lo = max(PALETTE_MIN, min_palette_size)
    hi = min(PALETTE_MAX, len(bar_pool))
    if lo > hi:
        raise ValueError("palette request exceeds available colors")
    size = int(rng.integers(lo, hi + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, target_color, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        return [(1, primary)], 0

    two_bars = bool(rng.random() < P_TWO_BARS and capacity >= 2)
    if not two_bars:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, capacity))
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


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    target_col: int | None,
) -> None:
    for length, color in bars:
        _paint_row(out, row, sep_col, bars_on_left, length, color, target_col)


def _sample_layout(
    rng: np.random.Generator, w: int
) -> tuple[int, int, int] | None:
    total = w - 2
    bar_lo = max(BAR_SIDE_MIN, total - (CORRIDOR_MAX + TAIL_MAX))
    bar_hi = min(BAR_SIDE_MAX, total - (CORRIDOR_MIN + TAIL_MIN))
    if bar_lo > bar_hi:
        return None
    bar_side_len = int(rng.integers(bar_lo, bar_hi + 1))
    rem = total - bar_side_len
    cor_lo = max(CORRIDOR_MIN, rem - TAIL_MAX)
    cor_hi = min(CORRIDOR_MAX, rem - TAIL_MIN)
    if cor_lo > cor_hi:
        return None
    corridor_len = int(rng.integers(cor_lo, cor_hi + 1))
    tail_margin = int(rem - corridor_len)
    return bar_side_len, corridor_len, tail_margin


def _variant_delta(
    base: np.ndarray,
    sep_col: int,
    target_col: int,
    bars_on_left: bool,
    row_bars_map: dict[int, list[tuple[int, int]]],
) -> int:
    limited = base.copy()
    unbounded = base.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(limited, row, sep_col, bars_on_left, bars, target_col)
        _paint_extension(unbounded, row, sep_col, bars_on_left, bars, None)
    return int(np.count_nonzero(limited != unbounded))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    layout = _sample_layout(rng, w)
    if layout is None:
        return None
    bar_side_len, corridor_len, tail_margin = layout

    bars_on_left = bool(rng.integers(0, 2) == 0)
    if bars_on_left:
        sep_col = bar_side_len
        target_col = sep_col + 1 + corridor_len
        if w - 1 - target_col != tail_margin:
            return None
    else:
        sep_col = w - 1 - bar_side_len
        target_col = sep_col - 1 - corridor_len
        if target_col != tail_margin:
            return None

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    sep_color, target_color, palette = _sample_colors(rng, active_count)

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    out[:, target_col] = target_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    base = out.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars, target_col)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 : target_col] = 0
        inp[:, target_col + 1 :] = 0
    else:
        inp[:, target_col + 1 : sep_col] = 0
        inp[:, :target_col] = 0

    if _variant_delta(base, sep_col, target_col, bars_on_left, row_bars_map) <= 0:
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if target_color not in colors_present(inp, exclude=[0]):
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
