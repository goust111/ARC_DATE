# Combo variant of 1ae2feb7 (index 9)
# combo_spec: D4/1+D6/0
# Name: 目标线截断后裁剪

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

"""
Design analysis for the combo variant
=====================================

a) Concretization + behavior trace
----------------------------------
- Original task:
  one separator column splits the grid into a bar side and an empty extension side.
  Each active row has 1-2 horizontal monochrome bars. The output paints periodic
  dots on the extension side all the way to the far border, and keeps the full grid.
- Combo task:
  add a second full-height vertical line on the extension side. This is the target
  line. Periodic painting stops before that target line, and the final output is
  cropped down to only the open corridor between the separator and the target line.
- Concrete trace, bars on the left:
  row input:   33302200000700
               ^sep     ^target
  original full output row would be:
               33302230300730
  combo full solved row becomes:
               33302230300700
  combo returned row is the crop between the two lines:
                      3030
- So two pipeline stages change together:
  1. destination: paint only until the target line;
  2. projection: return only the separator-to-target corridor, excluding both lines.

b) Complexity calibration
-------------------------
- The visual control variables are still small and explicit:
  one target line location, one corridor width, and the original bar lengths.
- To keep the crop observable, the corridor is forced to be at least 4 columns wide.
- To keep the truncation observable, the generator also forces a non-empty tail
  beyond the target line and rejects samples where bounded painting matches the
  old edge-to-border behavior.
- No hidden symbolic encoding or color remapping is introduced.

c) Implicit-assumption audit of the original solver
---------------------------------------------------
- Original `_find_sep_col` assumed there was only one separator-like column.
  That fails here because the target line is intentionally a second such column.
- Original `bars_on_left = left_mass >= right_mass` can fail because the target line
  adds a full non-background column on the extension side.
- Original paint loops stop at the border, which would leak paint past the target line.
- Original `return out` assumes the output keeps bars and separator, which is false
  after the crop change.
- Revised solver strategy:
  1. detect all separator-like vertical columns;
  2. choose as separator the unique candidate with non-background structure on both sides;
  3. choose the nearest other candidate as the target line;
  4. infer bar side as the side opposite that target line;
  5. paint only in the open interval up to the target line;
  6. crop the final answer to that open interval.

d) Placement legality
---------------------
- The width is partitioned as:
  `bar_side_len + 1(separator) + corridor_len + 1(target) + tail_len`.
- Generator samples these widths so every object is in bounds and pairwise disjoint.
- Bars are placed only on the bar side, adjacent outward from the separator with
  an optional one-cell gap exactly as in the original family.
- The corridor is empty in the input, the target line is visible in the input, and
  the tail beyond the target line stays empty in both input and output.

e) Visual observability
-----------------------
- In examples, the human can directly see two vertical lines and can see that the
  output width equals the space between them rather than the full blank side.
- The tail beyond the target line makes the truncation effect visible; otherwise the
  target line could collapse into a border-equivalent case.
- The crop is also visible because the returned grid omits both the bar side and the
  target line itself, leaving only the painted corridor.

f) Visual cleanliness
---------------------
- Only one extra object is introduced beyond the original task: a clean target line.
- The corridor contains only the periodic paint, so the returned crop is visually tidy.
- Target line color is distinct from separator and bar colors, which keeps roles clear.

g) Change inferability
----------------------
- There is no color remapping rule to infer. Colors remain object labels.
- The only new reasoning is geometric: "paint stops at the second line" and
  "only the space between the two lines is returned".
- Those rules are recoverable from a few examples because both the stopping object
  and the output crop boundary are visible and vary across samples.

h) Fusion-specific anti-degeneracy checks
-----------------------------------------
- The generator measures how many cells differ between bounded painting and the old
  border-reaching painting, and rejects zero-delta samples.
- It also requires at least one painted cell in the returned corridor per active-row
  pattern overall, so the crop is not trivially all background.

i) New solver assumptions introduced here
-----------------------------------------
- Exactly two separator-like columns exist: the main separator and one target line.
- The main separator is the only such column with non-background content on both sides.
- Bars occupy only the side opposite the target line.
- The target region returned by the task is the open interval between the two lines.
- The generator guarantees these assumptions by construction.
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
    scored: list[tuple[int, int]] = []
    for c, _color in cands:
        left_has = bool(c > 0 and np.any(g[:, :c] != bg))
        right_has = bool(c + 1 < g.shape[1] and np.any(g[:, c + 1 :] != bg))
        if left_has and right_has:
            balance = min(
                int(np.sum(g[:, :c] != bg)) if c > 0 else 0,
                int(np.sum(g[:, c + 1 :] != bg)) if c + 1 < g.shape[1] else 0,
            )
            scored.append((balance, c))

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
    bg: int,
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
        if int(out[row, cc]) == bg:
            out[row, cc] = color


def _crop_target_region(
    full: np.ndarray,
    sep_col: int,
    target_col: int | None,
    bars_on_left: bool,
) -> np.ndarray:
    if target_col is None:
        if bars_on_left:
            return full[:, sep_col + 1 :].copy()
        return full[:, :sep_col].copy()
    if target_col > sep_col:
        return full[:, sep_col + 1 : target_col].copy()
    return full[:, target_col + 1 : sep_col].copy()


def _solve_full(
    grid: np.ndarray,
) -> tuple[np.ndarray, int | None, int | None, bool | None, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    cands = _find_line_candidates(g, bg)
    if not cands:
        return g.copy(), None, None, None, bg

    sep_col, target_col = _pick_separator_and_target(g, bg, cands)
    out = g.copy()

    if target_col is None:
        left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
        right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
        bars_on_left = left_mass >= right_mass
    else:
        bars_on_left = bool(target_col > sep_col)

    bars: list[tuple[int, int, int, int]] = []
    for r in range(g.shape[0]):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, g.shape[1], bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, length, col in bars:
        _paint_row(out, r, sep_col, bars_on_left, length, col, target_col, bg)

    return out, sep_col, target_col, bars_on_left, bg


# Overview:
#   Solve the original periodic extension on the empty side, but stop at a visible
#   target line and then return only the corridor between the separator and target.
# Input structure:
#   The input contains one bar side with 1-2 horizontal bars per active row, one
#   main separator column, an empty corridor on the opposite side, and a second
#   full-height target line farther out on that corridor side.
# Rule:
#   Parse bars from nearest to farthest from the separator; each bar paints periodic
#   cells with period equal to its length, without overwriting existing non-background
#   cells, and painting stops before the target line. After solving, crop away the
#   bar side, the separator column, the target line, and any tail beyond the target.
# Key insight:
#   The second separator-like column is not another bar source. It is the geometric
#   stop marker, and the true output is the open interval between the two vertical lines.
def main(grid: np.ndarray) -> np.ndarray:
    full_out, sep_col, target_col, bars_on_left, _bg = _solve_full(grid)
    if sep_col is None or bars_on_left is None:
        return full_out
    return _crop_target_region(full_out, sep_col, target_col, bars_on_left)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 10
CORRIDOR_MIN, CORRIDOR_MAX = 4, 9
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
        _paint_row(out, row, sep_col, bars_on_left, length, color, target_col, 0)


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
    corridor_lo = max(CORRIDOR_MIN, rem - TAIL_MAX)
    corridor_hi = min(CORRIDOR_MAX, rem - TAIL_MIN)
    if corridor_lo > corridor_hi:
        return None
    corridor_len = int(rng.integers(corridor_lo, corridor_hi + 1))
    tail_len = int(rem - corridor_len)
    return bar_side_len, corridor_len, tail_len


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
    bar_side_len, corridor_len, tail_len = layout

    bars_on_left = bool(rng.integers(0, 2) == 0)
    if bars_on_left:
        sep_col = bar_side_len
        target_col = sep_col + 1 + corridor_len
        if w - 1 - target_col != tail_len:
            return None
    else:
        sep_col = w - 1 - bar_side_len
        target_col = sep_col - 1 - corridor_len
        if target_col != tail_len:
            return None

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_row_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_row_set]

    sep_color, target_color, palette = _sample_colors(rng, active_count)

    full_out = np.zeros((h, w), dtype=int)
    full_out[:, sep_col] = sep_color
    full_out[:, target_col] = target_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        full_out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(full_out, row, sep_col, bars_on_left, bars, sep_gap)

    base = full_out.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(full_out, row, sep_col, bars_on_left, bars, target_col)

    expected = _crop_target_region(full_out, sep_col, target_col, bars_on_left)
    if expected.shape != (h, corridor_len):
        return None
    if np.count_nonzero(expected) < active_count:
        return None
    if _variant_delta(base, sep_col, target_col, bars_on_left, row_bars_map) <= 0:
        return None

    inp = full_out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 : target_col] = 0
        inp[:, target_col + 1 :] = 0
    else:
        inp[:, target_col + 1 : sep_col] = 0
        inp[:, :target_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if target_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
