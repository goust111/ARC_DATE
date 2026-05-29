# Variant of 1ae2feb7
# Dimension: D17 — 活跃行的选择规则 (meta)
# Change: 活跃行（包含色条的行）由生成策略随机选择，2-5行，行间至少1空行间隔... → 活跃行成对出现

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis
# a) Concretization / behavior trace:
#    The original task is row-local: each active row owns 1-2 bars, and that same row's bars
#    generate periodic dots on the extension side. For D17 we make active rows appear only as
#    adjacent pairs (r, r+1), separated from the next pair by at least one empty row. Inside a
#    pair, both rows carry the same two colors but in reversed near/far order, and their bar
#    lengths are also reversed. Example on the left of the separator:
#      row r   : [far blue len 3][near red len 2] | sep
#      row r+1 : [far red  len 2][near blue len 3] | sep
#    Original logic would paint row r with step-2 red first, then step-3 blue; the variant instead
#    uses the partner row as the donor template, so row r is painted with step-3 blue first, then
#    step-2 red. The solver therefore must first detect adjacent active-row pairs, then swap donor
#    bars within each pair before doing the usual nearest-first no-overwrite periodic painting.
# b) Complexity calibration:
#    If paired rows were merely adjacent while keeping the original per-row painting rule, the task
#    would degenerate to the original. To make the pair relation visible and necessary, every active
#    row uses exactly two bars and each pair is a strict reversal of colors and lengths. Lengths are
#    forced to differ, and generation prefers extension sides of length >= 5 so the swapped periods
#    are visually observable.
# c) Hidden-assumption audit of the original main():
#    - Separator finding remains valid: the separator is still one nearly full monochrome column.
#    - Bar-side detection by counting non-bg mass remains valid because all bars still live on one side.
#    - The old `bars` list flattened all rows together and implicitly assumed "row r is painted from
#      bars found on row r". That assumption is invalid now. The new solver still extracts bars per
#      row in nearest->far order, but it paints each row using its paired row's bar list instead.
#    - Processing order within a row (nearest first, no overwrite) is still required and unchanged.
# d) Placement legality:
#    Each pair occupies rows (s, s+1). Pair starts differ by at least 3, so pairs never overlap and
#    always leave >=1 empty row between them. Bars remain contiguous on one side of the separator and
#    fit within `bar_side_len - sep_gap`. The separator gap, when used, is only placed on an inactive
#    row so the solver still sees a single mostly full separator column.
# e) Visual observability:
#    The paired rows are directly adjacent and use the same two colors in reversed order, making the
#    pairing easy to spot. Because outputs swap the donor template across the pair, a human can see
#    that row r's extension matches row r+1's bar ordering/periods rather than its own.
# f) Visual cleanliness:
#    Only three visual layers coexist: separator column, compact bar-side runs, and sparse periodic
#    dots on the opposite side. Pairs are separated by empty rows, so different pair interactions do
#    not overlap or create ambiguous clutter.
# g) Color-inference audit:
#    No color-id arithmetic or hidden ordering is used. Colors are only copied from visible bars.
#    Each pair uses two visible colors with a clear reversed arrangement, so the color transfer rule
#    is inferable from structure rather than from arbitrary ids.
# New solver assumptions after the change:
#    - Active rows are intended to come in adjacent pairs; generator guarantees this.
#    - Each active row has a non-empty bar list, and paired rows are the only adjacent active rows.
#      The solver still falls back to self-donation for an unpaired active row instead of crashing.
#    - Donor bars are reused exactly as extracted, preserving their visible near->far order.


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


def _extract_row_bars(
    g: np.ndarray, row: int, sep_col: int, bars_on_left: bool, bg: int
) -> list[tuple[int, int]]:
    """Return row bars as (length, color) ordered near->far from the separator."""
    if bars_on_left:
        runs = list(_iter_runs_1d(g[row, :], 0, sep_col, bg))
        ordered = reversed(runs)
    else:
        ordered = _iter_runs_1d(g[row, :], sep_col + 1, g.shape[1], bg)
    return [(c1 - c0 + 1, col) for c0, c1, col in ordered]


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    donor_bars: list[tuple[int, int]],
    bg: int,
) -> None:
    w = out.shape[1]
    for length, color in donor_bars:
        if length <= 0:
            continue
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == bg:
                out[row, cc] = color


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

    row_bars: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        bars = _extract_row_bars(g, r, sep_col, bars_on_left, bg)
        if bars:
            row_bars[r] = bars

    active_rows = sorted(row_bars)
    partner: dict[int, int] = {}
    idx = 0
    while idx < len(active_rows):
        r = active_rows[idx]
        if idx + 1 < len(active_rows) and active_rows[idx + 1] == r + 1:
            partner[r] = active_rows[idx + 1]
            partner[active_rows[idx + 1]] = r
            idx += 2
        else:
            partner[r] = r
            idx += 1

    for r in active_rows:
        donor_row = partner.get(r, r)
        _paint_extension(out, r, sep_col, bars_on_left, row_bars[donor_row], bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PAIR_MIN, PAIR_MAX = 1, 2

P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_pair_starts(rng: np.random.Generator, h: int, pair_count: int) -> list[int] | None:
    starts = [
        combo
        for combo in itertools.combinations(range(h - 1), pair_count)
        if all(combo[i + 1] - combo[i] >= 3 for i in range(pair_count - 1))
    ]
    if not starts:
        return None
    choice = starts[int(rng.integers(0, len(starts)))]
    return list(map(int, choice))


def _sample_colors(rng: np.random.Generator, pair_count: int) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    needed = 2 * pair_count
    picked = list(map(int, rng.choice(bar_pool, size=needed, replace=False)))
    return sep_color, picked


def _sample_pair_bars(
    rng: np.random.Generator, colors: tuple[int, int], bar_side_len: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    sep_gap = 1 if (bar_side_len >= 4 and rng.random() < P_SEP_GAP) else 0
    capacity = bar_side_len - sep_gap
    if capacity < 2:
        raise ValueError("capacity too small for a paired row")

    candidates = [
        (near_len, far_len)
        for near_len in range(1, capacity)
        for far_len in range(1, capacity - near_len + 1)
        if near_len != far_len
        and not (capacity >= 5 and min(near_len, far_len) == 1)
    ]
    if not candidates:
        candidates = [
            (near_len, far_len)
            for near_len in range(1, capacity)
            for far_len in range(1, capacity - near_len + 1)
            if near_len != far_len
        ]
    if not candidates:
        raise ValueError("no distinct-length bar pair available")

    near_len, far_len = candidates[int(rng.integers(0, len(candidates)))]
    c_near, c_far = colors
    top = [(near_len, c_near), (far_len, c_far)]
    bottom = [(far_len, c_far), (near_len, c_near)]
    return top, bottom, sep_gap


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


def _build_output(
    base: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    row_bars_map: dict[int, list[tuple[int, int]]],
    partner_map: dict[int, int],
    use_partner: bool,
) -> np.ndarray:
    out = base.copy()
    for row, bars in row_bars_map.items():
        donor_bars = row_bars_map[partner_map[row]] if use_partner else bars
        _paint_extension(out, row, sep_col, bars_on_left, donor_bars, bg=0)
    return out


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

    max_pairs = min(PAIR_MAX, (h + 1) // 3)
    if max_pairs < PAIR_MIN:
        return None
    pair_count = int(rng.integers(PAIR_MIN, max_pairs + 1))
    pair_starts = _choose_pair_starts(rng, h, pair_count)
    if pair_starts is None:
        return None

    sep_color, pair_palette = _sample_colors(rng, pair_count)
    active_rows = {r for start in pair_starts for r in (start, start + 1)}
    inactive_rows = [r for r in range(h) if r not in active_rows]

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    partner_map: dict[int, int] = {}
    for idx, start in enumerate(pair_starts):
        colors = (pair_palette[2 * idx], pair_palette[2 * idx + 1])
        top_bars, bottom_bars, sep_gap = _sample_pair_bars(rng, colors, bar_side_len)
        top_row, bottom_row = start, start + 1
        row_bars_map[top_row] = top_bars
        row_bars_map[bottom_row] = bottom_bars
        partner_map[top_row] = bottom_row
        partner_map[bottom_row] = top_row
        _place_bars(base, top_row, sep_col, bars_on_left, top_bars, sep_gap)
        _place_bars(base, bottom_row, sep_col, bars_on_left, bottom_bars, sep_gap)

    out = _build_output(base, sep_col, bars_on_left, row_bars_map, partner_map, use_partner=True)
    baseline = _build_output(base, sep_col, bars_on_left, row_bars_map, partner_map, use_partner=False)
    if np.array_equal(out, baseline):
        return None

    inp = base.copy()
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
