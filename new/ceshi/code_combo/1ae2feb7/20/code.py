# Combo variant of 1ae2feb7 (index 20)
# combo_spec: D6/0+D10/1+D17/0
# Name: 成对活跃行间隔色条裁剪输出

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import colors_present, get_bg

# Design analysis for the combo variant
# a) Concretization and behavior trace
#    - Source task: each active row owns 1-2 bars on one side of a vertical separator.
#      Each bar paints the opposite side periodically using its length as the step,
#      starting immediately across the separator; the full grid is returned.
#    - Combo task:
#      1. Active rows occur only as adjacent pairs (r, r+1), with at least one empty
#         row between pairs.
#      2. Inside each pair, both rows contain exactly two bars with one internal gap.
#         The two rows use the same two colors, but near/far order and lengths are
#         swapped, so the partner relation is visually explicit.
#      3. A row is not painted from its own bars. Instead, it uses its partner row's
#         bars as the donor template.
#      4. The donor bar's distance from the separator is now meaningful: a donor bar
#         at distance D and length L paints the target row at offsets D, D+L, D+2L, ...
#      5. The final answer is cropped to the ext_side rectangle only.
#    - Small left-side trace with separator at column 7 and ext width 8:
#        row r   input bars:  ..BB.111|........
#                           near 111 has (dist=0,len=3), far BB has (dist=4,len=2)
#        row r+1 input bars: ..22.AAA|........
#                           near AAA has (dist=0,len=3), far 22 has (dist=4,len=2)
#      In the combo rule, row r is painted from row r+1's bars and row r+1 from row r's.
#      The far donor bar does not start next to the separator; it starts 4 cells later.
#      After painting the full target side, only the target-side crop is returned.
#    - Difference from source: pairing changes which row donates, gaps make distance
#      affect phase, and cropping removes the separator/bar side from the output.
# b) Complexity calibration
#    - Every active row has exactly two bars and a real internal gap, so D10 is always
#      observable. Distances stay in a small discrete range because only sep-gap 0/1
#      and inter-gap 1/2 are used.
#    - Every active row belongs to a visibly matched adjacent pair sharing the same two
#      colors in reversed order, so D17 is always observable and does not rely on color
#      ids or hidden metadata.
#    - ext_side is kept at width >= 6 so the cropped outputs still show enough periodic
#      structure for humans to recover both the donor-row swap and the phase shift.
# c) Implicit-assumption audit against the source main()
#    - `_find_sep_col` remains valid: the separator is still a nearly full monochrome
#      column and bars still live entirely on one side.
#    - The source code flattened all bars and implicitly assumed "row r paints itself".
#      That assumption is false here. We must extract bars per row, pair adjacent active
#      rows, and choose donor rows before painting.
#    - The source used `dist` only for nearest-first ordering. Here `dist` also shifts
#      the first painted target cell; otherwise inter-bar gaps would carry no meaning.
#    - The source returned the whole solved grid. That is no longer valid because D6
#      requires cropping to the ext_side only.
#    - New solver assumption: every meaningful active row has a partner immediately
#      above or below. The solver still falls back to self-donation for malformed input;
#      the generator guarantees strict pairing so the intended branch is always used.
# d) Placement legality
#    - Generator samples bar-side width large enough for two bars plus one gap.
#    - Bars are represented as `(dist_from_separator, length, color)` and placed with
#      closed-form coordinates, so they never overlap or cross the separator.
#    - A bar is only accepted if its shifted periodic pattern paints at least one visible
#      ext-side cell after nearest-first blocking on an empty target row.
#    - Pair starts differ by at least 3 rows, so adjacent-pair grouping is unambiguous.
# e) Visual observability
#    - Input visibly shows adjacent row pairs with mirrored color/length layouts.
#    - Output keeps only the painted target strip, forcing the solver to relate each
#      cropped row back to the paired bar rows in the input.
#    - Because partner donation and distance-based phase shift are both rejected if they
#      collapse to simpler baselines, the training examples consistently display the
#      fusion rather than accidental source-like cases.
# f) Visual cleanliness
#    - Only three layers remain: compact gapped bars on one side, one separator column,
#      and sparse dots on the ext side. Cropping the output removes the cluttering half.
#    - Rows inside a pair are adjacent, but different pairs are separated by empty rows,
#      so no cross-pair ambiguity appears in the cropped output.
# g) Color-rule audit
#    - Colors are copied directly from visible donor bars. No arithmetic on color ids is
#      introduced.
#    - Colors serve only as structural labels that make pairing and donor swapping easy
#      to see; the actual transformation is driven by row pairing, bar length, and bar
#      distance from the separator.


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
        if int(row[c]) == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == col:
            c += 1
        yield c0, c - 1, col


def _extract_row_bars(
    g: np.ndarray, row: int, sep_col: int, bars_on_left: bool, bg: int
) -> list[tuple[int, int, int]]:
    """Return row bars as (dist_from_separator, length, color), ordered near -> far."""
    bars: list[tuple[int, int, int]] = []
    if bars_on_left:
        runs = list(_iter_runs_1d(g[row, :], 0, sep_col, bg))
        for c0, c1, col in reversed(runs):
            bars.append(((sep_col - 1) - c1, c1 - c0 + 1, col))
    else:
        for c0, c1, col in _iter_runs_1d(g[row, :], sep_col + 1, g.shape[1], bg):
            bars.append((c0 - (sep_col + 1), c1 - c0 + 1, col))
    return bars


def _build_partner_map(active_rows: list[int]) -> dict[int, int]:
    partner: dict[int, int] = {}
    idx = 0
    while idx < len(active_rows):
        row = active_rows[idx]
        if idx + 1 < len(active_rows) and active_rows[idx + 1] == row + 1:
            mate = active_rows[idx + 1]
            partner[row] = mate
            partner[mate] = row
            idx += 2
        else:
            partner[row] = row
            idx += 1
    return partner


def _paint_donor_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    donor_bars: list[tuple[int, int, int]],
    bg: int,
    use_shift: bool,
) -> None:
    w = out.shape[1]
    for dist, length, color in donor_bars:
        if length <= 0:
            continue
        shift = dist if use_shift else 0
        if bars_on_left:
            start = sep_col + 1 + shift
            positions = range(start, w, length)
        else:
            start = sep_col - 1 - shift
            positions = range(start, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == bg:
                out[row, cc] = color


def _crop_ext_side(full: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    if bars_on_left:
        return full[:, sep_col + 1 :].copy()
    return full[:, :sep_col].copy()


def _simulate(
    grid: np.ndarray,
    *,
    use_partner: bool,
    use_shift: bool,
    crop: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    row_bars: dict[int, list[tuple[int, int, int]]] = {}
    for row in range(g.shape[0]):
        bars = _extract_row_bars(g, row, sep_col, bars_on_left, bg)
        if bars:
            row_bars[row] = bars

    active_rows = sorted(row_bars)
    partner = _build_partner_map(active_rows)

    for row in active_rows:
        donor_row = partner[row] if use_partner else row
        _paint_donor_bars(out, row, sep_col, bars_on_left, row_bars[donor_row], bg, use_shift)

    if crop:
        return _crop_ext_side(out, sep_col, bars_on_left)
    return out


# Overview:
#   Adjacent active rows form explicit pairs. Each row's cropped output is generated
#   from its partner row's gapped bars, not from its own.
# Input structure:
#   A nearly full vertical separator splits the grid into one bar side and one blank
#   ext side. Active rows appear only as adjacent pairs, and every active row contains
#   exactly two monochrome bars separated by background cells.
# Rule:
#   Extract bars as `(distance, length, color)` ordered from near to far. Pair adjacent
#   active rows. For each row, take the partner row's bars and paint periodic dots on
#   the row's ext side: the first dot is `distance` cells away from the separator and
#   the period is `length`, with nearer donor bars processed first and never overwritten.
#   Return only the ext-side crop.
# Key insight:
#   The gap between bars is not decorative: it changes the far bar's distance, which
#   shifts the phase of the painted dots. The pair structure tells you which row's bar
#   template to apply before cropping away the separator and source bars.
def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, use_partner=True, use_shift=True, crop=True)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 5, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 6, 14

PAIR_MIN, PAIR_MAX = 1, 2

P_SEP_GAP = 0.18
INTER_GAP_MIN, INTER_GAP_MAX = 1, 2
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _choose_pair_starts(rng: np.random.Generator, h: int, pair_count: int) -> list[int] | None:
    candidates = [
        combo
        for combo in itertools.combinations(range(h - 1), pair_count)
        if all(combo[i + 1] - combo[i] >= 3 for i in range(pair_count - 1))
    ]
    if not candidates:
        return None
    choice = candidates[int(rng.integers(0, len(candidates)))]
    return list(map(int, choice))


def _sample_colors(rng: np.random.Generator, pair_count: int) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    picked = list(map(int, rng.choice(bar_pool, size=2 * pair_count, replace=False)))
    return sep_color, picked


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int]],
) -> None:
    for dist, length, color in bars:
        if bars_on_left:
            c1 = sep_col - 1 - dist
            c0 = c1 - length + 1
        else:
            c0 = sep_col + 1 + dist
            c1 = c0 + length - 1
        out[row, c0 : c1 + 1] = color


def _ext_visibility_counts(ext_side_len: int, bars: list[tuple[int, int, int]]) -> list[int]:
    row = np.zeros(ext_side_len, dtype=int)
    counts: list[int] = []
    for dist, length, color in bars:
        count = 0
        for pos in range(dist, ext_side_len, length):
            if int(row[pos]) == 0:
                row[pos] = color
                count += 1
        counts.append(count)
    return counts


def _sample_pair_bars(
    rng: np.random.Generator,
    colors: tuple[int, int],
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    inter_gap = int(rng.integers(INTER_GAP_MIN, INTER_GAP_MAX + 1))
    capacity = bar_side_len - sep_gap - inter_gap
    if capacity < 2:
        return None

    def build_candidates(min_len: int) -> list[tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]]:
        candidates: list[tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]] = []
        for near_len in range(min_len, capacity):
            for far_len in range(min_len, capacity - near_len + 1):
                if near_len == far_len:
                    continue
                top = [
                    (sep_gap, near_len, colors[0]),
                    (sep_gap + near_len + inter_gap, far_len, colors[1]),
                ]
                bottom = [
                    (sep_gap, far_len, colors[1]),
                    (sep_gap + far_len + inter_gap, near_len, colors[0]),
                ]
                top_counts = _ext_visibility_counts(ext_side_len, top)
                bottom_counts = _ext_visibility_counts(ext_side_len, bottom)
                if any(count == 0 for count in top_counts + bottom_counts):
                    continue
                candidates.append((top, bottom))
        return candidates

    candidates = build_candidates(min_len=2 if capacity >= 5 else 1)
    if not candidates:
        candidates = build_candidates(min_len=1)
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


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

    max_pairs = min(PAIR_MAX, (h + 1) // 3)
    if max_pairs < PAIR_MIN:
        return None
    pair_count = int(rng.integers(PAIR_MIN, max_pairs + 1))
    pair_starts = _choose_pair_starts(rng, h, pair_count)
    if pair_starts is None:
        return None

    sep_color, pair_palette = _sample_colors(rng, pair_count)
    active_rows = {row for start in pair_starts for row in (start, start + 1)}
    inactive_rows = [row for row in range(h) if row not in active_rows]

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    for idx, start in enumerate(pair_starts):
        bars = _sample_pair_bars(
            rng,
            (pair_palette[2 * idx], pair_palette[2 * idx + 1]),
            bar_side_len,
            ext_side_len,
        )
        if bars is None:
            return None
        top_bars, bottom_bars = bars
        _place_bars(inp, start, sep_col, bars_on_left, top_bars)
        _place_bars(inp, start + 1, sep_col, bars_on_left, bottom_bars)

    expected = _simulate(inp, use_partner=True, use_shift=True, crop=True)
    baseline_source = _simulate(inp, use_partner=False, use_shift=False, crop=True)
    baseline_shift_only = _simulate(inp, use_partner=False, use_shift=True, crop=True)
    baseline_pair_only = _simulate(inp, use_partner=True, use_shift=False, crop=True)

    if expected.shape != (h, ext_side_len):
        return None
    if np.count_nonzero(expected) < 2 * pair_count:
        return None
    if np.array_equal(expected, baseline_source):
        return None
    if np.array_equal(expected, baseline_shift_only):
        return None
    if np.array_equal(expected, baseline_pair_only):
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
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
