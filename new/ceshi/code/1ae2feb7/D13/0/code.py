# Variant of 1ae2feb7
# Dimension: D13 — 颜色约束与关系 (meta)
# Change: 色条颜色互不相同且与分隔条颜色不同，颜色数5-8种；涂点颜色等于源色条颜色... → 涂点颜色与源色条颜色不同（颜色映射）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# -----------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# a) Concrete behavior + trace
#    The geometry stays the same as 1ae2feb7:
#    - one mostly solid vertical separator column;
#    - 4-5 sparse active rows on one side, each with 1-2 horizontal solid bars;
#    - the other side is blank in the input and gets periodic dots in the output.
#    The only changed step is "which color gets written on ext_side".
#    Original row example:
#      input row: [1 1 1 | sep | 0 0 0 0 0 0]
#      original output: source color 1 paints positions 0,3,6,... as color 1.
#      new output: source color 1 paints the same positions, but with MAP[1].
#    If a row has two bars, proximity order and no-overwrite are unchanged:
#      near bar length decides the first periodic mask, far bar only fills the
#      remaining blank beats.  The variant changes only the written color, not
#      the mask or the priority.
#
# b) Complexity calibration
#    A per-sample random mapping would be impossible to infer from 2-4 train
#    pairs, so the mapping is fixed globally and low-entropy:
#      1 -> 2 -> 3 -> 4 -> 6 -> 1
#    To make this visible, every generated sample is forced to include all five
#    source colors on the bar side.  This keeps the task at "learn a stable
#    color pairing table" complexity rather than "guess an arbitrary hidden map".
#
# c) Implicit-assumption audit of the original main()
#    - _find_sep_col(...): depends only on the separator column being uniform;
#      still valid because D13 changes only ext-side output colors.
#    - left_mass/right_mass side detection: still valid because inputs keep the
#      extension side blank.
#    - _iter_runs_1d over the bar side: still valid because bars are still solid
#      monochrome runs.
#    - bars.sort(...): still valid because proximity priority is unchanged.
#    - out[r, cc] = col: this is the one D13-sensitive line.  It assumed the
#      fill color equals the source bar color.  In the variant it must become
#      out[r, cc] = COLOR_MAP[col].
#    New assumption introduced by the variant: every bar color lies in the
#    mapping domain.  The generator guarantees this by restricting source bars
#    to SOURCE_COLORS and separator colors to a disjoint set.
#
# d) Placement legality
#    Bars are still placed contiguously on one side of the separator, never
#    crossing it and never leaving the canvas.  For 2-bar rows we only sample
#    length pairs that can fit into the available width.  The generator also
#    rejects rows where the farther bar would be fully occluded on ext_side,
#    because such rows would hide one of the color-pair examples from the user.
#
# e) Visual observability
#    The mapping must be visible from a few train pairs.  Therefore:
#    - source colors come from a fixed 5-color set;
#    - every sample exposes all 5 source colors;
#    - every bar must contribute at least one visible mapped dot on ext_side.
#    This makes the changed rule obvious: same periodic masks as before, but
#    every source color consistently paints a different color.
#
# f) Visual cleanliness
#    We keep the original sparse row spacing and blank ext_side input.  The only
#    overlap on ext_side is the original "nearest-first, no-overwrite" layering.
#    Rejecting fully hidden bars prevents rows whose output would look like a
#    single-color mess with an invisible second rule contributor.
#
# g) Color-change inferability
#    The mapping is not derived from color ids at runtime.  It is a fixed visible
#    training relation encoded by repeated source->target examples:
#      1->2, 2->3, 3->4, 4->6, 6->1
#    Since every sample contains the full source set, train examples provide
#    enough repeated evidence for humans to read off the table and use it on test.
# -----------------------------------------------------------------------------


SOURCE_COLORS = (1, 2, 3, 4, 6)
COLOR_MAP = {1: 2, 2: 3, 3: 4, 4: 6, 6: 1}
SEPARATOR_COLORS = tuple(c for c in range(1, 10) if c not in SOURCE_COLORS)


def _find_sep_col(g, bg):
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _iter_runs_1d(row, c_lo, c_hi, bg):
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
        step = length
        if step <= 0 or col not in COLOR_MAP:
            continue
        paint_col = int(COLOR_MAP[col])
        if bars_on_left:
            for cc in range(sep_col + 1, w, step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = paint_col
        else:
            for cc in range(sep_col - 1, -1, -step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = paint_col

    return out


# --------------- generator (D13 color-mapping variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

ACTIVE_MIN, ACTIVE_MAX = 4, 5

P_TWO_BARS_EXTRA = 0.35
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_separator_color(rng: np.random.Generator) -> int:
    return int(rng.choice(SEPARATOR_COLORS))


def _choose_row_bar_counts(rng: np.random.Generator, active_count: int) -> list[int]:
    counts = [1] * active_count
    extra_needed = max(0, len(SOURCE_COLORS) - active_count)
    order = list(map(int, rng.permutation(active_count)))

    for idx in order[:extra_needed]:
        counts[idx] = 2

    for idx in order[extra_needed:]:
        if rng.random() < P_TWO_BARS_EXTRA:
            counts[idx] = 2

    return counts


def _assign_row_colors(
    rng: np.random.Generator,
    row_bar_counts: list[int],
) -> list[list[int]]:
    total_slots = sum(row_bar_counts)
    extras_needed = total_slots - len(SOURCE_COLORS)
    extras = []
    if extras_needed > 0:
        extras = list(map(int, rng.choice(SOURCE_COLORS, size=extras_needed, replace=True)))

    base = list(SOURCE_COLORS) + extras
    for _ in range(200):
        perm = list(map(int, rng.permutation(base)))
        rows = []
        idx = 0
        ok = True
        for count in row_bar_counts:
            colors = perm[idx : idx + count]
            idx += count
            if len(set(colors)) != count:
                ok = False
                break
            rows.append(colors)
        if ok and set(c for row in rows for c in row) == set(SOURCE_COLORS):
            return rows

    raise RuntimeError("failed to assign per-row source colors")


def _sample_row_bars(
    rng: np.random.Generator,
    row_colors: list[int],
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    if len(row_colors) == 2 and bar_side_len - sep_gap < 3:
        sep_gap = 0

    capacity = bar_side_len - sep_gap
    if len(row_colors) == 1:
        length = int(rng.integers(1, capacity + 1))
        return [(length, row_colors[0])], sep_gap

    candidates = []
    for near_len in range(2, capacity):
        for far_len in range(1, capacity - near_len + 1):
            if far_len % near_len != 0:
                candidates.append((near_len, far_len))
    if not candidates:
        raise RuntimeError("no visible 2-bar length pair available")

    near_len, far_len = candidates[int(rng.integers(0, len(candidates)))]
    colors = row_colors[:]
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
) -> list[int]:
    w = out.shape[1]
    visible_counts = [0] * len(bars)
    for idx, (length, source_color) in enumerate(bars):
        paint_color = int(COLOR_MAP[source_color])
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = paint_color
                visible_counts[idx] += 1
    return visible_counts


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

    sep_color = _sample_separator_color(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_bar_counts = _choose_row_bar_counts(rng, active_count)
    row_colors = _assign_row_colors(rng, row_bar_counts)

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for row, colors in zip(active_rows, row_colors):
        try:
            bars, sep_gap = _sample_row_bars(rng, colors, bar_side_len)
        except RuntimeError:
            return None
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        visible_counts = _paint_extension(out, row, sep_col, bars_on_left, bars)
        if any(count == 0 for count in visible_counts):
            return None

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None

    non_bg = colors_present(inp, exclude=[0])
    if sep_color not in non_bg:
        return None
    if set(c for c in non_bg if c != sep_color) != set(SOURCE_COLORS):
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
