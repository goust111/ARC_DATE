# Combo variant of 1ae2feb7 (index 7)
# combo_spec: D5/1+D13/0
# Name: 双重颜色变换
#
# Design analysis
# ===============
# a) Concrete behavior + trace
#    The geometry stays the same as the base task:
#    - one almost-solid vertical separator column;
#    - 4-5 sparse active rows on exactly one side of that separator;
#    - each active row has 1-2 horizontal monochrome bars, ordered by distance to
#      the separator;
#    - the opposite side is blank in the input and receives periodic dots in the
#      output.
#    The fused change is purely about output colors, but it happens in two
#    coupled stages:
#    - Stage 1: a source bar of color `src` still paints periodic dots with
#      period = bar length and priority = nearer bar first, but the painted color
#      is `PAINT_MAP[src]`, not `src`.
#    - Stage 2: after ext-side painting is finished, the original bar pixels on
#      the bar side are recolored to `BAR_RECOLOR_MAP[src]`.
#    Small row trace:
#      input : [1 1 1 | 5 | 0 0 0 0 0 0 0]
#      base  : [1 1 1 | 5 | 1 0 0 1 0 0 1]
#      combo : [4 4 4 | 5 | 2 0 0 2 0 0 2]
#      because PAINT_MAP[1] = 2 and BAR_RECOLOR_MAP[1] = 4.
#    Two-bar trace:
#      input : [3 3 | 4 4 4 | 5 | 0 0 0 0 0 0 0]
#      order : near bar (len 2, src 3) paints 4 at positions 0,2,4,6,...
#              far  bar (len 3, src 4) then fills only still-blank positions
#              with 6 at 3,6,... as available.
#      final : the left bars become BAR_RECOLOR_MAP[3] = 1 and
#              BAR_RECOLOR_MAP[4] = 2.
#
# b) Complexity calibration
#    Two arbitrary per-sample color mappings would be impossible to infer from a
#    few ARC examples, so both mappings are fixed global permutations on a small
#    five-color source set:
#      source colors      : 1, 2, 3, 4, 6
#      ext paint mapping  : 1->2, 2->3, 3->4, 4->6, 6->1
#      bar recolor mapping: 1->4, 2->6, 3->1, 4->2, 6->3
#    Every generated sample is forced to expose all five source colors on the
#    input bar side, so both mappings are fully visible in each example pair.
#    Both permutations have no fixed points, which makes the fused change easy
#    to notice and prevents degeneration back to the base task.
#
# c) Implicit-assumption review of main()
#    - `_find_sep_col(...)` assumes one near-solid uniform separator column.
#      The generator still guarantees that.
#    - left/right mass comparison assumes non-separator content exists only on
#      one side in the input.  The generator still keeps ext_side fully blank.
#    - `_iter_runs_1d(...)` assumes bars are contiguous monochrome runs; the
#      fusion changes only colors written to the output, not input geometry.
#    - Proximity sorting by `(dist, row, c0)` still works because the priority
#      rule is unchanged.
#    - The original write `out[r, cc] = col` is no longer valid: ext dots must
#      use `PAINT_MAP[col]`.
#    - Recoloring bars after painting introduces a new assumption: bar segments
#      must be collected from the input before recoloring, otherwise adjacent
#      segments could visually merge and destroy source-color evidence.
#    - Another new assumption is that every source bar color belongs to both
#      mapping domains.  The generator enforces this by restricting source bars
#      to `SOURCE_COLORS`.
#
# d) Placement legality
#    Bars still lie entirely on one side of the separator, never cross it, and
#    fit within the chosen bar-side width.  When a row has two bars, they remain
#    adjacent and ordered nearest-to-farthest from the separator.  The generator
#    also checks that every bar produces at least one visible ext-side dot after
#    no-overwrite competition, so no color example is hidden.
#
# e) Visual observability
#    A human sees two simultaneous regularities:
#    - ext-side dots follow the original periodic geometry but consistently use
#      a different color from the source bar;
#    - the source bars themselves also change color in the output, but according
#      to a different fixed mapping.
#    Because all five source colors are shown in every sample, 2-3 train pairs
#    are enough to read off both tables by direct comparison.
#
# f) Visual cleanliness
#    The layout remains sparse: active rows keep at least one empty row between
#    them, the input ext_side stays empty, and overlap on the output ext_side is
#    still only the original nearest-first masking.  Using a separator color
#    outside the source/mapped palette prevents the separator from being confused
#    with either mapping.
#
# g) Change inferability
#    The fusion would be too weak if one mapping collapsed to the identity or if
#    all bars recolored to a single constant color.  Instead, both output effects
#    are source-dependent and fully repeated across the fixed five-color domain.
#    This makes the "double color transform" a visible rule rather than a hidden
#    implementation detail.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


SOURCE_COLORS = (1, 2, 3, 4, 6)
PAINT_MAP = {1: 2, 2: 3, 3: 4, 4: 6, 6: 1}
BAR_RECOLOR_MAP = {1: 4, 2: 6, 3: 1, 4: 2, 6: 3}
SEPARATOR_COLORS = tuple(c for c in range(1, 10) if c not in SOURCE_COLORS)


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
        if int(row[c]) == int(bg):
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == col:
            c += 1
        yield c0, c - 1, col


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Return bars as (dist_to_sep, row, c0, c1, length, source_color)."""
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
    for r in range(h):
        if bars_on_left:
            runs = _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            runs = _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> list[int]:
    """Paint mapped periodic dots and return visible dot counts per bar."""
    w = out.shape[1]
    visible_counts = [0] * len(bars)
    for idx, (length, source_color) in enumerate(bars):
        if length <= 0:
            continue
        paint_color = int(PAINT_MAP[source_color])
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == 0:
                out[row, cc] = paint_color
                visible_counts[idx] += 1
    return visible_counts


def _place_bars(
    grid: np.ndarray,
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
            grid[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            grid[row, cursor : cursor + length] = color
            cursor += length


# Overview:
#   Extend each source bar across the blank side with the original nearest-first
#   periodic geometry, but write mapped colors on ext_side and then recolor the
#   source bars themselves with a second mapping.
# Input structure:
#   A mostly solid vertical separator column splits the grid into one blank
#   ext_side and one bar_side containing 4-5 spaced active rows of 1-2 adjacent
#   monochrome bars.  Source bar colors come from a fixed five-color domain.
# Rule:
#   Collect bars from the input, sort them by distance to the separator, paint
#   ext-side dots using `PAINT_MAP[source_color]` with period = bar length and
#   no overwrite, then recolor each original bar to
#   `BAR_RECOLOR_MAP[source_color]`.
# Key insight:
#   Geometry and priority are unchanged from the base task; only the two output
#   color channels change, and they must be inferred as separate source-color
#   mappings.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bars_on_left, bg)

    for _, r, _c0, _c1, length, source_color in bars:
        if length <= 0 or source_color not in PAINT_MAP:
            continue
        paint_color = int(PAINT_MAP[source_color])
        if bars_on_left:
            positions = range(sep_col + 1, g.shape[1], length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[r, cc]) == bg:
                out[r, cc] = paint_color

    for _, r, c0, c1, _length, source_color in bars:
        if source_color not in BAR_RECOLOR_MAP:
            continue
        out[r, c0 : c1 + 1] = int(BAR_RECOLOR_MAP[source_color])

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

ACTIVE_MIN, ACTIVE_MAX = 4, 5

P_TWO_BARS_EXTRA = 0.35
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


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

    pool = list(SOURCE_COLORS) + extras
    for _ in range(300):
        perm = list(map(int, rng.permutation(pool)))
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
    raise RuntimeError("failed to assign row colors covering the whole source domain")


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
        raise RuntimeError("no legal visible 2-bar length pair")

    near_len, far_len = candidates[int(rng.integers(0, len(candidates)))]
    colors = row_colors[:]
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])], sep_gap


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

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        if gap_row == 0 or gap_row == h - 1:
            inp[gap_row, sep_col] = 0

    row_bar_counts = _choose_row_bar_counts(rng, active_count)
    row_colors = _assign_row_colors(rng, row_bar_counts)

    row_specs: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for row, colors in zip(active_rows, row_colors):
        try:
            bars, sep_gap = _sample_row_bars(rng, colors, bar_side_len)
        except RuntimeError:
            return None
        row_specs[row] = (bars, sep_gap)
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None

    non_bg = colors_present(inp, exclude=[0])
    if sep_color not in non_bg:
        return None
    if set(c for c in non_bg if c != sep_color) != set(SOURCE_COLORS):
        return None

    expected = inp.copy()
    total_visible_ext_dots = 0
    for row, (bars, sep_gap) in row_specs.items():
        visible_counts = _paint_extension(expected, row, sep_col, bars_on_left, bars)
        if any(count == 0 for count in visible_counts):
            return None
        total_visible_ext_dots += sum(visible_counts)
        recolored = [(length, int(BAR_RECOLOR_MAP[color])) for length, color in bars]
        _place_bars(expected, row, sep_col, bars_on_left, recolored, sep_gap)

    out = main(inp)
    if not np.array_equal(out, expected):
        return None

    bar_mask = (inp != 0) & (inp != sep_color)
    if not np.all(out[bar_mask] != inp[bar_mask]):
        return None

    if total_visible_ext_dots <= 0:
        return None

    if bars_on_left:
        if np.any(inp[:, sep_col + 1 :] != 0):
            return None
    else:
        if np.any(inp[:, :sep_col] != 0):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
