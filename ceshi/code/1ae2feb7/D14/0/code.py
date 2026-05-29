# Variant of 1ae2feb7
# Dimension: D14 — 涂色粒度（每次涂色面积） (trajectory)
# Change: 每个周期位置涂1个像素——涂点是单个格子... → 涂色宽度等于色条长度
#
# Design analysis for this variant
# - Source task: None, so the design is derived only from the original solver/spec.
# - Concrete behavior:
#   The input still contains one vertical separator column and 1-2 horizontal bars on each
#   active row, all on the same side of the separator. The opposite side starts blank.
#   In the original task, a bar of length L painted only the single cells at periodic starts
#   p, p+L, p+2L, ... on the extension side. In this variant, each periodic start expands to
#   a solid segment of width L. If the bars are on the left, the extension side is on the
#   right and each start p paints [p, p+L-1]. If the bars are on the right, the extension
#   side is on the left and each anchor p paints [p-L+1, p]. This keeps the filled segment on
#   the extension side instead of crossing the separator.
# - Small trace:
#   Suppose the separator is at column 5, bars are on the left, and a row has one bar of
#   length 3, color 7. Original output paints columns 6, 9, 12, ... as isolated 7s.
#   Variant output paints [6:9), [9:12), [12:15), ... so the right side becomes a continuous
#   band of 7s until the boundary. With two bars, the nearest bar is still processed first and
#   already-painted cells remain unchanged, so farther bars only fill gaps left by the nearer
#   bar.
# - Complexity calibration:
#   The object inventory, separator detection, row spacing, and nearest-first priority all stay
#   the same. Only the footprint of each periodic paint event changes. This keeps the puzzle
#   close to the original while making the visual effect obvious: rows that were sparse dotted
#   patterns now become dense bands or dense partial tilings.
# - Implicit-assumption audit for main():
#   `_find_sep_col` and `_iter_runs_1d` rely only on the input structure and remain valid
#   because the input still clears the extension side.
#   The bar collection logic still assumes bars are horizontal monochrome runs on one side of
#   the separator; the generator preserves that.
#   The original extension loop assumed "one periodic event == one writable cell". That
#   assumption no longer holds. The new loop must expand each event into a segment and apply
#   the no-overwrite rule cell-by-cell within that segment.
#   Sorting by distance to separator is still valid because the variant keeps the same
#   nearest-first priority semantics.
# - Placement legality:
#   The generator remains output-first. It places bars exactly as before, then simulates dense
#   periodic segments on the extension side, truncating the last segment at the canvas edge.
#   Because the extension side is blank in the input, no dense segment can collide with bar-side
#   objects or the separator. Two bars on the same row are handled by ordered painting plus
#   bg-only writes, so there is no illegal overwrite.
# - Visual observability / cleanliness:
#   Samples are rejected unless the dense variant output differs from the original point-based
#   output on the same input. This avoids degenerate cases where all visible behavior collapses
#   back to the original rule, such as a nearest bar of length 1 hiding everything else.
#   The layout remains visually clean: separator, bar side, and extension side stay distinct.
# - New solver assumptions and generator guarantees:
#   `main()` assumes the extension side is entirely background in the input, that all bars lie
#   on one side, and that bar runs are horizontal. `generate_input()` constructs exactly that.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


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


def _paint_periodic_segments(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
) -> None:
    """Paint dense periodic segments, preserving any already-painted cells."""
    if length <= 0:
        return

    w = out.shape[1]
    if bars_on_left:
        starts = range(sep_col + 1, w, length)
        for start in starts:
            end = min(w, start + length)
            segment = out[row, start:end]
            segment[segment == bg] = color
    else:
        anchors = range(sep_col - 1, -1, -length)
        for anchor in anchors:
            start = max(0, anchor - length + 1)
            end = anchor + 1
            segment = out[row, start:end]
            segment[segment == bg] = color


def _paint_periodic_points(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
) -> None:
    """Original task behavior, kept only for degeneration checks in the generator."""
    if length <= 0:
        return

    w = out.shape[1]
    if bars_on_left:
        for cc in range(sep_col + 1, w, length):
            if int(out[row, cc]) == bg:
                out[row, cc] = color
    else:
        for cc in range(sep_col - 1, -1, -length):
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
        _paint_periodic_segments(out, r, sep_col, bars_on_left, length, col, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    """Pick separator color and a disjoint bar-color palette."""
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
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (length, color) ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    two_bars = rng.random() < P_TWO_BARS

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
    """Paint bars on the bar side, stacking from nearest-to-separator outward."""
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


def _paint_extension_dense(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the dense periodic extension used by the variant."""
    for length, color in bars:
        _paint_periodic_segments(out, row, sep_col, bars_on_left, length, color, bg=0)


def _paint_extension_points(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the original point-extension rule for degeneration checks."""
    for length, color in bars:
        _paint_periodic_points(out, row, sep_col, bars_on_left, length, color, bg=0)


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

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension_dense(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    point_out = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension_points(point_out, row, sep_col, bars_on_left, bars)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(point_out, out):
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
