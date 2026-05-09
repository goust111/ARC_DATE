# Variant of 1ae2feb7
# Dimension: D14 — 涂色粒度（每次涂色面积） (trajectory)
# Change: 每个周期位置涂1个像素——涂点是单个格子... → 涂色面积递增
#
# Design analysis
# - Concrete behavior:
#   The separator detection, bar-side detection, row structure, bar ordering, periodic step,
#   and no-overwrite policy all stay the same as in the base task. The only changed stage is
#   the actual paint action on the extension side. For a bar of length L, the anchor cells are
#   still at offsets 0, L, 2L, 3L, ... from the separator. In the base task each anchor paints
#   exactly one cell. In this variant the 1st anchor paints a 1-cell horizontal segment, the
#   2nd anchor paints a 2-cell segment, the 3rd anchor paints a 3-cell segment, and so on.
#   Each segment grows away from the separator, so on the right side it extends rightward and
#   on the left side it extends leftward. Existing non-background cells are still protected
#   cell-by-cell.
# - Small trace:
#   Suppose bars are on the left, separator at column 5, and a single bar of length 2 is on row r.
#   Base task: anchors are 6, 8, 10, 12, ... and row r gets colored at [6], [8], [10], [12], ...
#   Variant: the same anchors produce [6], [8,9], [10,11,12], [12,13,14,15], ... clipped by bounds.
#   The anchors still encode the bar length; only paint granularity changes.
# - Complexity calibration:
#   A length-1 bar would usually collapse to the same fully-filled result as the base task, so the
#   generator must not rely on such rows to reveal the new rule. To keep the D14 change visible,
#   generation forces two showcase rows with a single nearest bar of length 2 and keeps the
#   extension side at least 7 cells wide, which yields visible 1/2/3 growth before edge clipping.
# - Implicit-assumption audit for the original main():
#   `_find_sep_col` and `_iter_runs_1d` do not depend on paint granularity and remain valid.
#   The bar ordering by distance also remains valid because the priority rule is unchanged.
#   The critical invalidated assumption is the old write `out[r, cc] = col`: that was safe only
#   because each action touched one pixel. With growing segments we must check every cell in the
#   segment against background before painting, otherwise a later segment could overwrite colors
#   that should be protected.
# - Placement legality:
#   Bars are still placed only on one side, within the original bar-side capacity, optionally with
#   a 1-cell gap to the separator. Extension painting is output-first and clipped to the grid, so
#   no generated target writes out of bounds. Solver and generator share the same segment-painting
#   helper to keep semantics aligned.
# - Visual observability / cleanliness:
#   The grid is still sparse by rows, and the extension side contains only horizontal paint on the
#   corresponding active rows. Two showcase rows provide repeated evidence of the 1/2/3 growth,
#   while other rows preserve the original diversity of one-bar / two-bar interactions. No hidden
#   ordering by color id is introduced.
#
# Source task for the creative change: None.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


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


def _paint_growing_periodic(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    color: int,
    bg: int,
) -> None:
    """Paint increasing-size segments at periodic anchors, without overwriting."""
    if step <= 0:
        return

    w = out.shape[1]
    if bars_on_left:
        anchor = sep_col + 1
        seg_len = 1
        while anchor < w:
            end = min(w, anchor + seg_len)
            seg = out[row, anchor:end]
            seg[seg == bg] = color
            anchor += step
            seg_len += 1
    else:
        anchor = sep_col - 1
        seg_len = 1
        while anchor >= 0:
            start = max(0, anchor - seg_len + 1)
            seg = out[row, start : anchor + 1]
            seg[seg == bg] = color
            anchor -= step
            seg_len += 1


def _paint_point_periodic(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    color: int,
    bg: int,
) -> None:
    """Base-task painting, used only to detect generator degeneration."""
    if step <= 0:
        return

    if bars_on_left:
        for cc in range(sep_col + 1, out.shape[1], step):
            if int(out[row, cc]) == bg:
                out[row, cc] = color
    else:
        for cc in range(sep_col - 1, -1, -step):
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
                bars.append((dist, r, c0, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, length, col in bars:
        _paint_growing_periodic(out, r, sep_col, bars_on_left, length, col, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 7, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
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


def _paint_extension_variant(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    for length, color in bars:
        _paint_growing_periodic(out, row, sep_col, bars_on_left, length, color, bg=0)


def _paint_extension_base(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    for length, color in bars:
        _paint_point_periodic(out, row, sep_col, bars_on_left, length, color, bg=0)


def _variant_metrics(
    inp: np.ndarray,
    out: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    row_bars_map: dict[int, list[tuple[int, int]]],
) -> tuple[int, int]:
    base = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension_base(base, row, sep_col, bars_on_left, bars)

    ext_mask = np.zeros_like(inp, dtype=bool)
    if bars_on_left:
        ext_mask[:, sep_col + 1 :] = True
    else:
        ext_mask[:, :sep_col] = True

    diff_rows = 0
    for row in row_bars_map:
        if np.any(out[row] != base[row]):
            diff_rows += 1

    extra_cells = int(np.sum((out != 0) & (base == 0) & ext_mask))
    return diff_rows, extra_cells


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
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
    showcase_rows = active_rows[: min(2, active_count)]

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        if row in showcase_rows:
            bars = [(2, row_primary[idx])]
            sep_gap = 0
        else:
            bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension_variant(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    diff_rows, extra_cells = _variant_metrics(inp, out, sep_col, bars_on_left, row_bars_map)
    if diff_rows < len(showcase_rows):
        return None
    if extra_cells < len(showcase_rows) * max(1, ext_side_len - 5):
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
