# Variant of 1ae2feb7
# Dimension: D2 — 轨迹涂色模式 (trajectory)
# Change: periodic_pattern——以间隔=色条像素数在ext_side上涂单色点（位置0, length, 2*len... → 周期性平铺（重复整个色条图案）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg

# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input geometry stays the same: one vertical separator column, 2-5 active rows,
#      and on each active row 1-2 adjacent monochrome bars on one side of the separator.
#    - The changed object is the "trajectory" on ext_side.
#    - Original rule: each bar acts independently. A bar of length L and color c paints
#      ext-side dots at offsets 0, L, 2L, ... from the separator, with nearer bars
#      processed first and later bars forbidden to overwrite.
#    - New rule: each active row is treated as one 1D tile. Read that row's bars from
#      nearest-to-separator to farthest, expand them into a color list, and repeat that
#      whole list periodically on ext_side. The period is the total bar-pattern length.
#    - Example on a left-bar row: bars near->far are [(2, red), (3, blue)].
#      Original output on ext_side: red at offsets 0,2,4,... then blue tries offsets
#      0,3,6,... but can only fill blank cells, so the row becomes a sparse collision
#      pattern. Variant output: ext-side cells become [red, red, blue, blue, blue,
#      red, red, blue, blue, blue, ...].
#    - The optional 1-cell gap between separator and nearest bar remains an input-side
#      placement detail only; the tiled copy still starts immediately across the separator.
#
# b) Complexity calibration
#    - If rows were allowed to have only a single 1-pixel bar, periodic tiling would
#      collapse to the original behavior. To keep the D2 change visible, generation
#      enforces total bar-pattern length >= 2 on every active row.
#    - To make "repetition" legible rather than a single truncated copy, generator also
#      constrains each row pattern to be short enough that ext_side usually contains at
#      least two periods (or as many as the grid width allows when ext_side is tiny).
#    - At least one row per sample is forced to have two bars, so the copied multi-color
#      arrangement is visible in every generated instance.
#
# c) Implicit-assumption audit against the original main()
#    - `_find_sep_col(...)` still works: separator geometry does not change.
#    - `bars_on_left = left_mass >= right_mass` still works because generated inputs keep
#      ext_side empty, so only the real bar side contributes mass.
#    - Original code collected runs globally and used `step = length` per run. That
#      depends on the old assumption that each run emits its own independent sparse
#      sequence. The variant breaks this assumption, so the solver must switch to
#      per-row grouping and use the concatenated row-pattern length as the period.
#    - Original no-overwrite logic was crucial because different bars could target the
#      same ext cells. In the variant, each row has a single deterministic tiling stream,
#      so overlap between bars disappears. We still only write onto bg cells for safety.
#
# d) Placement legality
#    - Bars remain within one side of the separator and are laid contiguously from the
#      separator outward; only the separator gap may be empty.
#    - Total row-pattern length is capped by both bar-side capacity and ext-side length
#      constraints, so no row needs impossible geometry and the tiled output is visible.
#    - The generator is output-first: it builds the fully tiled output, clears ext_side
#      to obtain input, and validates with `main(input) == output`.
#
# e/f/g) Visual observability, cleanliness, and color inference
#    - The rule is purely spatial. No sorting or color remapping uses hidden color ids.
#    - Each active row is independent and horizontally separated by blank rows, so the
#      copied pattern is easy to compare against its source bars.
#    - Two-bar rows expose the key innovation directly: the ext side alternates blocks
#      that reproduce the exact local color arrangement, not isolated dots.
#
# New solver-side assumptions introduced by the variant
#    - Every active row's meaningful source pattern is exactly the concatenation of its
#      non-bg runs ordered by separator proximity.
#    - The generator guarantees those runs are adjacent on the bar side, so "whole bar
#      pattern" is unambiguous and contains no internal blank cells.
#    - The ext side starts blank in generated inputs, preserving the intended tiling.


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


def _row_pattern(
    row: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[int]:
    if bars_on_left:
        runs = list(_iter_runs_1d(row, 0, sep_col, bg))
        runs.sort(key=lambda t: ((sep_col - 1) - t[1], t[0]))
    else:
        runs = list(_iter_runs_1d(row, sep_col + 1, len(row), bg))
        runs.sort(key=lambda t: (t[0] - (sep_col + 1), t[0]))

    pattern: list[int] = []
    for c0, c1, col in runs:
        pattern.extend([int(col)] * (c1 - c0 + 1))
    return pattern


def _tile_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    pattern: list[int],
    bg: int,
) -> None:
    if not pattern:
        return

    period = len(pattern)
    if bars_on_left:
        for k, cc in enumerate(range(sep_col + 1, out.shape[1])):
            if int(out[row, cc]) == bg:
                out[row, cc] = int(pattern[k % period])
    else:
        for k, cc in enumerate(range(sep_col - 1, -1, -1)):
            if int(out[row, cc]) == bg:
                out[row, cc] = int(pattern[k % period])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    for r in range(h):
        pattern = _row_pattern(g[r, :], sep_col, bars_on_left, bg)
        _tile_row(out, r, sep_col, bars_on_left, pattern, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.55
P_SEP_GAP = 0.12
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
    ext_side_len: int,
    *,
    force_two: bool,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    max_total = min(capacity, max(2, ext_side_len // 2))
    if max_total < 2:
        raise ValueError("row pattern capacity too small for visible tiling")

    two_bars = force_two or (rng.random() < P_TWO_BARS)
    if not two_bars:
        length = int(rng.integers(2, max_total + 1))
        return [(length, primary)], sep_gap

    total_len = int(rng.integers(2, max_total + 1))
    near_len = int(rng.integers(1, total_len))
    far_len = total_len - near_len
    if far_len == 0:
        near_len -= 1
        far_len = 1

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


def _pattern_from_bars(bars: list[tuple[int, int]]) -> list[int]:
    pattern: list[int] = []
    for length, color in bars:
        pattern.extend([int(color)] * int(length))
    return pattern


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    pattern = _pattern_from_bars(bars)
    if not pattern:
        return

    period = len(pattern)
    if bars_on_left:
        for k, cc in enumerate(range(sep_col + 1, out.shape[1])):
            out[row, cc] = int(pattern[k % period])
    else:
        for k, cc in enumerate(range(sep_col - 1, -1, -1)):
            out[row, cc] = int(pattern[k % period])


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
    forced_two_row = active_rows[int(rng.integers(0, active_count))]

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            force_two=(row == forced_two_row),
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
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
