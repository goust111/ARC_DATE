# Combo variant of 1ae2feb7 (index 31)
# combo_spec: D2/1+D12/0
# Name: 两侧色条颜色循环涂色

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - Preserve the original geometry: one nearly solid vertical separator column,
#      sparse active rows, and 1-2 monochrome horizontal bars per active row.
#    - D12 changes placement: some rows have source bars on the left and project
#      dots rightward; other rows have source bars on the right and project dots
#      leftward. Each row still uses only one source side to keep the picture clean.
#    - D2 changes coloring: for one row+side, every bar still contributes the same
#      periodic candidate positions as the source task, but colors are assigned by
#      cycling through that side-row's visible bar colors ordered nearest->farthest
#      from the separator.
#    - Small bilateral trace:
#      left-driven row input  : `333442000000` with separator `2` after the `44`
#      bars nearest->farthest : [(len=2,col=4), (len=3,col=3)]
#      original right dots    : union positions {0,2,3,4,6,8,9,...} but colored
#                               by first-hit ownership => 4,4,3,4,4,4,3,...
#      combo right dots       : same positions, color cycle [4,3] outward =>
#                               4,3,4,3,4,3,4,...
#      right-driven row input : `00000029966` with separator `2` before the `99`
#      bars nearest->farthest : [(len=2,col=9), (len=2,col=6)] on the right side
#      combo left dots        : candidate positions on the left half ordered from
#                               the separator outward, colored 9,6,9,6,...
#    - Difference from the source task:
#      geometry of candidate dots is unchanged, but dot colors are no longer owned
#      by the first bar that reaches them, and both source directions coexist.
#
# b) Complexity calibration
#    - Both sides are guaranteed to appear in every generated grid, so D12 is visible.
#    - Each side is guaranteed to have at least one 2-bar row whose cycle output
#      differs from the original nearest-first overwrite rule, so D2 is visible too.
#    - Side palettes are disjoint and small (2-3 colors per side). This keeps the
#      bilateral structure readable and gives repeated evidence for each side's rule.
#
# c) Implicit assumption review
#    - Original `bars_on_left = left_mass >= right_mass` breaks because bars now
#      exist on both sides. The new solver scans left and right independently.
#    - Original global bar list plus overwrite loop assumes "one source side only"
#      and "candidate color belongs to the earliest bar". Both are invalid now.
#    - The new solver groups by row and by source side, because the cycle order is
#      local to the visible bars on that row-side, while left and right halves are
#      independent targets.
#    - `_find_sep_col` still assumes one mostly full monochrome separator column;
#      the generator preserves that, including the optional single inactive-row gap.
#
# d) Placement legality
#    - Generator samples left/right widths so both halves have room for bars and dots.
#    - Each active row is assigned to exactly one source side, preventing source bars
#      from overlapping across the separator and keeping target clearing unambiguous.
#    - If a row has two bars, they are contiguous and their total length fits entirely
#      inside that side after an optional 1-cell separator gap.
#
# e) Visual observability
#    - From 2-4 examples, a human can see two row families: rows radiating rightward
#      from left bars and rows radiating leftward from right bars.
#    - On each side, 2-bar rows visibly alternate bar colors along the dotted path,
#      which is different from the source task's nearer-bar domination.
#    - No hidden color-id sorting is used: the cycle order is purely spatial
#      (nearest bar to farthest bar on that row-side).
#
# f) Visual cleanliness
#    - Active rows keep at least one blank row between them, so dotted trajectories
#      do not blend vertically.
#    - Because each row uses only one source side, the opposite half is a clean target
#      field rather than a mixture of source bars and new dots.
#
# g) Inference of color change
#    - Colors are not remapped arbitrarily. The output cycle is always built from
#      the visible bar colors on that same row-side.
#    - Disjoint side palettes and repeated multi-bar rows provide enough structure to
#      infer that left-driven rows and right-driven rows follow the same cycle idea
#      independently.
#
# Post-change solver assumptions checked against the generator
#    - Separator detection: guaranteed by the nearly solid single-color separator.
#    - Row-side run extraction: guaranteed because every source object is a horizontal
#      monochrome run fully contained on one side of the separator.
#    - Cycle order: solver assumes sorting by distance to the separator is meaningful;
#      generator keeps bars contiguous so "nearest then farther" is visually explicit.
#    - Empty target side in the input: generator clears the target half for every
#      active row, so solver never has to distinguish source dots from target dots.
# ---------------------------------------------------------------------------


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
        color = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == color:
            c += 1
        yield c0, c - 1, color


def _ext_positions(sep_col: int, width: int, source_side: str, step: int) -> list[int]:
    if step <= 0:
        return []
    if source_side == "left":
        return list(range(sep_col + 1, width, step))
    return list(range(sep_col - 1, -1, -step))


def _collect_side_row_bars(
    g: np.ndarray,
    sep_col: int,
    bg: int,
    *,
    source_side: str,
) -> dict[int, list[tuple[int, int, int, int, int, int]]]:
    h, w = g.shape
    row_bars: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for r in range(h):
        found: list[tuple[int, int, int, int, int, int]] = []
        if source_side == "left":
            runs = _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            for c0, c1, color in runs:
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                found.append((dist, r, c0, c1, length, color))
        else:
            runs = _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
            for c0, c1, color in runs:
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                found.append((dist, r, c0, c1, length, color))
        if found:
            row_bars[r] = sorted(found, key=lambda t: (t[0], t[2]))
    return row_bars


def _paint_original_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> None:
    for _, _, _, _, length, color in bars:
        for cc in _ext_positions(sep_col, out.shape[1], source_side, length):
            if int(out[row, cc]) == bg:
                out[row, cc] = color


def _paint_cycle_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> None:
    if not bars:
        return
    if len(bars) == 1:
        _paint_original_row(out, row, sep_col, source_side, bars, bg)
        return

    cycle = [int(color) for _, _, _, _, _, color in bars]
    positions = sorted(
        {
            cc
            for _, _, _, _, length, _ in bars
            for cc in _ext_positions(sep_col, out.shape[1], source_side, length)
        },
        reverse=(source_side == "right"),
    )
    for idx, cc in enumerate(positions):
        if int(out[row, cc]) == bg:
            out[row, cc] = cycle[idx % len(cycle)]


# Overview:
# Bilateral rows emit the same periodic geometry as the source task, but each row-side
# colors those dots by cycling through its visible bar colors from nearest to farthest.
# Input structure:
# A nearly solid separator column splits the grid; active rows contain 1-2 bars on one
# side only, and the opposite half of that row is empty in the input.
# Rule:
# Left bars paint to the right, right bars paint to the left, and candidate positions
# from all bars on the same row-side are merged then colored by a repeating cycle.
# Key insight:
# The solver must separate geometry from ownership: lengths decide where dots can go,
# while ordered visible bar colors decide how those candidate dots are colored.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Bilateral periodic extension with side-local color cycling.

    Input structure:
    - A nearly solid vertical separator column divides the grid into a left half and a
      right half.
    - Sparse active rows contain 1-2 monochrome horizontal bars on exactly one side of
      the separator; the opposite half of those rows is empty in the input.

    Rule:
    - Left-side bars project periodic dots to the right half, and right-side bars
      project periodic dots to the left half.
    - For one row and one source side, each bar of length L contributes the same
      candidate positions as in the source task: 0, L, 2L, ... away from the separator.
    - Take the union of those candidate positions ordered outward from the separator,
      then color them with a repeating cycle formed by that row-side's bar colors in
      nearest-to-farthest order.

    Key insight:
    - The dotted geometry still comes from bar lengths, but the colors come from a
      row-side cycle, so the solver must group bars by row and by source side instead
      of processing one global nearest-first ownership list.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    for source_side in ("left", "right"):
        row_bars = _collect_side_row_bars(g, sep_col, bg, source_side=source_side)
        for row, bars in row_bars.items():
            _paint_cycle_row(out, row, sep_col, source_side, bars, bg)
    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

SIDE_MIN, SIDE_MAX = 4, 10

LEFT_PALETTE_MIN, LEFT_PALETTE_MAX = 2, 3
RIGHT_PALETTE_MIN, RIGHT_PALETTE_MAX = 2, 3

ACTIVE_TOTAL_MIN, ACTIVE_TOTAL_MAX = 4, 6
SIDE_ROWS_MIN = 2

P_TWO_BARS = 0.78
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_MATCH_LENGTHS = 0.35
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_side_palettes(rng: np.random.Generator) -> tuple[int, list[int], list[int]]:
    sep_color = int(rng.integers(1, 10))
    pool = [c for c in range(1, 10) if c != sep_color]
    rng.shuffle(pool)
    left_size = int(rng.integers(LEFT_PALETTE_MIN, LEFT_PALETTE_MAX + 1))
    right_size = int(rng.integers(RIGHT_PALETTE_MIN, RIGHT_PALETTE_MAX + 1))
    if left_size + right_size > len(pool):
        raise ValueError("not enough colors to split side palettes")
    left_palette = list(map(int, pool[:left_size]))
    right_palette = list(map(int, pool[left_size : left_size + right_size]))
    return sep_color, left_palette, right_palette


def _row_ext_signature(
    width: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int]],
) -> tuple[int, ...]:
    positions = sorted(
        {
            cc
            for length, _ in bars
            for cc in _ext_positions(sep_col, width, source_side, length)
        },
        reverse=(source_side == "right"),
    )
    return tuple(map(int, positions))


def _paint_extension_original(
    out: np.ndarray,
    row: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int]],
) -> None:
    for length, color in bars:
        for cc in _ext_positions(sep_col, out.shape[1], source_side, length):
            if int(out[row, cc]) == 0:
                out[row, cc] = color


def _paint_extension_cycle(
    out: np.ndarray,
    row: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int]],
) -> None:
    if len(bars) <= 1:
        _paint_extension_original(out, row, sep_col, source_side, bars)
        return

    positions = _row_ext_signature(out.shape[1], sep_col, source_side, bars)
    cycle = [int(color) for _, color in bars]
    for idx, cc in enumerate(positions):
        if int(out[row, cc]) == 0:
            out[row, cc] = cycle[idx % len(cycle)]


def _row_variant_signal(
    width: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int]],
) -> bool:
    if len(bars) <= 1:
        return False

    positions = _row_ext_signature(width, sep_col, source_side, bars)
    if len(positions) < 4:
        return False

    cycle = [int(color) for _, color in bars]
    cycle_vals = [cycle[idx % len(cycle)] for idx in range(len(positions))]

    tmp = np.zeros(width, dtype=int)
    for length, color in bars:
        for cc in _ext_positions(sep_col, width, source_side, length):
            if int(tmp[cc]) == 0:
                tmp[cc] = color
    original_vals = [int(tmp[cc]) for cc in positions]
    return cycle_vals != original_vals


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    side_len: int,
    width: int,
    sep_col: int,
    source_side: str,
    *,
    require_signal: bool,
) -> tuple[list[tuple[int, int]], int]:
    for _ in range(100):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = side_len - sep_gap
        if capacity <= 0:
            continue

        want_two = require_signal or (capacity >= 2 and rng.random() < P_TWO_BARS)
        if not want_two or capacity < 2:
            color = int(rng.choice(palette))
            return [(int(rng.integers(1, capacity + 1)), color)], sep_gap

        if rng.random() < P_MATCH_LENGTHS:
            max_len = capacity // 2
            if max_len <= 0:
                continue
            near_len = far_len = int(rng.integers(1, max_len + 1))
        else:
            near_len = int(rng.integers(1, capacity))
            far_len = int(rng.integers(1, capacity - near_len + 1))

        colors = list(map(int, rng.choice(palette, size=2, replace=False)))
        if rng.random() < P_SWAP_COLORS:
            colors.reverse()
        bars = [(near_len, colors[0]), (far_len, colors[1])]
        if require_signal and not _row_variant_signal(width, sep_col, source_side, bars):
            continue
        return bars, sep_gap

    sep_gap = 0
    colors = list(map(int, palette[:2]))
    fallback = [(1, colors[0]), (1, colors[1])]
    return fallback, sep_gap


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    source_side: str,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if source_side == "left":
        cursor = sep_col - sep_gap
        for length, color in bars:
            out[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = color
            cursor += length


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(SIDE_MIN, w - 1 - SIDE_MAX)
    hi = min(SIDE_MAX, w - 1 - SIDE_MIN)
    if lo > hi:
        return None
    left_width = int(rng.integers(lo, hi + 1))
    right_width = w - 1 - left_width
    sep_col = left_width

    sep_color, left_palette, right_palette = _sample_side_palettes(rng)

    max_active = min(ACTIVE_TOTAL_MAX, (h + 1) // 2)
    if max_active < ACTIVE_TOTAL_MIN:
        return None
    active_total = int(rng.integers(ACTIVE_TOTAL_MIN, max_active + 1))
    left_rows_count = int(rng.integers(SIDE_ROWS_MIN, active_total - SIDE_ROWS_MIN + 1))

    active_rows = _choose_active_rows(rng, h, active_total)
    left_rows = set(map(int, rng.choice(active_rows, size=left_rows_count, replace=False)))
    right_rows = [r for r in active_rows if r not in left_rows]
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    left_signal_rows = set(map(int, rng.choice(sorted(left_rows), size=1, replace=False)))
    right_signal_rows = set(map(int, rng.choice(right_rows, size=1, replace=False)))

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = int(rng.choice(inactive_rows))
        out[gap_row, sep_col] = 0

    row_specs: dict[int, tuple[str, list[tuple[int, int]]]] = {}
    left_signal_count = 0
    right_signal_count = 0

    for row in active_rows:
        source_side = "left" if row in left_rows else "right"
        palette = left_palette if source_side == "left" else right_palette
        side_len = left_width if source_side == "left" else right_width
        require_signal = (row in left_signal_rows) or (row in right_signal_rows)
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            side_len,
            w,
            sep_col,
            source_side,
            require_signal=require_signal,
        )
        row_specs[int(row)] = (source_side, bars)
        _place_bars(out, int(row), sep_col, source_side, bars, sep_gap)
        if _row_variant_signal(w, sep_col, source_side, bars):
            if source_side == "left":
                left_signal_count += 1
            else:
                right_signal_count += 1

    if left_signal_count < 1 or right_signal_count < 1:
        return None

    for row, (source_side, bars) in row_specs.items():
        _paint_extension_cycle(out, row, sep_col, source_side, bars)

    inp = out.copy()
    for row, (source_side, _bars) in row_specs.items():
        if source_side == "left":
            inp[row, sep_col + 1 :] = 0
        else:
            inp[row, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not left_rows or not right_rows:
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
