# Combo variant of 1ae2feb7 (index 23)
# combo_spec: D1/2+D9/0
# Name: 水平分隔垂直射线

# Design analysis for the fused variant
# =====================================
# a) Concrete behavior and trace
#    - Base task:
#      one nearly full vertical separator column splits the canvas into a bar side and
#      an empty extension side. Each active row contains 1-2 adjacent horizontal bars.
#      Every bar paints sparse dots horizontally into the extension side, using
#      `bar_length` as the period. Nearer bars are processed first and never overwritten.
#    - Fused task:
#      the separator becomes a horizontal row, but the source bars intentionally stay
#      horizontal rather than rotating into vertical sticks. Therefore a source bar is no
#      longer summarized by a single target row. Instead, every pixel of that horizontal
#      run emits the same vertical periodic ray onto the opposite side. A length-3 bar on
#      columns 4..6 above the separator paints rows `sep+1, sep+4, sep+7, ...` in all of
#      columns 4..6. The visible result is a stack of repeated horizontal segments.
#    - Tiny trace:
#      if rows above the separator contain
#        near row: color A on cols 4..6 (length 3)
#        far  row: color B on cols 5..7 (length 2)
#      then the output below the separator gets
#        row sep+1: A A A on cols 4..6 first, so B can only appear at col 7
#        row sep+3: B B B on cols 5..7 except any cells already claimed earlier
#        row sep+4: A A A again on cols 4..6
#      This shows both fused changes at once: the obstacle is horizontal, while the path
#      is vertical and uses the horizontal bar width as a repeated segment.
#
# b) Complexity calibration
#    - A pure 90-degree rotation would have vertical source bars and single-cell dots in
#      each target row. This combo avoids that by preserving horizontal source runs and by
#      forcing output rows beyond the first extension row to contain horizontal runs of
#      length >= 2.
#    - The generator keeps the source vocabulary small: 2-4 active rows, 1-2 adjacent
#      bars per active row, small bar lengths (mostly 2-5), and one shared overlap band
#      across two different source distances. That is enough to expose the fused rule
#      without creating a noisy dense texture.
#
# c) Hidden-assumption audit against the original main()
#    - `_find_sep_col(...)` is invalid because the separator is no longer vertical.
#      The solver must detect a mostly solid monochrome row instead.
#    - The original side test `left_mass/right_mass` is invalid. With a horizontal
#      separator, the meaningful split is top vs bottom.
#    - `_iter_runs_1d(...)` still remains useful because the source objects are still
#      horizontal runs inside rows. This is the important difference from the single
#      D1/2 or D9/0 rotations, which would have switched to column scanning.
#    - The old `for cc in range(..., step=length)` assumed one target cell per period on
#      the same source row. The fused solver must instead iterate target rows and paint a
#      whole horizontal slice `[c0:c1]` on each hit.
#    - Original "nearer bar first" depended on distance to the separator along columns.
#      Here the distance is row distance to the separator row. Runs on the same source row
#      tie on distance and are processed left-to-right; the generator does not rely on
#      those ties for the core behavior.
#
# d) Placement legality
#    - The separator row is sampled so bar-side height is 3-7 and extension-side height is
#      4-10. This keeps both the source side and the vertical periodic output physically
#      feasible.
#    - Active source rows are spaced by at least one blank row so the input stays legible.
#    - Each source row holds either one bar or two horizontally adjacent bars. Their total
#      width is capped well below the full canvas width, so no source row can masquerade
#      as a second separator row.
#    - Output is built first, then the whole extension side is cleared to obtain the input.
#      This guarantees `main(input)` reconstructs the intended output exactly.
#
# e) Visual observability
#    - Humans can see a long horizontal divider and several horizontal bars on only one
#      side of it. The output adds repeated horizontal segments on the opposite side at
#      regular vertical intervals. That pattern is visibly different from the base task's
#      horizontal dotting and from the pure-rotation vertical-bar variant.
#    - The generator forces at least one overlap witness where a farther source row tries
#      to paint cells already claimed by a nearer row. This makes the nearest-first rule
#      observable from geometry alone rather than from color identities.
#
# f) Visual cleanliness
#    - Only one kind of object appears on the extension side: repeated horizontal slices.
#      Because active rows are few and spaced apart, overlap happens in a controlled band
#      rather than everywhere.
#    - The repeated slices stay aligned to the source bar spans, so the output looks like
#      clean vertical stacks of short horizontal bars rather than an amorphous fill.
#
# g) Change inferability
#    - Colors are copied directly; there is no hidden recoloring or color-index ordering.
#      The only latent variable is bar length, which is already visible as source width and
#      reappears as the vertical step size.
#
# h) Fusion-specific non-degeneracy target
#    - `main(input) == output` alone would not prove the fusion is visible. The generator
#      therefore also requires:
#        * at least one blocked cell caused by different source distances;
#        * at least one non-adjacent extension row containing a horizontal run of length >= 2.
#      Those two metrics certify that the sample really exhibits "horizontal bars emitting
#      vertical repeated segments" rather than collapsing into a weak or rotated-looking case.
#
# i) New solver assumptions guaranteed by the generator
#    - Exactly one separator-like row exists.
#    - All source bars lie strictly on one side of that row, and the opposite side is empty
#      in the input.
#    - Source objects are horizontal monochrome runs, optionally with a second adjacent run
#      on the same row.
#    - At least two source distances appear, and at least one overlap witness exists, so the
#      global row-distance ordering in `main()` has visible consequences.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_row(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        vals = row[row != bg]
        if vals.size >= w - 1 and (vals == vals[0]).all():
            return r, int(vals[0])
    return None, None


def _iter_runs_1d(line: np.ndarray, i_lo: int, i_hi: int, bg: int):
    i = i_lo
    while i < i_hi:
        if int(line[i]) == bg:
            i += 1
            continue
        color = int(line[i])
        i0 = i
        while i < i_hi and int(line[i]) == color:
            i += 1
        yield i0, i - 1, color


def _collect_bars(
    g: np.ndarray,
    sep_row: int,
    bars_on_top: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    bars: list[tuple[int, int, int, int, int, int]] = []
    h, _w = g.shape
    if bars_on_top:
        rows = range(0, sep_row)
    else:
        rows = range(sep_row + 1, h)

    for r in rows:
        for c0, c1, color in _iter_runs_1d(g[r, :], 0, g.shape[1], bg):
            length = c1 - c0 + 1
            dist = (sep_row - 1 - r) if bars_on_top else (r - (sep_row + 1))
            bars.append((dist, r, c0, c1, length, color))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _paint_bar_vertical_band(
    out: np.ndarray,
    sep_row: int,
    bars_on_top: bool,
    c0: int,
    c1: int,
    length: int,
    color: int,
    bg: int,
) -> tuple[int, int]:
    """Paint one bar's vertical periodic band; return (painted_cells, blocked_cells)."""
    if length <= 0:
        return 0, 0

    h = out.shape[0]
    if bars_on_top:
        rows = range(sep_row + 1, h, length)
    else:
        rows = range(sep_row - 1, -1, -length)

    painted = 0
    blocked = 0
    for rr in rows:
        segment = out[rr, c0 : c1 + 1]
        blocked += int(np.sum(segment != bg))
        mask = segment == bg
        if np.any(mask):
            segment[mask] = color
            painted += int(np.sum(mask))
    return painted, blocked


# Overview:
#   Detect a horizontal separator row, keep the source bars horizontal, and extend each
#   bar vertically as repeated copies of the whole run onto the opposite side.
# Input structure:
#   The grid contains one almost-solid monochrome separator row. All source objects are
#   horizontal monochrome runs on either the top side or the bottom side of that row, and
#   the opposite side is blank in the input.
# Rule:
#   Process source runs from nearest to farthest relative to the separator row. A run of
#   length L spanning columns c0..c1 paints rows immediately across the separator and then
#   every L rows farther away, always across the entire span c0..c1. Existing non-bg cells
#   are never overwritten.
# Key insight:
#   The separator orientation changes, but the source object orientation does not. The
#   solver must therefore keep row-wise run extraction and only rotate the propagation
#   direction, producing vertical stacks of horizontal segments rather than single dots.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _bg_color(g)

    sep_row, _sep_color = _find_sep_row(g, bg)
    if sep_row is None:
        return g.copy()

    out = g.copy()
    top_mass = int(np.sum(g[:sep_row, :] != bg)) if sep_row > 0 else 0
    bottom_mass = int(np.sum(g[sep_row + 1 :, :] != bg)) if sep_row + 1 < h else 0
    bars_on_top = top_mass >= bottom_mass

    for _dist, _r, c0, c1, length, color in _collect_bars(g, sep_row, bars_on_top, bg):
        _paint_bar_vertical_band(out, sep_row, bars_on_top, c0, c1, length, color, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 10, 16
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 7
EXT_SIDE_MIN, EXT_SIDE_MAX = 4, 10

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 4

MIN_BAR_LEN = 2
MAX_BAR_LEN = 5
MAX_TOTAL_ROW_WIDTH = 8

P_TWO_BARS = 0.4
P_SEP_ROW_GAP = 0.15

MIN_BLOCKED_CELLS = 2
MIN_FAR_RUN_CELLS = 4
MAX_ATTEMPTS = 500


def _choose_spaced_indices(rng: np.random.Generator, size: int, count: int) -> list[int]:
    """Pick `count` indices in [0, size) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(size - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_row_runs(
    rng: np.random.Generator,
    w: int,
    ext_side_len: int,
    palette: list[int],
    primary: int,
    *,
    forced: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    """
    Return row runs as (c0, c1, color).

    If `forced` is given, it fixes the first run's `(start, length)` and may still add one
    adjacent second run when space allows.
    """
    max_len = min(MAX_BAR_LEN, ext_side_len)
    if max_len < MIN_BAR_LEN:
        raise ValueError("extension side too short for visible vertical repetition")

    runs: list[tuple[int, int, int]] = []
    if forced is not None:
        c0, length = forced
        c1 = c0 + length - 1
        runs.append((int(c0), int(c1), int(primary)))

        can_add = (
            rng.random() < 0.3
            and length + MIN_BAR_LEN <= MAX_TOTAL_ROW_WIDTH
            and c1 + MIN_BAR_LEN < w - 1
        )
        if can_add:
            extra_len = int(rng.integers(MIN_BAR_LEN, min(max_len, MAX_TOTAL_ROW_WIDTH - length) + 1))
            other_choices = [c for c in palette if c != primary]
            runs.append((c1 + 1, c1 + extra_len, int(rng.choice(other_choices))))
        return runs

    two_bars = bool(rng.random() < P_TWO_BARS)
    if not two_bars:
        length = int(rng.integers(MIN_BAR_LEN, max_len + 1))
        c0 = int(rng.integers(1, w - length))
        return [(c0, c0 + length - 1, int(primary))]

    len1 = int(rng.integers(MIN_BAR_LEN, max_len + 1))
    max_len2 = min(max_len, MAX_TOTAL_ROW_WIDTH - len1)
    if max_len2 < MIN_BAR_LEN:
        c0 = int(rng.integers(1, w - len1))
        return [(c0, c0 + len1 - 1, int(primary))]

    len2 = int(rng.integers(MIN_BAR_LEN, max_len2 + 1))
    total = len1 + len2
    c0 = int(rng.integers(1, w - total))
    other_choices = [c for c in palette if c != primary]
    color2 = int(rng.choice(other_choices))
    return [
        (c0, c0 + len1 - 1, int(primary)),
        (c0 + len1, c0 + total - 1, color2),
    ]


def _paint_source_rows(
    out: np.ndarray,
    row_runs: dict[int, list[tuple[int, int, int]]],
) -> None:
    for row, runs in row_runs.items():
        for c0, c1, color in runs:
            out[row, c0 : c1 + 1] = color


def _paint_all_extensions(
    out: np.ndarray,
    row_runs: dict[int, list[tuple[int, int, int]]],
    sep_row: int,
    bars_on_top: bool,
) -> tuple[int, int]:
    bg = 0
    blocked = 0
    painted = 0
    bars: list[tuple[int, int, int, int, int, int]] = []
    for row, runs in row_runs.items():
        dist = (sep_row - 1 - row) if bars_on_top else (row - (sep_row + 1))
        for c0, c1, color in runs:
            length = c1 - c0 + 1
            bars.append((dist, row, c0, c1, length, color))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _dist, _row, c0, c1, length, color in bars:
        add_painted, add_blocked = _paint_bar_vertical_band(
            out,
            sep_row,
            bars_on_top,
            c0,
            c1,
            length,
            color,
            bg,
        )
        painted += add_painted
        blocked += add_blocked
    return painted, blocked


def _ext_view(out: np.ndarray, sep_row: int, bars_on_top: bool) -> np.ndarray:
    if bars_on_top:
        return out[sep_row + 1 :, :]
    return out[:sep_row, :]


def _far_horizontal_run_stats(out: np.ndarray, sep_row: int, bars_on_top: bool) -> tuple[int, int, int]:
    """
    Quantify visible fused behavior beyond the first extension row.
    Returns (far_cells_in_runs, far_rows_with_run, max_far_run_len).
    """
    ext = _ext_view(out, sep_row, bars_on_top)
    if ext.shape[0] <= 1:
        return 0, 0, 0

    far = ext[1:, :]
    total_cells = 0
    run_rows = 0
    max_run = 0
    for row in far:
        row_has_run = False
        c = 0
        while c < row.size:
            if int(row[c]) == 0:
                c += 1
                continue
            color = int(row[c])
            c0 = c
            while c < row.size and int(row[c]) == color:
                c += 1
            run_len = c - c0
            if run_len >= 2:
                total_cells += run_len
                row_has_run = True
                max_run = max(max_run, run_len)
        if row_has_run:
            run_rows += 1
    return total_cells, run_rows, max_run


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, h - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, h - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = h - 1 - bar_side_len

    max_active = min(ACTIVE_MAX, (bar_side_len + 1) // 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))

    bars_on_top = bool(rng.integers(0, 2) == 0)
    sep_row = bar_side_len if bars_on_top else (h - 1 - bar_side_len)
    local_rows = _choose_spaced_indices(rng, bar_side_len, active_count)
    active_rows = local_rows if bars_on_top else [sep_row + 1 + idx for idx in local_rows]
    active_rows = list(map(int, active_rows))

    sep_color, palette = _sample_palette(rng)
    out = np.zeros((h, w), dtype=int)
    out[sep_row, :] = sep_color

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_runs: dict[int, list[tuple[int, int, int]]] = {}

    # Force two witness rows with overlapping spans at different distances.
    sorted_rows = sorted(
        active_rows,
        key=lambda r: (sep_row - 1 - r) if bars_on_top else (r - (sep_row + 1)),
    )
    near_row = int(sorted_rows[0])
    far_row = int(sorted_rows[-1])

    max_show_len = min(MAX_BAR_LEN, ext_side_len - 1, w - 3)
    if max_show_len < MIN_BAR_LEN:
        return None

    near_len = int(rng.integers(MIN_BAR_LEN, max_show_len + 1))
    far_len_choices = [l for l in range(MIN_BAR_LEN, max_show_len + 1) if l != near_len]
    if not far_len_choices:
        far_len_choices = [near_len]
    far_len = int(rng.choice(far_len_choices))
    base_c0 = int(rng.integers(1, w - max(near_len, far_len) - 1))
    far_c0 = min(base_c0 + int(rng.integers(0, 2)), w - far_len - 1)
    overlap = min(base_c0 + near_len - 1, far_c0 + far_len - 1) - max(base_c0, far_c0) + 1
    if overlap < 2:
        return None

    near_color = int(row_primary[active_rows.index(near_row)])
    far_color_choices = [c for c in palette if c != near_color]
    far_color = int(rng.choice(far_color_choices))
    row_runs[near_row] = _sample_row_runs(
        rng,
        w,
        ext_side_len,
        palette,
        near_color,
        forced=(base_c0, near_len),
    )
    row_runs[far_row] = _sample_row_runs(
        rng,
        w,
        ext_side_len,
        palette,
        far_color,
        forced=(far_c0, far_len),
    )

    two_bar_rows = int(len(row_runs[near_row]) == 2) + int(len(row_runs[far_row]) == 2)
    for row, primary in zip(active_rows, row_primary):
        if row in row_runs:
            continue
        runs = _sample_row_runs(rng, w, ext_side_len, palette, int(primary))
        row_runs[int(row)] = runs
        if len(runs) == 2:
            two_bar_rows += 1

    _paint_source_rows(out, row_runs)

    all_used_cols = {c for runs in row_runs.values() for c0, c1, _color in runs for c in range(c0, c1 + 1)}
    if rng.random() < P_SEP_ROW_GAP:
        gap_choices = [c for c in range(w) if c not in all_used_cols]
        if gap_choices:
            out[sep_row, int(rng.choice(gap_choices))] = 0

    _painted, blocked = _paint_all_extensions(out, row_runs, sep_row, bars_on_top)
    far_run_cells, far_run_rows, max_far_run = _far_horizontal_run_stats(out, sep_row, bars_on_top)

    if blocked < MIN_BLOCKED_CELLS:
        return None
    if far_run_cells < MIN_FAR_RUN_CELLS or far_run_rows < 1 or max_far_run < 2:
        return None
    if two_bar_rows == 0 and active_count < 3:
        return None

    inp = out.copy()
    if bars_on_top:
        inp[sep_row + 1 :, :] = 0
    else:
        inp[:sep_row, :] = 0

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
