# Combo variant of 1ae2feb7 (index 16)
# combo_spec: D2/3+D11/2
# Name: 交错平铺完整图案

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Geometry is unchanged from 1ae2feb7: one nearly-full vertical separator column,
#      2-5 active rows, and 1-2 adjacent monochrome bars per active row on one side.
#    - Original behavior: each bar is handled independently and paints sparse dots onto
#      ext_side at offsets 0, L, 2L, ... where L is that bar's own length.
#    - D2/3 changes the unit of repetition: the ext-side motion is no longer "one dot per
#      bar step", but "repeat a full row-level tile" whose period is the total bar length.
#    - D11/2 changes the phase inside that repeated tile: in a two-bar row, the nearest
#      bar still starts at offset 0 inside each period, while the second bar starts at
#      floor(its_length / 2). Bars are still applied nearest-first and never overwrite.
#    - Example on a left-bar row with bars near->far = [(3, A), (4, B)]:
#      original output: A at 0,3,6,... then B at 0,4,8,... filling only blanks;
#      plain D2 tiling: [A A A B B B B][A A A B B B B]...
#      combo tiling: A fills 0..2 in every 7-cell period, B starts at offset 2 and fills
#      only blank cells, so the period becomes [A A A B B B 0] and then repeats.
#      The solver must therefore reason about a repeated period plus a phase shift, not
#      about independent single-bar trajectories.
#
# b) Complexity calibration
#    - Single-bar rows would show only the D2/3 effect, so the generator forces at least
#      one two-bar row whose second bar has length >= 2 and whose shifted phase produces a
#      different output from both the original task and from plain contiguous tiling.
#    - The period is capped to at most about half of ext_side, so repeated structure is
#      visible at least twice in most samples rather than appearing as one truncated copy.
#    - Two-bar rows use a small, structured family of lengths that creates a consistent
#      one-cell overlap-and-tail-gap effect. This keeps the fusion visible without making
#      the row look chaotic.
#
# c) Implicit-assumption audit against the original main()
#    - `_find_sep_col(...)` still works because separator geometry is unchanged.
#    - The left/right mass heuristic still works because generated inputs keep ext_side
#      empty; only the true bar side contributes non-background mass.
#    - The original global `bars.sort(...)` + `step = length` logic no longer matches the
#      task. Repetition is now per row with a shared period equal to the row's total bar
#      length, so the solver must group runs by row before painting.
#    - Original code assumed every bar starts at ext offset 0. The combo variant breaks
#      that assumption for the second local bar, so the solver must track row-local bar
#      order and compute a phase offset inside the repeated period.
#    - Original no-overwrite behavior still matters: the shifted second bar can collide
#      with the first bar inside a period, and the combo rule explicitly keeps the first
#      bar's claim on overlapping cells.
#
# d) Placement legality
#    - Bars remain contiguous on the bar side and fit within `bar_side_len`, optionally
#      leaving a 1-cell gap next to the separator just like the base task.
#    - The forced two-bar family is sampled only when both the bar side and ext side can
#      support a visible repeated period. This avoids impossible geometry and weak samples.
#    - Generation is output-first, then ext_side is cleared to form the input, and the
#      sample is accepted only if `main(input)` reconstructs the output exactly.
#
# e) Visual observability
#    - The repeated unit is visible directly as a horizontally repeating block pattern.
#      The D11 shift is also visible because the two-bar row no longer matches a simple
#      concatenation of source bars.
#    - No hidden property such as raw color id is used for ordering. Only visible spatial
#      facts matter: separator side, row, bar order by proximity, and bar length.
#
# f) Visual cleanliness
#    - Active rows are separated by blank rows, so each row's repeated tile can be parsed
#      independently by a human.
#    - The overlap pattern is constrained and consistent. Outputs can contain periodic tail
#      blanks, but only in a regular way tied to the shifted second bar, not as noise.
#
# g) Change inferability
#    - Colors are copied, not remapped. The only inference burden is spatial: identify the
#      per-row source bars, infer the shared period, and notice that the second bar starts
#      half a bar-length later inside each period.
#
# New solver assumptions introduced by the combo
#    - Within any active row, runs on the bar side are the meaningful source bars ordered
#      by distance to the separator; the generator guarantees this by placing them from the
#      separator outward with no internal gaps.
#    - The repeated period is the sum of source-bar lengths, not the entire side width.
#    - The second local bar must have a non-zero half-length offset in at least one row so
#      the D11 contribution is visible; the generator enforces and verifies that.


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


def _row_bars(
    row: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int]]:
    if bars_on_left:
        runs = list(_iter_runs_1d(row, 0, sep_col, bg))
        runs.sort(key=lambda t: ((sep_col - 1) - t[1], t[0]))
    else:
        runs = list(_iter_runs_1d(row, sep_col + 1, len(row), bg))
        runs.sort(key=lambda t: (t[0] - (sep_col + 1), t[0]))
    return [(c1 - c0 + 1, int(col)) for c0, c1, col in runs]


def _bar_phase_offset(length: int, bar_index: int) -> int:
    return 0 if bar_index % 2 == 0 else length // 2


def _ext_len(w: int, sep_col: int, bars_on_left: bool) -> int:
    return w - sep_col - 1 if bars_on_left else sep_col


def _ext_col(sep_col: int, bars_on_left: bool, ext_pos: int) -> int:
    return sep_col + 1 + ext_pos if bars_on_left else sep_col - 1 - ext_pos


def _paint_combo_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
) -> None:
    period = sum(length for length, _ in bars)
    if period <= 0:
        return

    ext_len = _ext_len(out.shape[1], sep_col, bars_on_left)
    for bar_index, (length, color) in enumerate(bars):
        start = _bar_phase_offset(length, bar_index)
        for base in range(start, ext_len, period):
            stop = min(base + length, ext_len)
            for ext_pos in range(base, stop):
                cc = _ext_col(sep_col, bars_on_left, ext_pos)
                if int(out[row, cc]) == bg:
                    out[row, cc] = int(color)


def _paint_plain_tiling_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    pattern: list[int] = []
    for length, color in bars:
        pattern.extend([int(color)] * int(length))
    if not pattern:
        return

    ext_len = _ext_len(out.shape[1], sep_col, bars_on_left)
    period = len(pattern)
    for ext_pos in range(ext_len):
        cc = _ext_col(sep_col, bars_on_left, ext_pos)
        out[row, cc] = int(pattern[ext_pos % period])


def _paint_original_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
) -> None:
    ext_len = _ext_len(out.shape[1], sep_col, bars_on_left)
    for length, color in bars:
        for ext_pos in range(0, ext_len, length):
            cc = _ext_col(sep_col, bars_on_left, ext_pos)
            if int(out[row, cc]) == bg:
                out[row, cc] = int(color)


def _ext_diff_count(
    a: np.ndarray,
    b: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
) -> int:
    if bars_on_left:
        return int(np.sum(a[:, sep_col + 1 :] != b[:, sep_col + 1 :]))
    return int(np.sum(a[:, :sep_col] != b[:, :sep_col]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Copy each active row's bar pattern onto the opposite side as a repeated tile, but
        phase-shift the second local bar so the repeated unit becomes an interleaved motif.

    Input structure:
        The grid contains one almost-solid vertical separator column. All source bars lie
        on the same side of that separator, with 1-2 monochrome horizontal runs per active
        row and blank space on the extension side.

    Rule:
        For each row independently, collect its bars from nearest to farthest from the
        separator. Let the period be the sum of their lengths. Repeat this period across
        ext_side. Inside each period, bar 0 starts at offset 0, bar 1 starts at offset
        floor(length / 2), bar 2 would restart the alternation, and so on. Paint bars in
        that order and never overwrite an already-painted ext cell.

    Key insight:
        The right abstraction is no longer "each run emits dots every L cells". Instead,
        each row defines a repeated period plus an internal phase schedule.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    for r in range(g.shape[0]):
        bars = _row_bars(g[r, :], sep_col, bars_on_left, bg)
        _paint_combo_row(out, r, sep_col, bars_on_left, bars, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 4, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 8, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.65
P_SEP_GAP = 0.12
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300
MIN_COMBO_VS_PLAIN = 4
MIN_COMBO_VS_ORIG = 8


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _choose_forced_two_rows(
    rng: np.random.Generator,
    active_rows: list[int],
) -> set[int]:
    quota = 1 if len(active_rows) <= 2 else 2
    chosen = rng.choice(active_rows, size=quota, replace=False)
    return set(map(int, np.atleast_1d(chosen)))


def _structured_two_bar_options(max_total: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for far_len in range(2, max_total):
        near_len = far_len // 2 + 1
        total = near_len + far_len
        if total <= max_total and near_len != far_len // 2:
            options.append((near_len, far_len))
    return options


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    *,
    force_two_bars: bool,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    max_total = min(capacity, max(4, ext_side_len // 2))

    if force_two_bars and max_total < 4:
        sep_gap = 0
        capacity = bar_side_len
        max_total = min(capacity, max(4, ext_side_len // 2))

    if max_total < 2:
        raise ValueError("row pattern capacity too small")

    two_bar_options = _structured_two_bar_options(max_total)
    can_use_two = bool(two_bar_options)
    use_two = force_two_bars or (can_use_two and rng.random() < P_TWO_BARS)

    if not use_two:
        length = int(rng.integers(2, max_total + 1))
        return [(length, primary)], sep_gap

    near_len, far_len = two_bar_options[int(rng.integers(0, len(two_bar_options)))]
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


def _combo_metrics(
    inp: np.ndarray,
    row_bars_map: dict[int, list[tuple[int, int]]],
    sep_col: int,
    bars_on_left: bool,
) -> tuple[int, int]:
    combo_out = inp.copy()
    plain_out = inp.copy()
    orig_out = inp.copy()

    for row, bars in row_bars_map.items():
        _paint_combo_row(combo_out, row, sep_col, bars_on_left, bars, bg=0)
        _paint_plain_tiling_row(plain_out, row, sep_col, bars_on_left, bars)
        _paint_original_row(orig_out, row, sep_col, bars_on_left, bars, bg=0)

    return (
        _ext_diff_count(combo_out, plain_out, sep_col, bars_on_left),
        _ext_diff_count(combo_out, orig_out, sep_col, bars_on_left),
    )


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
    forced_two_rows = _choose_forced_two_rows(rng, active_rows)
    row_bars_map: dict[int, list[tuple[int, int]]] = {}

    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            force_two_bars=(row in forced_two_rows),
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_combo_row(out, row, sep_col, bars_on_left, bars, bg=0)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not any(len(bars) == 2 and bars[1][0] >= 2 for bars in row_bars_map.values()):
        return None
    if sum(len(bars) == 2 for bars in row_bars_map.values()) < len(forced_two_rows):
        return None

    combo_vs_plain, combo_vs_orig = _combo_metrics(inp, row_bars_map, sep_col, bars_on_left)
    if combo_vs_plain < MIN_COMBO_VS_PLAIN or combo_vs_orig < MIN_COMBO_VS_ORIG:
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
