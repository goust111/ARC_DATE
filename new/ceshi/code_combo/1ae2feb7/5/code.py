# Combo variant of 1ae2feb7 (index 5)
# combo_spec: D7/0+D15/2
# Name: 远优先处理冲突留白

from __future__ import annotations

import math

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    The preserved scaffold is the original 1ae2feb7 layout: one near-solid vertical separator,
#    2-5 active rows, and each active row has horizontal monochrome bars on exactly one side.
#    The opposite side is blank in the input and becomes the periodic projection area in the
#    output. The combo changes the projection semantics in two coupled places:
#    - order changes from near->far to far->near
#    - the old "first paint sticks" rule changes to "if a later ray hits an occupied extension
#      cell, erase that cell back to background"
#    A concrete row example, bars on the left, extension width 8, bars listed near->far as
#    [(2, A), (3, B), (6, C)]:
#    - original task: A paints offsets 0,2,4,6; B then only fills still-zero 3; C fills nothing
#      on overlaps, so the row starts A 0 A B A 0 A 0.
#    - D15 alone (simultaneous unique claims): offsets 0 and 6 are multi-claimed and stay 0,
#      giving 0 0 A B A 0 0 0.
#    - this combo: process C, then B, then A with erase-on-collision:
#        offset 0: C paints, B erases, A paints again -> A
#        offset 3: C paints, B erases? no, C also at 6 only; B paints -> B
#        offset 6: C paints, B erases, A paints again -> A
#      So odd-count overlap cells become the nearest later color, while even-count overlaps end
#      as background. This is the smallest readable setting where both dimensions matter.
# b) Complexity calibration
#    If we kept the source task's strict 1-2 bars per row, then "far-first" would be visually
#    swallowed by "conflict erases" because every overlap would end in the same blank cell no
#    matter which of the two bars arrived first. To keep the combo non-degenerate, the generator
#    forces exactly one 3-bar witness row. Other active rows stay at 1-2 bars, and one extra
#    2-bar conflict row is usually included so the blanking rule is still visible in a simpler
#    setting. This keeps the task close to the source while making both control variables legible.
# c) Hidden-assumption audit on the original main()
#    - `_find_sep_col` assumed one mostly solid monochrome separator column. That still holds.
#    - `bars_on_left` inferred the bar side from non-background mass. That still holds because the
#      generator still zeroes the extension side in the input.
#    - `_iter_runs_1d` assumed bars are contiguous same-color runs. That still holds; the combo
#      only adds a possible third adjacent run on the same row.
#    - The original global `bars.sort(key=lambda t: (t[0], ...))` encoded near-first priority.
#      The combo invalidates that ordering; we now sort each row far-first.
#    - The original `if out[r, cc] == bg: out[r, cc] = col` encoded "first claimant wins".
#      The combo invalidates that assumption too; a later claimant must erase the prior paint
#      instead of skipping it.
#    - New solver assumption: valid inputs keep the extension side background before simulation.
#      The generator preserves that by constructing the output first and then clearing the whole
#      extension side to form the input.
# d) Placement legality
#    Bars remain on exactly one side of the separator and are stacked contiguously from the
#    separator outward, with optional separator gap only on regular rows. The witness/support
#    rows use gap=0 so their interaction pattern is easier to read and always fits. Length tuples
#    are chosen with total length <= available bar-side width, so bars never overlap or cross the
#    separator. Active rows remain separated by at least one empty row, so row-local ray logic
#    never interferes across rows.
# e) Visual observability
#    The generator rejects samples unless two measurable effects are present on the same input:
#    - `blank_from_conflicts > 0`: combo output has background cells where far-first/no-overwrite
#      would have kept a color. This exposes the D15-style erasure rule.
#    - `revived_after_odd_conflicts > 0`: combo output has colored cells where simultaneous
#      conflict-canceling would stay blank. This exposes that order still matters in the combo,
#      hence D7 is not redundant.
#    These metrics depend only on visible geometry and colors, never on raw color ids.
# f) Visual cleanliness
#    Only one 3-bar witness row is forced; other rows are simpler. Rows are spaced apart and the
#    separator still partitions the image into bar-side and extension-side regions. Because each
#    row is solved independently, the added complexity stays localized rather than producing a
#    tangled full-grid overlay.
# g) Change inferability
#    Colors are not remapped. A projected pixel always uses the color of the bar that most
#    recently painted it, and blank cells remain background. The human inference burden is thus
#    about row geometry and collision order, not about hidden color correspondences.
# 9) New assumptions introduced by this combo
#    The solver now supports 1-3 adjacent bars on an active row. The generator guarantees that
#    the only rows with 3 bars are still horizontally contiguous and visually separable, and it
#    enforces a witness triple whose combo behavior differs from the base task, D7-only, and
#    D15-only outputs. That closes the main degeneracy introduced by combining the two dimensions.


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


def _extension_positions(w: int, sep_col: int, bars_on_left: bool, step: int):
    if bars_on_left:
        return range(sep_col + 1, w, step)
    return range(sep_col - 1, -1, -step)


def _collect_row_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> dict[int, list[tuple[int, int, int, int]]]:
    h, w = g.shape
    row_bars: dict[int, list[tuple[int, int, int, int]]] = {}
    for r in range(h):
        if bars_on_left:
            runs = _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                row_bars.setdefault(r, []).append((dist, c0, length, col))
        else:
            runs = _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                row_bars.setdefault(r, []).append((dist, c0, length, col))
    return row_bars


def _row_order(
    bars: list[tuple[int, int, int, int]],
    *,
    far_first: bool,
) -> list[tuple[int, int, int, int]]:
    if far_first:
        return sorted(bars, key=lambda t: (-t[0], t[1]))
    return sorted(bars, key=lambda t: (t[0], t[1]))


def _simulate(
    grid: np.ndarray,
    *,
    far_first: bool,
    mode: str,
) -> np.ndarray:
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

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    for r, bars in row_bars.items():
        ordered = _row_order(bars, far_first=far_first)
        if mode == "unique":
            claims: dict[int, int] = {}
            claim_color: dict[int, int] = {}
            for _dist, _c0, length, col in ordered:
                if length <= 0:
                    continue
                for cc in _extension_positions(w, sep_col, bars_on_left, length):
                    claims[cc] = claims.get(cc, 0) + 1
                    claim_color.setdefault(cc, col)
            for cc, count in claims.items():
                out[r, cc] = claim_color[cc] if count == 1 else bg
            continue

        for _dist, _c0, length, col in ordered:
            if length <= 0:
                continue
            for cc in _extension_positions(w, sep_col, bars_on_left, length):
                if mode == "no_overwrite":
                    if int(out[r, cc]) == bg:
                        out[r, cc] = col
                elif mode == "erase":
                    if int(out[r, cc]) == bg:
                        out[r, cc] = col
                    else:
                        out[r, cc] = bg
                else:
                    raise ValueError(f"unknown mode: {mode}")
    return out


def _solve_base(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, far_first=False, mode="no_overwrite")


def _solve_d7(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, far_first=True, mode="no_overwrite")


def _solve_d15(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, far_first=False, mode="unique")


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, far_first=True, mode="erase")


def _ext_side_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return np.zeros_like(g, dtype=bool)
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass
    mask = np.zeros_like(g, dtype=bool)
    if bars_on_left:
        mask[:, sep_col + 1 :] = True
    else:
        mask[:, :sep_col] = True
    return mask


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = _solve_combo(grid)
    d7 = _solve_d7(grid)
    d15 = _solve_d15(grid)
    base = _solve_base(grid)
    mask = _ext_side_mask(grid)
    return {
        "diff_vs_base": int(np.count_nonzero((combo != base) & mask)),
        "diff_vs_d7": int(np.count_nonzero((combo != d7) & mask)),
        "diff_vs_d15": int(np.count_nonzero((combo != d15) & mask)),
        "blank_from_conflicts": int(np.count_nonzero((combo == 0) & (d7 != 0) & mask)),
        "revived_after_odd_conflicts": int(
            np.count_nonzero((combo != 0) & (d15 == 0) & mask)
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Extend each active row across the separator using periodic dots derived from that row's
        bar lengths, but with farthest bars processed first and collisions erasing prior paint.

    Input structure:
        A mostly solid vertical separator column splits the grid into a bar side and an empty
        extension side. Active rows contain 1-3 adjacent monochrome horizontal bars on one side;
        non-active rows are background except for the separator.

    Rule:
        For each active row, sort its bars by distance from the separator from far to near.
        Each bar projects onto the opposite side starting at the first cell past the separator
        and repeating every `bar_length` cells. If a projected cell is background, paint it with
        the bar's color. If it is already painted by a farther bar, erase it back to background.

    Key insight:
        The output is neither the original near-first no-overwrite pattern nor the simultaneous
        hole-making D15 rule: because erasure happens sequentially in far->near order, odd-count
        overlaps can become colored again by the nearest later bar while even-count overlaps stay
        blank.
    """
    return _solve_combo(grid)


# --------------- generator for the combo variant ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.25
P_SEP_GAP = 0.12
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _pick_colors(
    rng: np.random.Generator,
    palette: list[int],
    count: int,
) -> list[int]:
    return list(map(int, rng.choice(palette, size=count, replace=False)))


def _row_pattern(
    ext_side_len: int,
    bars: list[tuple[int, int]],
    *,
    far_first: bool,
    mode: str,
) -> np.ndarray:
    row = np.zeros(ext_side_len, dtype=int)
    ordered = list(reversed(bars)) if far_first else list(bars)
    if mode == "unique":
        claims = np.zeros(ext_side_len, dtype=int)
        claim_color = np.zeros(ext_side_len, dtype=int)
        for length, color in ordered:
            for off in range(0, ext_side_len, length):
                claims[off] += 1
                if claim_color[off] == 0:
                    claim_color[off] = color
        row[claims == 1] = claim_color[claims == 1]
        return row

    for length, color in ordered:
        for off in range(0, ext_side_len, length):
            if mode == "no_overwrite":
                if row[off] == 0:
                    row[off] = color
            elif mode == "erase":
                if row[off] == 0:
                    row[off] = color
                else:
                    row[off] = 0
            else:
                raise ValueError(f"unknown mode: {mode}")
    return row


def _best_witness_triple(
    capacity: int,
    ext_side_len: int,
) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, tuple[int, int, int]]] = []
    for near_len in range(1, capacity - 1):
        for mid_len in range(1, capacity - near_len):
            for far_len in range(1, capacity - near_len - mid_len + 1):
                bars = [(near_len, 1), (mid_len, 2), (far_len, 3)]
                combo = _row_pattern(ext_side_len, bars, far_first=True, mode="erase")
                d7 = _row_pattern(ext_side_len, bars, far_first=True, mode="no_overwrite")
                d15 = _row_pattern(ext_side_len, bars, far_first=False, mode="unique")
                base = _row_pattern(ext_side_len, bars, far_first=False, mode="no_overwrite")
                blank_from_conflicts = int(np.count_nonzero((combo == 0) & (d7 != 0)))
                revived_after_odd = int(np.count_nonzero((combo != 0) & (d15 == 0)))
                diff_vs_base = int(np.count_nonzero(combo != base))
                if blank_from_conflicts == 0 or revived_after_odd == 0 or diff_vs_base == 0:
                    continue
                score = (
                    20 * revived_after_odd
                    + 8 * blank_from_conflicts
                    + 4 * diff_vs_base
                    + 2 * int(min(near_len, mid_len, far_len) >= 2)
                    + int(len({near_len, mid_len, far_len}) == 3)
                )
                candidates.append((score, (near_len, mid_len, far_len)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], sum(item[1]), min(item[1])))
    return candidates[-1][1]


def _conflict_pair_choices(capacity: int, ext_side_len: int) -> list[tuple[int, int]]:
    pairs = [
        (near_len, far_len)
        for near_len in range(1, capacity)
        for far_len in range(1, capacity - near_len + 1)
    ]
    good = [
        pair
        for pair in pairs
        if math.lcm(pair[0], pair[1]) <= max(1, ext_side_len - 1)
        and pair[0] != pair[1]
    ]
    return good if good else [pair for pair in pairs if pair[0] != pair[1]]


def _sample_regular_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar-side capacity must stay positive")

    want_two = capacity >= 2 and rng.random() < P_TWO_BARS
    if not want_two:
        color = _pick_colors(rng, palette, 1)[0]
        length = int(rng.integers(1, capacity + 1))
        return [(length, color)], sep_gap

    cleaner_pairs = [
        (near_len, far_len)
        for near_len in range(2, capacity)
        for far_len in range(2, capacity - near_len + 1)
    ]
    if cleaner_pairs:
        near_len, far_len = cleaner_pairs[int(rng.integers(0, len(cleaner_pairs)))]
    else:
        near_len = int(rng.integers(1, capacity))
        far_len = int(rng.integers(1, capacity - near_len + 1))
    colors = _pick_colors(rng, palette, 2)
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])], sep_gap


def _sample_support_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    capacity: int,
    ext_side_len: int,
) -> list[tuple[int, int]]:
    pairs = _conflict_pair_choices(capacity, ext_side_len)
    if not pairs:
        raise ValueError("no legal conflict-support pair")
    near_len, far_len = pairs[int(rng.integers(0, len(pairs)))]
    colors = _pick_colors(rng, palette, 2)
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])]


def _sample_witness_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    capacity: int,
    ext_side_len: int,
) -> list[tuple[int, int]]:
    triple = _best_witness_triple(capacity, ext_side_len)
    if triple is None:
        raise ValueError("no legal combo witness triple")
    colors = _pick_colors(rng, palette, 3)
    return [
        (triple[0], colors[0]),
        (triple[1], colors[1]),
        (triple[2], colors[2]),
    ]


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


def _paint_combo_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in reversed(bars):
        for cc in _extension_positions(w, sep_col, bars_on_left, length):
            if out[row, cc] == 0:
                out[row, cc] = color
            else:
                out[row, cc] = 0


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None

    valid_bar_side_lens = [
        bar_side_len
        for bar_side_len in range(lo, hi + 1)
        if _best_witness_triple(bar_side_len, w - 1 - bar_side_len) is not None
    ]
    if not valid_bar_side_lens:
        return None
    bar_side_len = int(valid_bar_side_lens[int(rng.integers(0, len(valid_bar_side_lens)))])
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

    witness_row = active_rows[0]
    support_row = active_rows[1] if len(active_rows) > 1 else active_rows[0]

    row_specs: dict[int, tuple[list[tuple[int, int]], int]] = {}
    try:
        row_specs[witness_row] = (
            _sample_witness_row_bars(rng, palette, bar_side_len, ext_side_len),
            0,
        )
        row_specs[support_row] = (
            _sample_support_row_bars(rng, palette, bar_side_len, ext_side_len),
            0,
        )
    except ValueError:
        return None

    for row in active_rows:
        if row in row_specs:
            continue
        try:
            row_specs[row] = _sample_regular_row_bars(rng, palette, bar_side_len)
        except ValueError:
            return None

    for row, (bars, sep_gap) in row_specs.items():
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, (bars, _sep_gap) in row_specs.items():
        _paint_combo_extension(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    solved_combo = main(inp)
    if not np.array_equal(solved_combo, out):
        return None

    metrics = _combo_metrics(inp)
    if metrics["blank_from_conflicts"] <= 0:
        return None
    if metrics["revived_after_odd_conflicts"] <= 0:
        return None
    if metrics["diff_vs_base"] <= 0 or metrics["diff_vs_d7"] <= 0 or metrics["diff_vs_d15"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
