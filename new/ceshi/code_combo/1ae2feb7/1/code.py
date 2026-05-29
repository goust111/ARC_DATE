# Combo variant of 1ae2feb7 (index 1)
# combo_spec: D2/2+D8/2
# Name: 行号决定间隔的间歇涂色

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior and trace
#    Seed task: each active row has bars on one side of a vertical separator and
#    paints single pixels onto the opposite side. Originally a bar of length L
#    on row r paints at ext offsets {0, L, 2L, 3L, ...}, with bars processed
#    nearest-first and without overwriting.
#    Fusion rule:
#      - D8/2 changes the stride source from bar length to row index:
#        base_step = r + 1.
#      - D2/2 then sparsifies that row-based periodic pattern by keeping only
#        even-numbered hits. So the painted offsets become
#        {0, 2*(r+1), 4*(r+1), ...}.
#    Small trace on a left-bar grid:
#      - Row r=0: fused offsets are {0, 2, 4, 6, ...}.
#      - Row r=2: fused offsets are {0, 6, 12, ...}.
#      - If row 2 has a length-4 bar, the seed task would use {0, 4, 8, ...},
#        D2-only would use {0, 8, ...}, D8-only would use {0, 3, 6, 9, ...},
#        and the fused task uses {0, 6, 12, ...}.
#    The solver therefore has to infer both "step comes from the row" and
#    "odd periodic hits are skipped".
#
# b) Complexity calibration
#    Under row-based stride, multiple bars on the same row share both phase and
#    cadence, so farther bars become visually redundant after nearest-first
#    painting. To keep the two new control variables observable, generation
#    mostly uses one bar per active row, while still preserving the original
#    separator/bar object vocabulary. Active rows are forced to span top and
#    bottom regions so humans can compare dense upper-row dots against sparse
#    lower-row dots.
#
# c) Hidden assumptions audit for main()
#    - _find_sep_col(): still valid because the separator remains a nearly full
#      monochrome vertical column.
#    - left/right mass comparison: still valid because all source bars stay on
#      one side and the input ext side is cleared before solving.
#    - _iter_runs_1d(): still valid because source objects remain monochrome
#      horizontal runs.
#    - bars sorted by (distance,row,col): still valid; "nearest first, no
#      overwrite" remains part of the rule even though generation prefers one
#      bar per row for clarity.
#    - The seed assumption `step = length` is no longer valid. It must become
#      `step = r + 1`, and the paint loop must skip odd periodic hits, which is
#      equivalent to using stride `2 * (r + 1)` from the separator outward.
#
# d) Placement legality
#    Generation is still output-first. Bars are placed entirely on one side of
#    the separator, within the chosen bar-side width, optionally with a 1-cell
#    gap to the separator. The ext side starts empty in the input, and output
#    painting only fills background cells, so there are no overlaps or
#    out-of-bounds writes.
#
# e) Visual observability
#    Degenerate cases would occur if every active row only showed the first ext
#    pixel: then D8-only, D2-only, and the fusion could look identical. To
#    prevent that, the generator forces upper active rows 0 and 2 and requires
#    at least two rows to show repeated fused hits. It also rejects any sample
#    whose fused output matches the seed, D2-only, or D8-only outputs.
#
# f) Visual cleanliness
#    The output remains row-local and monochrome per painted dot, with no new
#    markers or overlays. Using one bar per row avoids same-row masking noise,
#    so the visible difference is a clean change in horizontal cadence only.
#
# g) Change inferability
#    No color remapping is introduced. Colors still identify source bars
#    directly, while the new logic is purely spatial: upper rows paint more
#    densely, and each such row shows a paint-one / skip-one pattern relative
#    to its own row-determined base period.
#
# 9) New solver assumptions and generator support
#    The fused solver newly assumes that row index, not bar width, controls the
#    ext-side period, and that blank odd hits are intentional. The generator
#    guarantees this remains inferable by constructing rows whose fused pattern
#    differs from all three simpler baselines (seed, D2-only, D8-only), while
#    keeping the separator and bar parsing assumptions unchanged.


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


def _row_step(r: int) -> int:
    return int(r) + 1


def _collect_bars(g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int):
    h, w = g.shape
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
    return bars


def _solve_with_modes(
    grid: np.ndarray,
    *,
    use_row_step: bool,
    intermittent_skip: bool,
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

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    for _, r, _c0, _c1, length, col in bars:
        base_step = _row_step(r) if use_row_step else length
        step = base_step * (2 if intermittent_skip else 1)
        if step <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each active row across the separator with row-indexed intermittent dots.

    Input structure:
    - A nearly solid vertical separator column splits the grid into a bar side and an empty extension side.
    - Several separated rows contain monochrome horizontal source bars, all on the same side of the separator.
    - The extension side is entirely background in the input.

    Rule:
    - Detect which side contains the bars and process bars nearest to the separator first.
    - For a source bar on row r, ignore its width when computing the extension period; use base step = r + 1.
    - Paint only every other row-based periodic hit on the extension side, so the actual painted offsets are 0, 2*(r+1), 4*(r+1), ...
    - Never overwrite an already painted cell; preserve the separator and source bars.

    Key insight:
    - The horizontal cadence comes from the bar's row index rather than its length, and that row-driven cadence is further thinned by a paint-one / skip-one filter.
    """
    return _solve_with_modes(grid, use_row_step=True, intermittent_skip=True)


# --------------- generator (combo-aware output-first) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 8, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 600


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    valid = []
    for combo in itertools.combinations(range(h), count):
        if 0 not in combo or 2 not in combo:
            continue
        if max(combo) < max(4, h - 4):
            continue
        if any(combo[i + 1] - combo[i] < 2 for i in range(len(combo) - 1)):
            continue
        valid.append(combo)
    if not valid:
        raise RuntimeError("failed to choose active rows")
    chosen = valid[int(rng.integers(0, len(valid)))]
    return list(map(int, chosen))


def _ext_positions(ext_side_len: int, step: int) -> tuple[int, ...]:
    if step <= 0:
        return ()
    return tuple(range(0, ext_side_len, step))


def _candidate_lengths(capacity: int, row: int, ext_side_len: int) -> list[int]:
    row_step = _row_step(row)
    fused = set(_ext_positions(ext_side_len, 2 * row_step))
    d8_only = set(_ext_positions(ext_side_len, row_step))

    scored: list[tuple[int, int]] = []
    for length in range(1, capacity + 1):
        seed = set(_ext_positions(ext_side_len, length))
        d2_only = set(_ext_positions(ext_side_len, 2 * length))
        if fused == seed or fused == d2_only or fused == d8_only:
            continue
        score = len(fused ^ seed) + len(fused ^ d2_only) + len(fused ^ d8_only)
        if len(fused) >= 2:
            if fused != d2_only:
                score += 3
            if fused != seed:
                score += 2
        if row in (0, 2) and len(fused) >= 2:
            score += 2
        scored.append((score, length))

    if scored:
        scored.sort(reverse=True)
        best_score = scored[0][0]
        return [length for score, length in scored if score == best_score]

    fallback = [length for length in range(1, capacity + 1) if length not in {_row_step(row), 2 * _row_step(row)}]
    return fallback if fallback else list(range(1, capacity + 1))


def _sample_row_bar(
    rng: np.random.Generator,
    primary: int,
    bar_side_len: int,
    row: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise RuntimeError("invalid bar capacity")

    choices = _candidate_lengths(capacity, row, ext_side_len)
    if not choices:
        raise RuntimeError("no legal bar lengths")
    length = int(rng.choice(choices))
    return [(length, primary)], sep_gap


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


def _ext_view(grid: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    return grid[:, sep_col + 1 :] if bars_on_left else grid[:, :sep_col]


def _ext_metrics(
    out: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    active_rows: list[int],
) -> tuple[int, int, int]:
    ext = _ext_view(out, sep_col, bars_on_left)
    painted = int(np.sum(ext != 0))
    per_row_counts = [int(np.sum(ext[r] != 0)) for r in active_rows]
    rows_with_repeat = sum(count >= 2 for count in per_row_counts)
    distinct_counts = len(set(count for count in per_row_counts if count > 0))
    return painted, rows_with_repeat, distinct_counts


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
    if max_active < ACTIVE_MIN:
        return None
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
        bars, sep_gap = _sample_row_bar(rng, row_primary[idx], bar_side_len, row, ext_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    out = main(inp)

    painted, rows_with_repeat, distinct_counts = _ext_metrics(out, sep_col, bars_on_left, active_rows)
    if painted == 0 or rows_with_repeat < 2 or distinct_counts < 2:
        return None

    seed_out = _solve_with_modes(inp, use_row_step=False, intermittent_skip=False)
    d2_out = _solve_with_modes(inp, use_row_step=False, intermittent_skip=True)
    d8_out = _solve_with_modes(inp, use_row_step=True, intermittent_skip=False)

    if np.array_equal(out, seed_out):
        return None
    if np.array_equal(out, d2_out):
        return None
    if np.array_equal(out, d8_out):
        return None

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
