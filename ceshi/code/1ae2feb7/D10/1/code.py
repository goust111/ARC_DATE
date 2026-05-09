# Variant of 1ae2feb7
# Dimension: D10 — 色条数量与排列 (src)
# Change: 每行1-2个单色色条，相邻排列（无间隔），紧邻分隔条；2-5个活跃行，行间至少隔1空行... → 色条之间有间隔

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg

# Design analysis for the D10 variant
# -----------------------------------
# a) Concrete behavior + trace
#    Original task:
#    - Each active row contains 1-2 monochrome bars on one side of a vertical separator.
#    - A bar of length L paints the opposite side at period L, always starting from the first cell
#      immediately across the separator.
#    - If a row has two bars, they are processed from nearest to farthest, so the near bar can block
#      some later dots.
#
#    Variant:
#    - Rows may still have 1-2 bars, but if there are two bars they are no longer adjacent:
#      there is at least one background cell between them.
#    - The periodic step is still the bar length L.
#    - New rule: the first painted target cell is shifted by the bar's distance D from the separator.
#      So a bar with distance D paints offsets D, D+L, D+2L, ... on the extension side.
#    - This forces the solver to use both pieces of visible information:
#      * length -> period
#      * position -> phase / starting offset
#
#    Small trace on a left-bar row with separator at column 6:
#    input row:   44 0 777 | 0 0 0 0 0 0 0
#    indices:     01 2 345 | 6 7 8 9 A B C
#    near bar 777 has length 3, distance 0 -> paints at 7,10,...
#    far  bar 44  has length 2, distance 4 -> paints at B,...
#    Original logic would start both bars at column 7; variant starts the far bar later.
#    That difference is exactly caused by the inter-bar gap and the farther position.
#
# b) Complexity calibration
#    - Gap sizes are restricted to 1-2 cells, so the phase rule is visible but not too unconstrained.
#    - Bar count stays at 1-2 per row; active rows stay at 2-5; row spacing stays >= 1 empty row.
#    - Each sample is forced to contain at least one 2-bar row with a real inter-bar gap, so the D10
#      change is observable and the task does not collapse back to the original layout.
#
# c) Hidden-assumption audit against the original main()
#    - _iter_runs_1d: still valid, because bars remain monochrome contiguous runs. Only adjacency changes.
#    - dist computation: originally used only for sorting. In the variant dist must also affect the start
#      column of target-side painting. Keeping the old "always start right across the separator" would be
#      incorrect because it ignores the newly meaningful bar position.
#    - bars.sort(...): still valid. Nearer bars must still be processed first so no-overwrite behaves the
#      same way conceptually, just with shifted phases.
#    - overwrite check `if out[r, cc] == bg`: still valid. The variant changes where dots start, not the
#      non-overwrite policy.
#
# d) Placement legality
#    - For a left-side bar with distance D and length L, its occupied columns are
#      [sep_col - D - L, sep_col - D - 1].
#      For a right-side bar they are [sep_col + 1 + D, sep_col + D + L].
#    - The generator samples lengths / separator-gap / inter-bar-gap so bars fit fully on the bar side.
#    - It also enforces distance < ext_side_len for every bar, so each bar can paint at least one target
#      cell.
#    - It validates that every sampled bar contributes at least one visible target cell after the
#      nearer-bar blocking rule. This avoids visually dead bars.
#
# e) Visual observability
#    - The human-visible cue is that farther bars do not begin painting immediately across the separator.
#      Two bars with the same length but different positions start at different offsets.
#    - Because each grid has at least one gapped 2-bar row and every bar paints at least one visible
#      cell, the new position-sensitive behavior can be inferred from 2-3 examples.
#
# f) Visual cleanliness
#    - The grid is still partitioned into bar side / separator / extension side, so no new object type is
#      introduced in the same region.
#    - Gaps are small and axis-aligned; rows remain separated by empty rows, so the output stays legible.
#
# g) Color reasoning
#    - Colors are not used for ordering or arithmetic.
#    - Color only labels which bar produced which dot pattern, exactly as in the source task.
#
# New solver assumptions and generator guarantees
#    - Assumption: all meaningful non-background runs on the bar side are bars.
#      Guarantee: generator puts only bars there.
#    - Assumption: each farther bar has strictly larger distance than the nearer bar on the same row.
#      Guarantee: generator uses explicit positive inter-bar gaps.
#    - Assumption: at least one bar position differs enough from the source-task behavior to matter.
#      Guarantee: every puzzle has at least one 2-bar row with an inter-bar gap, and the generator rejects
#      any sample whose output would still match the source solver.


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
        col = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == col:
            c += 1
        yield c0, c - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars: list[tuple[int, int, int, int, int, int]] = []
    for r in range(h):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))

    # Process bars from nearest to farthest. Unlike the source task, the bar's distance from the
    # separator also shifts the first painted cell on the extension side.
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for dist, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            start = sep_col + 1 + dist
            for cc in range(start, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            start = sep_col - 1 - dist
            for cc in range(start, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    return out


def _source_main(grid: np.ndarray) -> np.ndarray:
    """
    Source-task solver kept only as a degeneration filter for the generator.
    It ignores distance when choosing the first extension cell.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars: list[tuple[int, int, int, int, int, int]] = []
    for r in range(h):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _dist, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
    return out


# --------------- generator for D10 variant ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.7
P_SEP_GAP = 0.18
P_SWAP_COLORS = 0.5
INTER_GAP_MIN, INTER_GAP_MAX = 1, 2

P_SEP_COLUMN_GAP = 0.2

ROW_ATTEMPTS = 80
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
    force_two_bars: bool,
) -> list[tuple[int, int, int]] | None:
    """
    Return bars as [(dist_from_separator, length, color), ...] ordered near -> far.
    Distances encode the variant's new phase shift.
    """
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    want_two = force_two_bars or (rng.random() < P_TWO_BARS)

    if not want_two:
        max_len = bar_side_len - sep_gap
        if max_len < 1 or sep_gap >= ext_side_len:
            return None
        length = int(rng.integers(1, max_len + 1))
        return [(sep_gap, length, primary)]

    others = [c for c in palette if c != primary]
    if not others:
        return None

    near_color, far_color = primary, int(rng.choice(others))
    if rng.random() < P_SWAP_COLORS:
        near_color, far_color = far_color, near_color

    inter_gap = int(rng.integers(INTER_GAP_MIN, INTER_GAP_MAX + 1))
    max_near_len = bar_side_len - sep_gap - inter_gap - 1
    if max_near_len < 1:
        return None

    near_len = int(rng.integers(1, max_near_len + 1))
    far_dist = sep_gap + near_len + inter_gap
    max_far_len = bar_side_len - far_dist
    if max_far_len < 1 or far_dist >= ext_side_len:
        return None

    far_len = int(rng.integers(1, max_far_len + 1))
    return [(sep_gap, near_len, near_color), (far_dist, far_len, far_color)]


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int]],
) -> None:
    for dist, length, color in bars:
        if bars_on_left:
            c1 = sep_col - 1 - dist
            c0 = c1 - length + 1
        else:
            c0 = sep_col + 1 + dist
            c1 = c0 + length - 1
        out[row, c0 : c1 + 1] = color


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int]],
) -> list[int]:
    w = out.shape[1]
    counts: list[int] = []
    for dist, length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1 + dist, w, length)
        else:
            positions = range(sep_col - 1 - dist, -1, -length)
        count = 0
        for cc in positions:
            if int(out[row, cc]) == 0:
                out[row, cc] = color
                count += 1
        counts.append(count)
    return counts


def _bar_visibility_counts(
    width: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int]],
) -> list[int]:
    row = np.zeros(width, dtype=int)
    counts: list[int] = []
    for dist, length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1 + dist, width, length)
        else:
            positions = range(sep_col - 1 - dist, -1, -length)
        count = 0
        for cc in positions:
            if int(row[cc]) == 0:
                row[cc] = color
                count += 1
        counts.append(count)
    return counts


def _has_real_inter_bar_gap(bars: list[tuple[int, int, int]]) -> bool:
    if len(bars) < 2:
        return False
    near_dist, near_len, _ = bars[0]
    far_dist, _far_len, _ = bars[1]
    return far_dist > near_dist + near_len


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
    active_row_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_row_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    must_be_two_bar_row = int(rng.choice(active_rows))

    row_bars_map: dict[int, list[tuple[int, int, int]]] = {}
    saw_real_gap = False

    for idx, row in enumerate(active_rows):
        bars = None
        force_two = row == must_be_two_bar_row
        for _ in range(ROW_ATTEMPTS):
            candidate = _sample_row_bars(
                rng,
                palette,
                row_primary[idx],
                bar_side_len,
                ext_side_len,
                force_two_bars=force_two,
            )
            if candidate is None:
                continue
            vis_counts = _bar_visibility_counts(w, sep_col, bars_on_left, candidate)
            if any(count == 0 for count in vis_counts):
                continue
            bars = candidate
            break
        if bars is None:
            return None
        row_bars_map[row] = bars
        saw_real_gap = saw_real_gap or _has_real_inter_bar_gap(bars)
        _place_bars(out, row, sep_col, bars_on_left, bars)

    if not saw_real_gap:
        return None

    for row, bars in row_bars_map.items():
        counts = _paint_extension(out, row, sep_col, bars_on_left, bars)
        if any(count == 0 for count in counts):
            return None

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
    if np.array_equal(_source_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
