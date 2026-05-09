# Combo variant of 1ae2feb7 (index 21)
# combo_spec: D2/4+D4/2
# Name: 递减间隔有限涂点

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The separator and bar layout stay in the original family: one mostly solid vertical separator,
#    1-2 horizontal monochrome bars on exactly one side, and the opposite side empty in the input.
#    The fused change is on the extension side. For a bar of length L, we walk away from the
#    separator using gaps L, L-1, L-2, ... until the gap reaches 1, then keep moving by 1 while
#    remaining in bounds. Unlike the base task, the bar does not paint every visited target forever:
#    it stops after it has successfully colored L previously blank cells. Example with bars on the
#    left and separator at column 5:
#      single bar, L=3:
#        original      -> abs cols 6,9,12,...
#        periodic+ammo -> abs cols 8,11,14
#        combo         -> abs cols 9,11,12   (ext positions 3,5,6)
#      conflict row with near L=2 then far L=3:
#        near paints ext positions 2,3 first
#        far visits 3,5,6,7,... ; ext pos 3 is blocked, so it keeps going and paints 5,6,7
#        This is the key fusion effect: decreasing trajectory determines where it searches, while
#        the ammo limit counts successful new paints rather than raw attempts.
# b) Complexity calibration:
#    If we used the strict D2 single-variant interpretation ("only the L,L-1,...,1 jumps exist"),
#    then a bar of length L would have at most L candidate targets anyway, making D4 visually
#    redundant. To avoid that degeneracy, the fused rule keeps the decreasing phase as the visible
#    signature but lets the trail continue with unit steps once the gap bottoms out. That preserves a
#    small, human-trackable rule while giving the finite-ammo constraint a real effect.
# c) Main() assumption audit:
#    - `_find_sep_col` still assumes one mostly solid single-color separator column; generator keeps
#      that structure.
#    - Side detection by non-background mass still assumes the ext side is empty in the input; the
#      generator clears it before returning.
#    - `_iter_runs_1d` still assumes bars are horizontal monochrome runs; unchanged.
#    - The original `range(..., step=length)` assumed constant spacing and edge-only termination.
#      Both assumptions break here. We replace that with an explicit target iterator plus a painter
#      that counts successful placements and may skip over blocked cells.
#    - Because successful-paint counting depends on nearer bars already occupying cells, the original
#      nearest-first ordering becomes even more important; generator includes a dedicated conflict row
#      to exercise that dependency.
# d) Placement legality:
#    Bars still stay entirely on one side of the separator and never overlap each other on the bar
#    side. The generator constrains `bar_side_len >= 6` so a forced conflict row with lengths 2 and 3
#    fits, and `ext_side_len >= 8` so the fused trail has room to show both dimensions. Output is
#    built first, then the ext side is cleared to form the input, and every sample is validated with
#    `main(input) == output`.
# e) Visual observability:
#    Each sample includes:
#      * one single-bar showcase row of length 3 or 4, making the non-periodic tightening pattern
#        visible even without conflicts, and
#      * one two-bar conflict row where the farther bar must continue past blocked early targets to
#        spend all of its ammo.
#    This gives the human solver repeated evidence for both the geometric rule and the finite-count
#    rule using only spatial cues.
# f) Visual cleanliness:
#    Active rows remain separated by at least one empty row, so interactions only happen within a row.
#    The extension side still contains only point trails, and the bar side still contains only solid
#    runs, which keeps the scene readable.
# g) Color inference:
#    No new color remapping is introduced. Every painted point keeps the color of its source bar, and
#    the only ordering signal is spatial proximity to the separator, which is directly visible.


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


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    bars: list[tuple[int, int, int, int, int, int]] = []
    h, w = g.shape
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


def _ext_len(sep_col: int, bars_on_left: bool, w: int) -> int:
    return w - sep_col - 1 if bars_on_left else sep_col


def _to_abs_col(sep_col: int, bars_on_left: bool, ext_pos: int) -> int:
    return sep_col + 1 + ext_pos if bars_on_left else sep_col - 1 - ext_pos


def _iter_combo_abs_positions(
    sep_col: int,
    bars_on_left: bool,
    w: int,
    length: int,
):
    """Yield target cells for the fused rule: decreasing gaps, then unit gaps."""
    ext_len = _ext_len(sep_col, bars_on_left, w)
    pos = length
    gap = max(1, length - 1)
    while pos < ext_len:
        yield _to_abs_col(sep_col, bars_on_left, pos)
        pos += gap
        if gap > 1:
            gap -= 1


def _iter_periodic_abs_positions(
    sep_col: int,
    bars_on_left: bool,
    w: int,
    length: int,
):
    ext_len = _ext_len(sep_col, bars_on_left, w)
    pos = length
    while pos < ext_len:
        yield _to_abs_col(sep_col, bars_on_left, pos)
        pos += length


def _iter_strict_decreasing_abs_positions(
    sep_col: int,
    bars_on_left: bool,
    w: int,
    length: int,
):
    """D2-only reference: only the positive decreasing gaps exist."""
    ext_len = _ext_len(sep_col, bars_on_left, w)
    pos = length
    gap = length - 1
    while pos < ext_len:
        yield _to_abs_col(sep_col, bars_on_left, pos)
        if gap <= 0:
            break
        pos += gap
        gap -= 1


def _paint_with_quota(
    out: np.ndarray,
    row: int,
    targets,
    color: int,
    bg: int,
    quota: int | None,
) -> int:
    painted = 0
    for cc in targets:
        if int(out[row, cc]) == bg:
            out[row, cc] = color
            painted += 1
            if quota is not None and painted >= quota:
                break
    return painted


def _run_solver(
    grid: np.ndarray,
    *,
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

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if mode == "combo":
            _paint_with_quota(
                out,
                r,
                _iter_combo_abs_positions(sep_col, bars_on_left, w, length),
                col,
                bg,
                quota=length,
            )
        elif mode == "periodic_quota":
            _paint_with_quota(
                out,
                r,
                _iter_periodic_abs_positions(sep_col, bars_on_left, w, length),
                col,
                bg,
                quota=length,
            )
        elif mode == "decreasing_unlimited":
            _paint_with_quota(
                out,
                r,
                _iter_combo_abs_positions(sep_col, bars_on_left, w, length),
                col,
                bg,
                quota=None,
            )
        elif mode == "strict_decreasing":
            _paint_with_quota(
                out,
                r,
                _iter_strict_decreasing_abs_positions(sep_col, bars_on_left, w, length),
                col,
                bg,
                quota=None,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each horizontal bar across the empty side of the separator using the fused combo rule.

    Input structure:
    A grid contains one mostly solid vertical separator column, with 1-2 monochrome horizontal bars
    on exactly one side and an empty extension side on the other.

    Rule:
    Process bars from nearest to the separator outward. A bar of length L searches extension cells at
    offsets L, L+(L-1), L+(L-1)+(L-2), ... ; once the gap reaches 1 it continues one cell at a time.
    The bar paints only blank cells and stops after it has successfully painted L cells or it runs
    out of extension-side space.

    Key insight:
    The trajectory is not periodic anymore, and the stopping condition depends on successful paints,
    so a farther bar may skip over cells already claimed by a nearer bar and keep going until its own
    ammo is exhausted.
    """
    return _run_solver(grid, mode="combo")


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 6, 11
EXT_SIDE_MIN, EXT_SIDE_MAX = 8, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_TWO_BARS = 0.4
P_SEP_GAP = 0.1
P_SEP_COLUMN_GAP = 0.2
P_RANDOM_CONFLICT_ROW = 0.35

SHOWCASE_LENGTHS = (3, 4)
FILLER_MAX_LENGTH = 4
MIN_TOTAL_TRAIL_CELLS = 7
MAX_ATTEMPTS = 600


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_showcase_bars(
    rng: np.random.Generator,
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    for length in rng.permutation(SHOWCASE_LENGTHS):
        if int(length) <= bar_side_len and ext_side_len > int(length) + 2:
            return [(int(length), primary)], 0
    length = min(bar_side_len, 3)
    return [(length, primary)], 0


def _sample_conflict_bars(
    palette: list[int],
    primary: int,
) -> tuple[list[tuple[int, int]], int]:
    others = [c for c in palette if c != primary]
    far_color = int(others[0])
    return [(2, primary), (3, far_color)], 0


def _sample_filler_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    *,
    force_conflict: bool,
) -> tuple[list[tuple[int, int]], int]:
    if force_conflict:
        return _sample_conflict_bars(palette, primary)

    for _ in range(80):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        if capacity <= 0:
            continue

        can_two = capacity >= 2
        if not can_two or rng.random() >= P_TWO_BARS:
            length = int(rng.integers(1, min(FILLER_MAX_LENGTH, capacity) + 1))
            return [(length, primary)], sep_gap

        near_len = int(rng.integers(1, min(FILLER_MAX_LENGTH, capacity - 1) + 1))
        far_cap = min(FILLER_MAX_LENGTH, capacity - near_len)
        if far_cap <= 0:
            continue
        far_len = int(rng.integers(1, far_cap + 1))
        others = [c for c in palette if c != primary]
        far_color = int(rng.choice(others))
        return [(near_len, primary), (far_len, far_color)], sep_gap

    return [(1, primary)], 0


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


def _bars_fit(bar_side_len: int, bars: list[tuple[int, int]], sep_gap: int) -> bool:
    return sum(length for length, _ in bars) + sep_gap <= bar_side_len


def _paint_combo_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
) -> int:
    w = out.shape[1]
    painted = 0
    for length, color in bars:
        painted += _paint_with_quota(
            out,
            row,
            _iter_combo_abs_positions(sep_col, bars_on_left, w, length),
            color,
            bg,
            quota=length,
        )
    return painted


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
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    showcase_row = active_rows[0]
    conflict_row = active_rows[1]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}

    for idx, row in enumerate(active_rows):
        primary = row_primary[idx]
        if row == showcase_row:
            bars, sep_gap = _sample_showcase_bars(rng, primary, bar_side_len, ext_side_len)
        elif row == conflict_row:
            bars, sep_gap = _sample_conflict_bars(palette, primary)
        else:
            bars, sep_gap = _sample_filler_bars(
                rng,
                palette,
                primary,
                bar_side_len,
                force_conflict=bool(rng.random() < P_RANDOM_CONFLICT_ROW and bar_side_len >= 5),
            )
        if not _bars_fit(bar_side_len, bars, sep_gap):
            return None
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    total_painted = 0
    for row, bars in row_bars_map.items():
        total_painted += _paint_combo_row(out, row, sep_col, bars_on_left, bars, bg=0)

    if total_painted < MIN_TOTAL_TRAIL_CELLS:
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

    combo_out = main(inp)
    if not np.array_equal(combo_out, out):
        return None

    periodic_quota_out = _run_solver(inp, mode="periodic_quota")
    decreasing_unlimited_out = _run_solver(inp, mode="decreasing_unlimited")
    strict_decreasing_out = _run_solver(inp, mode="strict_decreasing")

    if np.array_equal(combo_out, periodic_quota_out):
        return None
    if np.array_equal(combo_out, decreasing_unlimited_out):
        return None
    if np.array_equal(combo_out, strict_decreasing_out):
        return None

    showcase_delta = int(np.count_nonzero(combo_out[showcase_row] != periodic_quota_out[showcase_row]))
    conflict_delta = int(np.count_nonzero(combo_out[conflict_row] != strict_decreasing_out[conflict_row]))
    quota_delta = int(np.count_nonzero(combo_out != decreasing_unlimited_out))
    if showcase_delta < 2 or conflict_delta < 1 or quota_delta < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
