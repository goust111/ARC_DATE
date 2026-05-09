# Combo variant of 1ae2feb7 (index 3)
# combo_spec: D1/0+D4/2
# Name: 对角线射线有限弹药

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    The input vocabulary stays the same as the base task: one mostly solid vertical separator
#    column, 1-2 horizontal bars on exactly one side, and a blank extension side. The fused rule
#    changes two linked parts of the old extension logic at once. A bar of length L no longer
#    paints along its own row. Instead, it fires along a 45-degree ray that starts one row lower
#    and one column across the separator, then repeats every L diagonal cells: offsets 1, 1+L,
#    1+2L, ... on both row and column axes. It also has finite ammo: it attempts at most L shots.
#    Example with bars on the left, separator at column 5, bar row r=1, length L=2:
#      original task        -> (1,6), (1,8), (1,10), ...
#      D1 only              -> (2,6), (4,8), (6,10), ...
#      fused combo          -> (2,6), (4,8) and then stop
#    The nearest bar on a row is still processed first, so a farther bar can only fill holes on
#    the same diagonal and never overwrite earlier paint.
# b) Complexity calibration
#    If long bars are allowed freely, finite ammo can become invisible because the diagonal hits the
#    bottom or side edge before L shots are exhausted. To keep the combo readable, generation caps
#    ordinary bar lengths at 3 and forces one showcase row to contain a single length-2 bar high
#    enough and with enough extension width that the diagonal-to-edge baseline would paint at least
#    one extra cell beyond the combo's two shots. Other rows still vary between 1 and 2 bars, so
#    the puzzle keeps repeated evidence without exploding into too many cases.
# c) Implicit assumptions audit
#    - `_find_sep_col` still assumes a mostly filled uniform separator column. The generator keeps
#      that geometry intact, with at most one optional background gap.
#    - Side detection by non-background mass still assumes the extension side is blank in the
#      input. The generator clears the whole target side after constructing the output, so this
#      remains valid.
#    - `_iter_runs_1d` and `_collect_bars` assume all source objects are horizontal monochrome runs
#      confined to one side of the separator. That remains unchanged.
#    - The original solver assumed horizontal same-row targets all the way to the edge. The combo
#      breaks both parts: target coordinates are diagonal and the number of candidate targets is
#      limited to L. Those assumptions are isolated into the target iterators below.
#    - Non-overwrite still matters only among bars from the same source row. With a fixed
#      down-right or down-left slope, rays from different rows are parallel invariants
#      (`row-sep_col` or `row+sep_col`), so they cannot intersect.
# d) Placement legality
#    Active rows are drawn from `0..h-2`, guaranteeing the first diagonal step stays inside the
#    grid. The showcase row is additionally chosen so `min(h-1-row, ext_side_len) >= 5`, which
#    guarantees its length-2 combo ray gets two in-bounds shots while the diagonal-to-edge variant
#    would continue farther. Bars are still packed contiguously from the separator outward, so they
#    never overlap or leave the source side.
# e) Visual observability
#    The diagonal path is visible because paint leaves the source row and forms slanted traces.
#    Finite ammo is visible because those traces stop after exactly L hits instead of continuing to
#    the bottom/edge. The generator rejects any sample whose fused output accidentally matches the
#    D1-only diagonal-to-edge solver or the D4-only horizontal-limited solver.
# f) Visual cleanliness
#    Active rows keep an empty row between them, rays are parallel, and only the blank extension
#    side is populated in the output. This avoids crossings and keeps the same-row nearest-first
#    competition local to at most two adjacent bars.
# g) Color inferability
#    No color remapping is introduced. Each ray uses the visible bar color directly, and the only
#    hidden variable is bar length, which is already encoded by the bar's width. Humans can infer
#    both the diagonal stride and the ammo count from repeated length-2 and length-3 examples.
#
# Post-change solver assumptions that the generator guarantees:
#    - At least one showcase row has a single length-2 bar with enough diagonal room to make the
#      finite-ammo stop visible against the diagonal-to-edge baseline.
#    - At least two rows paint two or more output cells, so the diagonal periodicity and finite
#      ammo stop are both supported by repeated evidence rather than a single showcase row.
#    - The extension side is completely blank in the input, so side detection by mass is stable.


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


def _iter_diagonal_targets(
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    h: int,
    w: int,
    *,
    limited: bool,
):
    if length <= 0:
        return
    rr = row + 1
    cc = sep_col + 1 if bars_on_left else sep_col - 1
    dc = 1 if bars_on_left else -1
    shots = 0
    while 0 <= rr < h and 0 <= cc < w and (not limited or shots < length):
        yield rr, cc
        shots += 1
        rr += length
        cc += dc * length


def _iter_horizontal_targets(
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    w: int,
    *,
    limited: bool,
):
    if length <= 0:
        return
    cc = sep_col + 1 if bars_on_left else sep_col - 1
    dc = length if bars_on_left else -length
    shots = 0
    while 0 <= cc < w and (not limited or shots < length):
        yield row, cc
        shots += 1
        cc += dc


def _run_solver(grid: np.ndarray, *, diagonal: bool, limited: bool) -> np.ndarray:
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
        if diagonal:
            targets = _iter_diagonal_targets(
                r,
                sep_col,
                bars_on_left,
                length,
                h,
                w,
                limited=limited,
            )
        else:
            targets = _iter_horizontal_targets(
                r,
                sep_col,
                bars_on_left,
                length,
                w,
                limited=limited,
            )
        for rr, cc in targets:
            if int(out[rr, cc]) == bg:
                out[rr, cc] = col

    return out


# Overview:
#   The puzzle still contains horizontal bars on one side of a vertical separator, but each bar now
#   fires a finite number of marks along a 45-degree ray instead of filling its own row to the edge.
# Input structure:
#   A mostly solid separator column splits the grid into a bar side and an empty extension side.
#   Each active row contains 1-2 adjacent monochrome bars on the same side of the separator.
# Rule:
#   Process bars from nearest to farthest from the separator. A bar of length L paints at diagonal
#   positions one step past the separator, then every L diagonal cells after that, and it attempts
#   at most L such shots. Existing non-background cells are never overwritten.
# Key insight:
#   The same visible quantity, the bar length, controls both the spacing between diagonal hits and
#   the total number of hits, so the solver only needs to extract bar lengths and enumerate targets.
def main(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, diagonal=True, limited=True)


def _main_diagonal_to_edge(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, diagonal=True, limited=False)


def _main_horizontal_limited(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, diagonal=False, limited=True)


H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 6, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_BAR_LENGTH = 3
SHOWCASE_LENGTH = 2
SHOWCASE_CAPACITY = 5
MIN_TOTAL_PAINT = 5
MAX_ATTEMPTS = 500


def _choose_active_rows_with_forced(
    rng: np.random.Generator,
    usable_rows: int,
    count: int,
    forced_row: int,
) -> list[int] | None:
    pool = list(range(usable_rows))
    if count == 1:
        return [forced_row]
    if count - 1 > usable_rows - 1:
        return None
    others_pool = [r for r in pool if r != forced_row]
    for _ in range(200):
        others = list(map(int, rng.choice(others_pool, size=count - 1, replace=False)))
        active = sorted([forced_row] + others)
        if all(b - a >= 2 for a, b in zip(active, active[1:])):
            return active
    return None


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
    *,
    force_showcase: bool,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar side too narrow for sampled separator gap")

    max_len = min(MAX_BAR_LENGTH, capacity)
    if force_showcase:
        if capacity < SHOWCASE_LENGTH:
            raise ValueError("showcase row cannot fit the forced bar")
        return [(SHOWCASE_LENGTH, primary)], sep_gap

    can_two = capacity >= 2
    if not can_two or rng.random() >= P_TWO_BARS:
        length = int(rng.integers(1, max_len + 1))
        return [(length, primary)], sep_gap

    max_near = min(MAX_BAR_LENGTH, capacity - 1)
    near_len = int(rng.integers(1, max_near + 1))
    max_far = min(MAX_BAR_LENGTH, capacity - near_len)
    if max_far <= 0:
        return [(near_len, primary)], sep_gap

    far_len = int(rng.integers(1, max_far + 1))
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


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> int:
    h, w = out.shape
    painted = 0
    for length, color in bars:
        for rr, cc in _iter_diagonal_targets(
            row,
            sep_col,
            bars_on_left,
            length,
            h,
            w,
            limited=True,
        ):
            if out[rr, cc] == 0:
                out[rr, cc] = color
                painted += 1
    return painted


def _clear_ext_side(out: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0
    return inp


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

    usable_rows = h - 1
    max_active = min(ACTIVE_MAX, usable_rows // 2 + usable_rows % 2)
    if max_active < ACTIVE_MIN:
        return None

    showcase_candidates = [
        r for r in range(usable_rows) if min(h - 1 - r, ext_side_len) >= SHOWCASE_CAPACITY
    ]
    if not showcase_candidates:
        return None
    showcase_row = int(rng.choice(showcase_candidates))

    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows_with_forced(rng, usable_rows, active_count, showcase_row)
    if active_rows is None:
        return None
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_candidates = [r for r in inactive_rows if r == 0 or r == h - 1]
        if not gap_candidates:
            gap_candidates = inactive_rows
        gap_row = int(rng.choice(gap_candidates))
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    row_paint_counts: dict[int, int] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            force_showcase=(row == showcase_row),
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        row_paint_counts[row] = _paint_extension(out, row, sep_col, bars_on_left, bars)

    if row_paint_counts.get(showcase_row) != SHOWCASE_LENGTH:
        return None
    if sum(row_paint_counts.values()) < MIN_TOTAL_PAINT:
        return None
    if sum(count >= 2 for count in row_paint_counts.values()) < 2:
        return None

    inp = _clear_ext_side(out, sep_col, bars_on_left)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_main_diagonal_to_edge(inp), out):
        return None
    if np.array_equal(_main_horizontal_limited(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
