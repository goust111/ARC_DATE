# Variant of 1ae2feb7
# Dimension: D1 — 射线路径方向 (path)
# Change: straight_ray——从分隔条出发，沿水平方向向对侧网格边缘延伸（垂直于分隔条方向的直线射线）... → L形路径（先水平后垂直）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for this D1 variant
# ===================================
# Source-task note:
# - The prompt names e87109e9 as the turning reference, but no solver file for that task
#   exists in this repo. Only its ARC JSON is present, so the turning mechanism below is
#   designed from the original 1ae2feb7 solver plus the requested "horizontal then vertical"
#   requirement, while keeping the original task's other invariants intact.
#
# a) Concrete behavior and trace
# - Original task:
#   * detect one vertical separator column
#   * bars all lie on one side of that separator
#   * each bar paints periodic dots only on its own row, starting at the cell immediately
#     across the separator, with step = bar length
#   * nearer bars are processed first and later bars never overwrite
# - Variant:
#   * keep the same separator detection, same bar extraction, same nearest-first priority,
#     same no-overwrite rule
#   * change only the target path shape:
#       1. start from the cell immediately across the separator on the extension side
#       2. move horizontally all the way to that side's outer grid edge
#       3. at the edge, turn 90 degrees
#       4. continue vertically along the edge toward the nearer of the top/bottom borders
#   * the bar still paints every `length` cells along the full concatenated path
# - Example:
#   * suppose bars are on the left, separator at c=4, row r=2 in an 8x10 grid, and a bar
#     has length 3
#   * original straight path cells were:
#       (2,5),(2,6),(2,7),(2,8),(2,9)
#     painted positions: indices 0,3 -> (2,5),(2,8)
#   * new L path cells are:
#       (2,5),(2,6),(2,7),(2,8),(2,9),(1,9),(0,9)
#     because row 2 is nearer the top than the bottom
#     painted positions: indices 0,3,6 -> (2,5),(2,8),(0,9)
#   * difference: the rhythm continues after the corner instead of stopping at the far edge
#
# b) Complexity calibration
# - The change is visually meaningful because output cells now appear off the active rows, on
#   the far boundary column. This is not a subtle recoloring; it changes the geometry.
# - To keep the rule inferable, the generator avoids the exact middle row (where "nearer top
#   or bottom" would tie) and usually includes active rows in both halves, so samples show
#   both upward and downward turns.
# - We do not add extra object types or extra ordering keys; the only new degree of freedom is
#   the vertical turn direction implied by row position.
#
# c) Implicit-assumption audit against the original main()
# - `_find_sep_col(...)`: still valid. The separator remains a mostly full monochrome column.
# - side detection by left/right non-bg mass: still valid. Bars still all live on one side.
# - `_iter_runs_1d(...)` over each row: still valid. Bars are unchanged monochrome row-runs.
# - `bars.sort(key=(dist,row,c0))`: now more important than before. In the original task,
#   rows were independent, so cross-row ordering barely mattered. In this variant, multiple
#   L paths can overlap on the far edge column, so generator and solver must share this exact
#   global order or outputs diverge.
# - the old in-row `for cc in range(...)` is no longer valid, because it assumes every path
#   stays on one row. It must be replaced by iteration over an explicit ordered path.
#
# d) Placement legality
# - Input legality remains simple: bars only occupy the bar side; extension side is blank.
# - Output legality is handled by simulation: each path is clipped to grid bounds by
#   construction, and no-overwrite prevents collisions from corrupting prior marks.
# - The generator validates every sample with `main(inp) == out`.
#
# e) Visual observability
# - A human solver can spot the change because some added cells leave the original row and hug
#   the far border column. This is visible in 2-3 examples.
# - We reject generated samples where the L-path output accidentally matches the original
#   straight-ray output, so the variant does not silently degenerate.
#
# f) Visual cleanliness
# - Overlap can happen only on the far edge column. To keep that readable, the generator uses
#   2-4 active rows with row spacing >= 2 and tries to place rows in both halves, limiting the
#   amount of edge-column pileup.
#
# g) Color inference
# - No color remapping is introduced. Colors still label bars directly and are copied along the
#   path. This avoids any hidden color-code logic.
#
# h) New solver assumptions introduced by the variant
# - The solver assumes the turn direction is determined by geometric proximity to the top/bottom
#   edge, with ties broken upward.
# - The generator guarantees tie-free active rows so this rule is visibly unambiguous in the
#   produced samples, though the solver still handles arbitrary tie cases deterministically.


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


def _bars_on_left(g: np.ndarray, sep_col: int, bg: int) -> bool:
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    return left_mass >= right_mass


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


def _turn_vertical_dir(row: int, h: int) -> int:
    top_dist = row
    bottom_dist = h - 1 - row
    return -1 if top_dist <= bottom_dist else 1


def _l_path_cells(
    h: int,
    w: int,
    row: int,
    sep_col: int,
    bars_on_left: bool,
) -> list[tuple[int, int]]:
    if bars_on_left:
        path = [(row, c) for c in range(sep_col + 1, w)]
        edge_col = w - 1
    else:
        path = [(row, c) for c in range(sep_col - 1, -1, -1)]
        edge_col = 0

    dr = _turn_vertical_dir(row, h)
    if dr < 0:
        path.extend((rr, edge_col) for rr in range(row - 1, -1, -1))
    else:
        path.extend((rr, edge_col) for rr in range(row + 1, h))
    return path


def _straight_path_cells(
    _h: int,
    w: int,
    row: int,
    sep_col: int,
    bars_on_left: bool,
) -> list[tuple[int, int]]:
    if bars_on_left:
        return [(row, c) for c in range(sep_col + 1, w)]
    return [(row, c) for c in range(sep_col - 1, -1, -1)]


def _paint_periodic_path(
    out: np.ndarray,
    path: list[tuple[int, int]],
    step: int,
    color: int,
    bg: int,
) -> None:
    if step <= 0:
        return
    for idx in range(0, len(path), step):
        r, c = path[idx]
        if int(out[r, c]) == bg:
            out[r, c] = color


def _solve_with_path(grid: np.ndarray, *, use_l_path: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    bars_on_left = _bars_on_left(g, sep_col, bg)
    bars = _collect_bars(g, sep_col, bars_on_left, bg)

    for _, r, _c0, _c1, length, col in bars:
        path = (
            _l_path_cells(h, w, r, sep_col, bars_on_left)
            if use_l_path
            else _straight_path_cells(h, w, r, sep_col, bars_on_left)
        )
        _paint_periodic_path(out, path, length, col, bg)

    return out


def _solve_straight(grid: np.ndarray) -> np.ndarray:
    return _solve_with_path(grid, use_l_path=False)


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_path(grid, use_l_path=True)


# --------------- generator (D1 L-path variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 4

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 600
ROW_PICK_ATTEMPTS = 200


def _has_spacing(rows: list[int]) -> bool:
    return all(b - a >= 2 for a, b in zip(rows, rows[1:]))


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    candidates = [r for r in range(h) if not (h % 2 == 1 and r == h // 2)]
    top_half = [r for r in candidates if r < h // 2]
    bottom_half = [r for r in candidates if r > (h - 1) // 2]

    for _ in range(ROW_PICK_ATTEMPTS):
        rows = sorted(map(int, rng.choice(candidates, size=count, replace=False)))
        if not _has_spacing(rows):
            continue
        if not any(r in top_half for r in rows):
            continue
        if not any(r in bottom_half for r in rows):
            continue
        return rows

    # Fallback: deterministic spaced rows in both halves.
    rows = list(range(0, h, 2))
    rows = [r for r in rows if r in candidates]
    if len(rows) >= count and any(r in top_half for r in rows[:count]) and any(r in bottom_half for r in rows[:count]):
        return rows[:count]
    raise RuntimeError("failed to pick active rows with visible turn directions")


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
    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS

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


def _count_l_extra_cells(inp: np.ndarray) -> int:
    straight = _solve_straight(inp)
    l_out = main(inp)
    return int(np.sum(straight != l_out))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    active_count = int(rng.integers(ACTIVE_MIN, ACTIVE_MAX + 1))
    try:
        active_rows = _choose_active_rows(rng, h, active_count)
    except RuntimeError:
        return None
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    out = main(inp)
    if np.array_equal(_solve_straight(inp), out):
        return None
    if _count_l_extra_cells(inp) < 2:
        return None

    # Ensure some off-row vertical continuation is actually visible.
    active_mask = np.zeros_like(inp, dtype=bool)
    for row in active_rows:
        active_mask[row, :] = True
    vertical_only = int(np.sum((out != 0) & (~active_mask) & (inp == 0)))
    if vertical_only < 1:
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
