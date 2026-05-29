# Combo variant of 1ae2feb7 (index 8)
# combo_spec: D1/3+D2/4
# Name: L形路径递减间隔涂色

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for this combo variant
# ======================================
# a) Concrete behavior and trace
# - Base task:
#   * find one mostly solid vertical separator column
#   * all horizontal bars lie on exactly one side of that separator
#   * process bars from nearest to separator outward
#   * on the opposite side, paint a periodic comb on the same row at path indices
#     0, L, 2L, ... where L is the bar length, never overwriting existing color
# - Combo task:
#   * keep the separator detection, side detection, bar extraction, ordering, and no-overwrite
#   * replace the straight row-only path by an L path:
#       1. start at the first extension cell across the separator
#       2. move horizontally to the outer border of the extension side
#       3. turn 90 degrees
#       4. continue vertically toward the nearer top/bottom border
#   * replace the constant period by decreasing gaps along that full ordered path:
#       first mark at index L, then L+(L-1), then L+(L-1)+(L-2), ...
# - Example:
#   * left bars, separator at c=4, row r=2, grid 8x10, bar length L=4
#   * straight periodic baseline marks path indices 0 and 4 -> (2,5),(2,9)
#   * L-periodic single-variant marks 0 and 4 on
#       (2,5),(2,6),(2,7),(2,8),(2,9),(1,9),(0,9)
#       so it paints (2,5),(2,9)
#   * straight-decreasing single-variant marks ext positions 4 and 7 -> (2,9) only
#   * combo marks L-path indices 4 and 7 -> (2,9),(0,9)
#   * difference: the first mark is delayed by the decreasing rule, and later marks can survive
#     past the corner because the path itself turns
#
# b) Complexity calibration
# - The change is visible in two independent ways:
#   * the first extension cell next to the separator stays blank, unlike the base task
#   * some painted cells leave the source row and appear on the far boundary column
# - The generator keeps only 2-4 active rows with empty rows between them so the far-edge
#   overlaps stay readable.
# - To avoid a weak "just one of the dimensions mattered" sample, generation rejects inputs
#   whose combo output matches either single-variant output on the same input.
#
# c) Implicit-assumption audit against the original main()
# - `_find_sep_col(...)` still works because the separator remains one mostly full monochrome
#   column, with at most one inactive-row gap.
# - Left/right mass side detection still works because the extension side is empty in the input
#   and all bars remain on one side.
# - `_iter_runs_1d(...)` still works because bars are still plain horizontal monochrome runs.
# - The original in-row `range(..., step=length)` is invalid twice over:
#   * the path is no longer row-local after the corner
#   * the mark positions are no longer an arithmetic progression with constant stride
#   Therefore the solver must materialize the ordered path and iterate explicit decreasing
#   cumulative indices.
# - Global `bars.sort(key=(dist,row,c0))` becomes more important than in the base task because
#   different rows can now collide on the far border column. Generator and solver share the same
#   order so no-overwrite stays deterministic.
#
# d) Placement legality
# - Bars stay inside one side of the separator and remain contiguous per row, with an optional
#   one-cell gap only between the nearest bar and the separator.
# - The L path is always clipped by construction: first horizontal segment stays on the extension
#   side, second segment stays on the outer border column.
# - Generator chooses feature rows away from the exact middle and away from the border, so the
#   turn direction is unambiguous and at least one vertical continuation cell exists.
#
# e) Visual observability
# - A human can infer the combo from 2-3 samples because both cues repeat:
#   * every active row's first extension cell remains empty
#   * at least one row paints onto the far border column past the corner
# - The generator forces a feature row whose trail has at least two decreasing-gap marks and at
#   least one visible vertical mark, then rejects any sample that collapses to the D1-only or
#   D2-only behavior.
#
# f) Visual cleanliness
# - The only multi-row overlap zone is the far border column. Rows are spaced by at least one
#   empty row, active-row count is capped at 4, and the feature row is chosen as an outermost
#   active row on its side of the grid so its vertical leg does not get swallowed by another
#   active row closer to the same border.
#
# g) Change inferability
# - No color remapping is introduced. Colors still copy directly from bars to painted trail cells.
#   This keeps the learned rule geometric instead of color-coded.
#
# h) New solver assumptions introduced here
# - Turn direction is determined by geometric proximity to top/bottom, with ties broken upward.
# - The generator avoids tie rows by excluding the exact middle row when the height is odd.
# - For the combo to be visible, at least one bar must be short enough to paint more than once on
#   its L path and long enough for one decreasing mark to land on the vertical segment. The
#   generator explicitly constructs and validates that case.


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
    return -1 if row <= (h - 1 - row) else 1


def _horizontal_len(w: int, sep_col: int, bars_on_left: bool) -> int:
    return w - sep_col - 1 if bars_on_left else sep_col


def _vertical_len(row: int, h: int) -> int:
    return row if _turn_vertical_dir(row, h) < 0 else (h - 1 - row)


def _path_len(h: int, w: int, row: int, sep_col: int, bars_on_left: bool) -> int:
    return _horizontal_len(w, sep_col, bars_on_left) + _vertical_len(row, h)


def _path_cells(
    h: int,
    w: int,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    use_l_path: bool,
) -> list[tuple[int, int]]:
    if bars_on_left:
        path = [(row, c) for c in range(sep_col + 1, w)]
        edge_col = w - 1
    else:
        path = [(row, c) for c in range(sep_col - 1, -1, -1)]
        edge_col = 0

    if not use_l_path:
        return path

    dr = _turn_vertical_dir(row, h)
    if dr < 0:
        path.extend((rr, edge_col) for rr in range(row - 1, -1, -1))
    else:
        path.extend((rr, edge_col) for rr in range(row + 1, h))
    return path


def _iter_decreasing_indices(path_len: int, length: int):
    pos = length
    gap = length - 1
    while pos < path_len:
        yield pos
        if gap <= 0:
            break
        pos += gap
        gap -= 1


def _paint_path(
    out: np.ndarray,
    path: list[tuple[int, int]],
    length: int,
    color: int,
    bg: int,
    *,
    decreasing: bool,
) -> int:
    if length <= 0:
        return 0

    if decreasing:
        indices = _iter_decreasing_indices(len(path), length)
    else:
        indices = range(0, len(path), length)

    painted = 0
    for idx in indices:
        r, c = path[idx]
        if int(out[r, c]) == bg:
            out[r, c] = color
            painted += 1
    return painted


def _solve_with_rules(
    grid: np.ndarray,
    *,
    use_l_path: bool,
    decreasing: bool,
) -> np.ndarray:
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
        path = _path_cells(h, w, r, sep_col, bars_on_left, use_l_path)
        _paint_path(out, path, length, col, bg, decreasing=decreasing)
    return out


def _solve_straight_periodic(grid: np.ndarray) -> np.ndarray:
    return _solve_with_rules(grid, use_l_path=False, decreasing=False)


def _solve_l_periodic(grid: np.ndarray) -> np.ndarray:
    return _solve_with_rules(grid, use_l_path=True, decreasing=False)


def _solve_straight_decreasing(grid: np.ndarray) -> np.ndarray:
    return _solve_with_rules(grid, use_l_path=False, decreasing=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the extension side using an L-shaped path and decreasing intervals.

    Input structure:
    A mostly solid vertical separator column divides the grid. One side contains 1-2 horizontal
    monochrome bars on a few spaced rows; the opposite side is empty background in the input.

    Rule:
    Process bars from nearest to the separator outward. For each bar, follow the path that starts
    immediately across the separator, runs horizontally to the outer edge, then turns toward the
    nearer top/bottom border. Along that ordered path, paint the bar color at cumulative indices
    L, L+(L-1), L+(L-1)+(L-2), ... where L is the bar length, without overwriting existing color.

    Key insight:
    Two old assumptions both change together: the trail is no longer a straight row-only comb, and
    its marks are no longer equally spaced. The correct output comes from applying a decreasing-gap
    sequence over the entire turned path, not from applying either variant in isolation.
    """
    return _solve_with_rules(grid, use_l_path=True, decreasing=True)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 5, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 4, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 4

P_TWO_BARS = 0.4
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MIN_TOTAL_TRAIL_CELLS = 4
MAX_ATTEMPTS = 800
ROW_PICK_ATTEMPTS = 200


def _has_spacing(rows: list[int]) -> bool:
    return all(b - a >= 2 for a, b in zip(rows, rows[1:]))


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    candidates = [r for r in range(1, h - 1) if not (h % 2 == 1 and r == h // 2)]
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

    fallback = [r for r in range(1, h - 1, 2) if not (h % 2 == 1 and r == h // 2)]
    if len(fallback) >= count:
        rows = fallback[:count]
        if any(r < h // 2 for r in rows) and any(r > (h - 1) // 2 for r in rows):
            return rows
    raise RuntimeError("failed to choose active rows with visible top/bottom turns")


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _count_decreasing_hits(path_len: int, length: int) -> int:
    return sum(1 for _ in _iter_decreasing_indices(path_len, length))


def _count_vertical_hits(path_len: int, horiz_len: int, length: int) -> int:
    return sum(1 for idx in _iter_decreasing_indices(path_len, length) if idx >= horiz_len)


def _feature_lengths(capacity: int, path_len: int, horiz_len: int) -> list[int]:
    return [
        length
        for length in range(2, capacity + 1)
        if _count_decreasing_hits(path_len, length) >= 2
        and _count_vertical_hits(path_len, horiz_len, length) >= 1
    ]


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    path_len: int,
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(80):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        max_len = min(capacity, path_len - 1)
        if max_len <= 0:
            continue

        two_bars = capacity >= 2 and max_len >= 2 and rng.random() < P_TWO_BARS
        if not two_bars:
            length = int(rng.integers(1, max_len + 1))
            return [(length, primary)], sep_gap

        near_len_max = min(capacity - 1, max_len)
        if near_len_max <= 0:
            continue
        near_len = int(rng.integers(1, near_len_max + 1))

        far_capacity = capacity - near_len
        far_len_max = min(far_capacity, path_len - 1)
        if far_len_max <= 0:
            continue
        far_len = int(rng.integers(1, far_len_max + 1))

        others = [c for c in palette if c != primary]
        colors = [primary, int(rng.choice(others))]
        if rng.random() < P_SWAP_COLORS:
            colors.reverse()
        return [(near_len, colors[0]), (far_len, colors[1])], sep_gap

    return None


def _sample_feature_row_bars(
    rng: np.random.Generator,
    primary: int,
    bar_side_len: int,
    path_len: int,
    horiz_len: int,
) -> tuple[list[tuple[int, int]], int] | None:
    lengths = _feature_lengths(bar_side_len, path_len, horiz_len)
    if not lengths:
        return None
    length = int(rng.choice(lengths))
    return [(length, primary)], 0


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


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    combo = main(inp)
    l_periodic = _solve_l_periodic(inp)
    straight_decreasing = _solve_straight_decreasing(inp)
    straight_periodic = _solve_straight_periodic(inp)

    bg = _bg_color(inp)
    h, w = inp.shape

    sep_col, _ = _find_sep_col(inp, bg)
    if sep_col is None:
        return {
            "total_new": 0,
            "vertical_new": 0,
            "diff_vs_l_periodic": 0,
            "diff_vs_straight_decreasing": 0,
            "diff_vs_straight_periodic": 0,
        }

    side_mask = np.ones(w, dtype=bool)
    side_mask[sep_col] = False
    active_mask = np.any(inp[:, side_mask] != bg, axis=1)

    new_mask = (combo != inp) & (inp == bg)
    vertical_new = int(np.sum(new_mask & (~active_mask[:, None])))
    return {
        "total_new": int(np.sum(new_mask)),
        "vertical_new": vertical_new,
        "diff_vs_l_periodic": int(np.sum(combo != l_periodic)),
        "diff_vs_straight_decreasing": int(np.sum(combo != straight_decreasing)),
        "diff_vs_straight_periodic": int(np.sum(combo != straight_periodic)),
    }


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
    horiz_len = _horizontal_len(w, sep_col, bars_on_left)

    max_active = min(ACTIVE_MAX, (h - 1) // 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    try:
        active_rows = _choose_active_rows(rng, h, active_count)
    except RuntimeError:
        return None

    edge_feature_rows = []
    top_row = min(active_rows)
    bottom_row = max(active_rows)
    for row in (top_row, bottom_row):
        path_len = _path_len(h, w, row, sep_col, bars_on_left)
        if _feature_lengths(bar_side_len, path_len, horiz_len):
            edge_feature_rows.append(row)
    if not edge_feature_rows:
        return None

    sep_color, palette = _sample_palette(rng)
    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    primary_by_row = {row: row_primary[idx] for idx, row in enumerate(active_rows)}

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    feature_row = int(rng.choice(edge_feature_rows))
    for row in active_rows:
        path_len = _path_len(h, w, row, sep_col, bars_on_left)
        if row == feature_row:
            sampled = _sample_feature_row_bars(
                rng,
                primary_by_row[row],
                bar_side_len,
                path_len,
                horiz_len,
            )
        else:
            sampled = _sample_row_bars(
                rng,
                palette,
                primary_by_row[row],
                bar_side_len,
                path_len,
            )
        if sampled is None:
            return None
        bars, sep_gap = sampled
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    metrics = _combo_metrics(inp)
    if metrics["total_new"] < MIN_TOTAL_TRAIL_CELLS:
        return None
    if metrics["vertical_new"] < 1:
        return None
    if metrics["diff_vs_l_periodic"] < 1:
        return None
    if metrics["diff_vs_straight_decreasing"] < 1:
        return None
    if metrics["diff_vs_straight_periodic"] < 2:
        return None

    out = main(inp)
    if np.array_equal(out, _solve_l_periodic(inp)):
        return None
    if np.array_equal(out, _solve_straight_decreasing(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
