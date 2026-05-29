# Combo variant of cb2d8a2c (index 36)
# combo_spec: D6/2+D12/3
# Name: 按距离排序处理+绕行N条后终止

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (generator) ---
ROWS_LO, ROWS_HI = 17, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT = 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 900


# Design analysis for the fused variant.
#
# a) Concrete behavior:
#    Original: every red/blue bar contributes a keep-out radius, and the path keeps going until
#    the opposite edge. Variant: first sort bars by shortest Manhattan distance to the green start,
#    then decode N from the start pixel's border position (left/middle/right third -> 1/2/3), and
#    only the first N bars in that sorted list receive keep-out treatment. The path stops on the
#    first row/column just beyond the deepest processed bar, so untouched bars can remain below it.
#    Example (top start): if bar A is earlier in row order but far to the left while the start is
#    near the right wall, and bar B is slightly lower but covers the right side, then B can be
#    closer than A. With N=1 the new solver detours around B and stops; the original/top-down logic
#    would instead treat A first and produce a different route.
#
# b) Complexity calibration:
#    N is limited to three visually distinct tiers. The generator fixes the start edge to the top so
#    the tertile cue remains legible, and always uses four bars so truncation is visible. This keeps
#    the rule noticeable without adding too many free variables.
#
# c) Hidden assumptions changed from the seed:
#    - Seed assumption "all bars matter" no longer holds. Unselected bars remain solid cells but do
#      not get expanded keep-out zones.
#    - Seed assumption "goal is opposite edge" no longer holds. The new goal is the line just past
#      the deepest selected bar.
#    - Seed assumption "bar order along the main axis is enough" no longer holds. We explicitly
#      sort by start-to-bar distance and use a deterministic tie break.
#
# d) Placement legality:
#    The generator preserves 1-cell-thick wall-attached bars, alternating sides, non-overlap, and
#    the original cross-gap constraint. It validates the final grid with the combo solver.
#
# e) Visual observability:
#    The only new latent variable is N, encoded by the start pixel's visible top-edge zone. The
#    generator rejects samples where distance sorting collapses to the same selected prefix as plain
#    row order, so the sorting signal remains observable.
#
# f) Visual neatness:
#    No extra markers are added. Bars keep the original red/blue appearance, and the only output
#    change is a shorter green route.
#
# g) Inferability:
#    Blue still only means per-bar radius. The new clue is positional, not color-based, so the
#    palette does not become harder to parse.
#
# New solver assumptions checked against the generator:
# - Exactly one green border start exists; generator enforces one top-edge pixel away from corners.
# - Mixed red/blue cells form disjoint 4-connected bar components; generator paints exactly that.
# - Bars are 1-cell-thick in the main travel axis, so "just beyond the bar" is a well-defined stop
#   line; generator keeps the original geometry.
# - The selected-bar stop line is reachable while ignored bars remain solid; generator filters by
#   actually running the combo solver and contrasting it with a row-order baseline.


def _get_start_info(sr: int, sc: int, h: int, w: int) -> dict[str, int | str]:
    if sr == 0:
        return {"axis": "row", "sign": 1, "init_dir": 2, "edge_coord": sc, "edge_len": w}
    if sr == h - 1:
        return {"axis": "row", "sign": -1, "init_dir": 0, "edge_coord": sc, "edge_len": w}
    if sc == 0:
        return {"axis": "col", "sign": 1, "init_dir": 1, "edge_coord": sr, "edge_len": h}
    if sc == w - 1:
        return {"axis": "col", "sign": -1, "init_dir": 3, "edge_coord": sr, "edge_len": h}
    raise AssertionError("green start must be on a border edge")


def _main_coord(info: dict[str, int | str], r: int, c: int, h: int, w: int) -> int:
    if info["axis"] == "row":
        return r if info["sign"] == 1 else (h - 1 - r)
    return c if info["sign"] == 1 else (w - 1 - c)


def _cross_coord(info: dict[str, int | str], r: int, c: int) -> int:
    return c if info["axis"] == "row" else r


def _decode_n(edge_coord: int, edge_len: int) -> int:
    if edge_coord * 3 < edge_len:
        return 1
    if edge_coord * 3 < 2 * edge_len:
        return 2
    return 3


def _zone_positions(length: int, zone: int) -> list[int]:
    return [idx for idx in range(1, length - 1) if _decode_n(idx, length) == zone]


def _collect_bars(g: np.ndarray, sr: int, sc: int, info: dict[str, int | str]) -> list[dict]:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict] = []
    for idx, comp in enumerate(comps):
        cells = [(int(r), int(c)) for r, c in comp]
        blue_count = sum(1 for r, c in cells if g[r, c] == BLUE)
        assert blue_count >= 1, "every bar must contain at least one blue radius marker"
        dist_to_start = min(abs(r - sr) + abs(c - sc) for r, c in cells)
        main_vals = [_main_coord(info, r, c, h, w) for r, c in cells]
        cross_vals = [_cross_coord(info, r, c) for r, c in cells]
        bars.append(
            {
                "id": idx,
                "cells": cells,
                "blue_count": blue_count,
                "dist_to_start": int(dist_to_start),
                "main_pos": int(min(main_vals)),
                "cross_pos": int(min(cross_vals)),
            }
        )
    return bars


def _expand_keepout(mask: np.ndarray, cells: list[tuple[int, int]], radius: int) -> None:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=np.int16)
    q = deque()
    for r, c in cells:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        mask[r, c] = True
        if d == radius:
            continue
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))


def _solve_variant(
    grid: np.ndarray,
    *,
    order_mode: str,
    truncate: bool,
    stop_at_opposite_edge: bool,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.array(grid, copy=True)
    h, w = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    info = _get_start_info(sr, sc, h, w)

    bars = _collect_bars(g, sr, sc, info)
    assert bars, "grid must contain at least one red/blue bar"

    if order_mode == "distance":
        ordered = sorted(bars, key=lambda b: (b["dist_to_start"], b["main_pos"], b["cross_pos"], b["id"]))
    elif order_mode == "main":
        ordered = sorted(bars, key=lambda b: (b["main_pos"], b["cross_pos"], b["dist_to_start"], b["id"]))
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")

    decoded_n = _decode_n(int(info["edge_coord"]), int(info["edge_len"]))
    selected = ordered[: min(decoded_n, len(ordered))] if truncate else ordered
    assert selected, "at least one bar must be selected"

    max_main = h - 1 if info["axis"] == "row" else w - 1
    goal_main = max_main if stop_at_opposite_edge else max(int(bar["main_pos"]) for bar in selected) + 1
    assert 1 <= goal_main <= max_main, "goal line must lie strictly inside or on the far edge"

    def is_goal(r: int, c: int) -> bool:
        return _main_coord(info, r, c, h, w) == goal_main and (r, c) != (sr, sc)

    bar_mask = (g == BLUE) | (g == RED)
    blocked = bar_mask.copy()
    keepout = np.zeros((h, w), dtype=bool)
    for bar in selected:
        _expand_keepout(keepout, bar["cells"], int(bar["blue_count"]))
    blocked |= keepout
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    init_dir = int(info["init_dir"])
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid path to the selected stopping line must exist"

    out = g.copy()
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED

    if return_debug:
        debug = {
            "n_value": decoded_n,
            "bar_count": len(bars),
            "selected_ids": tuple(int(bar["id"]) for bar in selected),
            "ordered_ids": tuple(int(bar["id"]) for bar in ordered),
            "goal_main": int(goal_main),
            "path_len": len(path),
        }
        return out, debug
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green route from the border, but only around the nearest N obstacle bars after sorting them by distance from the start.

    Input structure:
    - bg = azure (8). Each obstacle is a 4-connected red/blue bar (2/1), and one green (3) pixel sits on a border edge.
    - The start pixel's position along its border encodes N: first third means 1 bar, middle third means 2, last third means 3.

    Rule:
    - For each bar, blue count gives its Manhattan keep-out radius just as in the seed task.
    - Rank all bars by shortest distance from the start pixel, break ties by their travel-axis position, then only apply keep-out routing to the first N bars in that order.
    - Draw the shortest low-turn green path that clears those selected bars and stops immediately after the deepest selected bar; finally recolor all blue pixels to red.

    Key insight: Two signals interact: the start position says how many bars matter, while blue pixels say how widely each chosen bar must be avoided.
    """
    return _solve_variant(grid, order_mode="distance", truncate=True, stop_at_opposite_edge=False)


def _min_main_len(blues: tuple[int, ...]) -> int:
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...] | None:
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices) if choices else None


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _biased_length(lo: int, hi: int, *, prefer_long: bool) -> int:
    mid = (lo + hi) // 2
    if prefer_long:
        return random.randint(mid, hi)
    return random.randint(lo, mid)


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, grid.shape[1] - length :] = colors[::-1]


def _sample_rows(main_size: int, blues: tuple[int, ...]) -> list[int] | None:
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None
    extras = _random_composition(slack, len(blues) + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(blues)):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)
    return rows


def _reaches_opposite_edge(out: np.ndarray) -> bool:
    return bool(np.any(out[-1, :] == GREEN))


def _gen_attempt(n_value: int) -> np.ndarray | None:
    h = random.randint(ROWS_LO, ROWS_HI)
    w = random.randint(COLS_LO, COLS_HI)
    start_cols = _zone_positions(w, n_value)
    if not start_cols:
        return None

    blues = _choose_blues(BAR_COUNT, h)
    if blues is None:
        return None
    rows = _sample_rows(h, blues)
    if rows is None:
        return None

    start_col = random.choice(start_cols)
    start_pref_side = 0 if start_col <= (w - 1) // 2 else 1
    first_side = 1 - start_pref_side if random.random() < 0.85 else start_pref_side

    grid = np.full((h, w), BG, dtype=int)
    grid[0, start_col] = GREEN

    for i, (row, blue_count) in enumerate(zip(rows, blues)):
        side = (first_side + i) % 2
        bounds = _length_bounds(w, blue_count)
        if bounds is None:
            return None
        prefer_long = side == start_pref_side
        length = _biased_length(*bounds, prefer_long=prefer_long)
        _paint_bar(grid, row, side, length, blue_count)

    try:
        combo_out, combo_dbg = _solve_variant(
            grid, order_mode="distance", truncate=True, stop_at_opposite_edge=False, return_debug=True
        )
        row_out, row_dbg = _solve_variant(
            grid, order_mode="main", truncate=True, stop_at_opposite_edge=False, return_debug=True
        )
        full_out = _solve_variant(grid, order_mode="distance", truncate=False, stop_at_opposite_edge=True)
    except Exception:
        return None

    if combo_dbg["n_value"] != n_value:
        return None
    if combo_dbg["bar_count"] != BAR_COUNT or combo_dbg["n_value"] >= combo_dbg["bar_count"]:
        return None
    if combo_dbg["selected_ids"] == row_dbg["selected_ids"]:
        return None
    if np.array_equal(combo_out, row_out):
        return None
    if np.array_equal(combo_out, full_out):
        return None
    if _reaches_opposite_edge(combo_out):
        return None
    if not np.array_equal(main(grid), combo_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    target_n = random.choice((1, 2, 3))
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(target_n)
        if grid is not None:
            return grid
    for _ in range(MAX_GEN_RETRIES):
        for n_value in random.sample((1, 2, 3), 3):
            grid = _gen_attempt(n_value)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
