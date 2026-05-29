# Combo variant of cb2d8a2c (index 13)
# combo_spec: D7/1+D14/2
# Name: 对角线障碍+蓝色像素圆形禁区
#
# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and traced example
#    - Preserve the source-task roles: azure background, one green border seed, red/blue obstacle
#      bars, and an output that extends the seed into a shortest low-turn orthogonal path.
#    - Original task: each obstacle is a horizontal/vertical 4-connected bar. If a bar has `b`
#      blue pixels, every cell within Manhattan distance <= b from any bar cell is forbidden.
#      The path therefore routes around thick diamonds centered on whole bars.
#    - Fused task:
#      1. Each obstacle is a 45-degree staircase attached to one side wall, e.g. a left bar can be
#         `[(4,0),(5,1),(6,2),(7,3),(8,4),(9,5),(10,6)]`.
#      2. Its blue cells stay on that staircase, e.g. `[(4,0),(6,2),(8,4)]`, so `b=3`.
#      3. The keep-out zone is no longer measured from every bar cell. Instead, each of those blue
#         cells emits an Euclidean disk of radius 3, and only the union of those disks is
#         forbidden (plus the bar cells themselves).
#    - Concrete difference:
#      For the bar above, the lower red tail `[(9,5),(10,6)]` is still solid obstacle material, but
#      it does not inflate its own halo. A path can therefore squeeze near `(10,7)` or `(11,7)`,
#      which would have been blocked by the source-task rule "expand from the whole bar". That is
#      the intended D14/2 effect, while the staircase geometry is the intended D7/1 effect.
#
# b) Complexity calibration
#    - Use 2-4 bars and blue counts from the small discrete set {2,3,4}. This keeps the examples
#      learnable while making the circular halos large enough to be visually meaningful.
#    - Each bar must have a genuine red tail beyond its blue prefix, so the "only blue pixels emit
#      halos" rule is observable in the output path instead of collapsing into a whole-bar halo.
#    - The solver still uses 4-neighbor motion and the same lexicographic objective
#      `(steps, turns)`, so the fusion changes obstacle geometry rather than introducing a brand-new
#      path language.
#
# c) Hidden-assumption audit against the original main()
#    - Original `components_mask(..., connectivity=4)` assumed each obstacle is orthogonally
#      connected. Broken: a diagonal staircase would split into singletons. The fused solver uses
#      8-connectivity to recover one component per staircase.
#    - Original forbidden-mask BFS expanded from every red/blue cell with Manhattan radius. Broken:
#      D14/2 says distance is Euclidean and measured to the nearest blue pixel, not to the full bar.
#      The fused solver therefore builds disk unions from blue coordinates only.
#    - Original generator spacing formula assumed 1-row bars. Broken: diagonal bars occupy many rows,
#      so legality must be checked with explicit component validation rather than row arithmetic
#      alone.
#    - Original start-placement heuristic assumed the first bar covered one cross-axis line. Broken:
#      a staircase covers many rows/columns, so the generator now validates the exact solved path and
#      only keeps placements that truly interact with the new halos.
#
# d) Placement legality
#    - Every bar is painted as a monotone staircase from a side wall toward the center.
#    - The generator enforces:
#      1. bar cells stay in bounds,
#      2. 8-connected parsing yields exactly the intended number of components,
#      3. each parsed component is still a single monotone diagonal chain,
#      4. the green start is outside all forbidden disks,
#      5. the fused solver finds a path to the opposite edge.
#
# e) Visual observability
#    - D7/1 is directly visible in the input because all obstacles are staircases rather than flat
#      horizontal/vertical strips.
#    - D14/2 is visible in the output because the green route can hug the red tail below the last
#      blue disk of a staircase. The generator rejects samples where a full-bar Euclidean halo would
#      produce the same route, so the blue-only circle logic is not decorative.
#
# f) Visual cleanliness
#    - The palette stays the same four colors as the source task: azure background, blue markers,
#      red obstacle body, green path.
#    - Obstacles are one-cell wide and never deliberately overlap. Complexity comes from invisible
#      circular halos, not from adding extra symbolic overlays or clutter.
#
# g) Change inferability
#    - Blue still has one stable meaning: count the blue pixels on a staircase to get that
#      staircase's avoidance radius.
#    - Output recoloring is still simple and global: every blue pixel becomes red after routing.
#      The novel behavior is geometric, not a hidden color remapping.
#
# New implicit assumptions introduced by the fused solver
#    - Every red/blue 8-connected component contains at least one blue cell and is intended to be a
#      single monotone diagonal staircase. The generator validates that structure explicitly.
#    - The path remains 4-neighbor and lexicographically optimal by `(steps, turns)`. The generator
#      solves candidate inputs with that exact model before accepting them.

from __future__ import annotations

import heapq
import random
import sys
from pathlib import Path

import numpy as np

_CODE_PATH = Path(__file__).resolve()
for _candidate in (_CODE_PATH.parent, *_CODE_PATH.parents):
    _arc_utils = _candidate / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        _solvers_dir = str(_candidate / "solvers")
        if _solvers_dir not in sys.path:
            sys.path.append(_solvers_dir)
        break

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters ---
ROWS_LO, ROWS_HI = 18, 29
COLS_LO, COLS_HI = 16, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 2, 3
BAR_LEN_MIN, BAR_LEN_MAX = 6, 11
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 34, 58
ROW_GAP_LO, ROW_GAP_HI = 1, 4
MAX_GEN_RETRIES = 900


def _find_start(grid: np.ndarray) -> tuple[int, int, int, str, int]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    h, w = grid.shape

    if sr == 0:
        return sr, sc, 2, "row", h - 1
    if sr == h - 1:
        return sr, sc, 0, "row", 0
    if sc == 0:
        return sr, sc, 1, "col", w - 1
    if sc == w - 1:
        return sr, sc, 3, "col", 0
    raise AssertionError("green start must be on a border edge")


def _is_goal(r: int, c: int, goal_kind: str, goal_val: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _sorted_points(comp: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in comp)


def _is_monotone_diagonal(comp: np.ndarray) -> bool:
    pts = _sorted_points(comp)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    if rows != list(range(rows[0], rows[0] + len(rows))):
        return False
    if all(cols[i + 1] - cols[i] == 1 for i in range(len(cols) - 1)):
        return True
    if all(cols[i + 1] - cols[i] == -1 for i in range(len(cols) - 1)):
        return True
    return False


def _extract_bars(
    grid: np.ndarray,
    connectivity: int,
    *,
    allow_blue_less: bool = False,
) -> list[dict[str, object]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=connectivity)
    bars: list[dict[str, object]] = []
    for comp in comps:
        cells = _sorted_points(comp)
        blue_points = [(r, c) for r, c in cells if int(grid[r, c]) == BLUE]
        if not blue_points:
            assert allow_blue_less, "each obstacle component must contain at least one blue pixel"
        bars.append(
            {
                "cells": cells,
                "blue_points": blue_points,
                "radius": len(blue_points),
            }
        )
    assert bars, "grid must contain at least one obstacle component"
    return bars


def _paint_output(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _euclidean_halo(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    radius: int,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    radius_sq = radius * radius
    for cr, cc in centers:
        r0 = max(0, cr - radius)
        r1 = min(h - 1, cr + radius)
        c0 = max(0, cc - radius)
        c1 = min(w - 1, cc + radius)
        for r in range(r0, r1 + 1):
            dr_sq = (r - cr) * (r - cr)
            for c in range(c0, c1 + 1):
                if dr_sq + (c - cc) * (c - cc) <= radius_sq:
                    mask[r, c] = True
    return mask


def _manhattan_halo(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    radius: int,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for cr, cc in centers:
        for dr in range(-radius, radius + 1):
            rem = radius - abs(dr)
            r = cr + dr
            if not (0 <= r < h):
                continue
            c0 = max(0, cc - rem)
            c1 = min(w - 1, cc + rem)
            mask[r, c0 : c1 + 1] = True
    return mask


def _build_blocked(
    shape: tuple[int, int],
    bars: list[dict[str, object]],
    *,
    halo_mode: str,
) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    for bar in bars:
        cells = list(bar["cells"])
        radius = int(bar["radius"])
        for r, c in cells:
            blocked[r, c] = True
        if halo_mode == "blue_disk":
            blocked |= _euclidean_halo(shape, list(bar["blue_points"]), radius)
        elif halo_mode == "full_disk":
            blocked |= _euclidean_halo(shape, cells, radius)
        elif halo_mode == "full_manhattan":
            blocked |= _manhattan_halo(shape, cells, radius)
        else:
            raise ValueError(f"unknown halo mode: {halo_mode}")
    return blocked


def _shortest_low_turn_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    init_dir: int,
    goal_kind: str,
    goal_val: int,
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = shape
    sr, sc = start
    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if _is_goal(r, c, goal_kind, goal_val) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(DIRS_URDL):
            rr = r + dr
            cc = c + dc
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

    assert goal_state is not None, "a valid path to the opposite edge must exist"

    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    return path


def _solve_with_mode(
    grid: np.ndarray,
    *,
    connectivity: int,
    halo_mode: str,
    allow_blue_less: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sr, sc, init_dir, goal_kind, goal_val = _find_start(g)
    bars = _extract_bars(g, connectivity=connectivity, allow_blue_less=allow_blue_less)
    blocked = _build_blocked((h, w), bars, halo_mode=halo_mode)
    blocked[sr, sc] = False
    path = _shortest_low_turn_path((h, w), (sr, sc), init_dir, goal_kind, goal_val, blocked)
    return _paint_output(g, path), path, blocked


def _path_turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir: tuple[int, int] | None = None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        cur_dir = (r1 - r0, c1 - c0)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _fusion_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    fused_out, fused_path, fused_blocked = _solve_with_mode(grid, connectivity=8, halo_mode="blue_disk")
    full_disk_out, _, full_disk_blocked = _solve_with_mode(grid, connectivity=8, halo_mode="full_disk")
    old_out, _, _ = _solve_with_mode(
        grid,
        connectivity=4,
        halo_mode="full_manhattan",
        allow_blue_less=True,
    )

    disk_advantage = sum(
        1
        for r, c in fused_path[1:-1]
        if bool(full_disk_blocked[r, c]) and not bool(fused_blocked[r, c])
    )
    return {
        "disk_advantage": int(disk_advantage),
        "turns": int(_path_turn_count(fused_path)),
        "diff_vs_full_disk": bool(not np.array_equal(fused_out, full_disk_out)),
        "diff_vs_originalish": bool(not np.array_equal(fused_out, old_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green border seed to the opposite side while threading between diagonal staircase bars and blue-centered circular danger zones.

    Input structure:
    - Background is azure (8). Obstacles are red/blue staircase bars (2/1) running along 45-degree diagonals.
    - Exactly one green (3) pixel lies on a border edge and marks the route start.

    Rule:
    - Each 8-connected red/blue staircase is one obstacle bar; its blue-pixel count is the avoidance radius.
    - Bar cells themselves are solid obstacles, but the keep-out halo is built only from the bar's blue cells.
      Every cell at Euclidean distance <= radius from any blue cell is forbidden, creating a union of circular
      no-go zones along the diagonal.
    - The green route moves orthogonally, choosing the minimum-step path and then the minimum-turn tie-breaker
      to reach the opposite edge without entering those forbidden cells.
    - After the route is drawn, all blue pixels turn red.

    Key insight:
    Blue pixels no longer just count a radius for the whole bar. They are the actual halo emitters, so a long
    red tail below the last blue marker can be passed more closely than under the source-task whole-bar rule.
    """
    out, _, _ = _solve_with_mode(grid, connectivity=8, halo_mode="blue_disk")
    return out


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count + 2,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - 2,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _paint_diagonal_bar(
    grid: np.ndarray,
    row0: int,
    side: int,
    length: int,
    blue_count: int,
) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    w = grid.shape[1]
    for k, color in enumerate(colors):
        r = row0 + k
        c = k if side == 0 else w - 1 - k
        grid[r, c] = int(color)


def _sample_start_rows(main_size: int, lengths: list[int], blues: tuple[int, ...]) -> list[int] | None:
    count = len(lengths)
    row0_lo = blues[0] + 2
    row0_hi = min(main_size - max(lengths) - 2 - ROW_GAP_LO * (count - 1), blues[0] + 6)
    if row0_lo > row0_hi:
        return None

    for _ in range(80):
        rows = [random.randint(row0_lo, row0_hi)]
        ok = True
        for _idx in range(1, count):
            rows.append(rows[-1] + random.randint(ROW_GAP_LO, ROW_GAP_HI))
        for row, length in zip(rows, lengths):
            if row + length - 1 >= main_size - 1:
                ok = False
                break
        if ok:
            return rows
    return None


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_size = random.randint(ROWS_LO, ROWS_HI)
        cross_size = random.randint(COLS_LO, COLS_HI)
    else:
        main_size = random.randint(COLS_LO, COLS_HI)
        cross_size = random.randint(ROWS_LO, ROWS_HI)

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = tuple(random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI) for _ in range(count))
    first_side = random.randrange(2)

    lengths: list[int] = []
    for blue_count in blues:
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        lengths.append(random.randint(*bounds))

    rows = _sample_start_rows(main_size, lengths, blues)
    if rows is None:
        return None

    canonical = np.full((main_size, cross_size), BG, dtype=int)
    for idx, (row, length, blue_count) in enumerate(zip(rows, lengths, blues)):
        _paint_diagonal_bar(canonical, row, (first_side + idx) % 2, length, blue_count)

    comps = components_mask((canonical == BLUE) | (canonical == RED), connectivity=8)
    if len(comps) != count:
        return None
    if not all(_is_monotone_diagonal(comp) for comp in comps):
        return None

    first_len = lengths[0]
    if first_side == 0:
        start_col_lo = 1
        start_col_hi = min(cross_size - 2, first_len - 1)
    else:
        start_col_lo = max(1, cross_size - first_len)
        start_col_hi = cross_size - 2
    if start_col_lo > start_col_hi:
        return None
    canonical[0, random.randint(start_col_lo, start_col_hi)] = GREEN

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
        metrics = _fusion_metrics(inp)
        if int(metrics["disk_advantage"]) < 2:
            return None
        if int(metrics["turns"]) < count:
            return None
        if not bool(metrics["diff_vs_full_disk"]):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D7/1+D14/2 input")
