# Combo variant of cb2d8a2c (index 38)
# combo_spec: D16/0+D19/2
# Name: 端点颜色编码+穿越型障碍

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, neighbors4

# Design analysis for the combo variant
# ====================================
# a) Concrete behavior and trace
#    - Original task: every red/blue bar is a detour obstacle. The number of blue cells on that bar is the Manhattan
#      keep-out radius, so the green route must bend around every bar. Blue later becomes red.
#    - Combo rule here: bars still carry blue cells that count distance, but the open-end endpoint color now decides
#      whether that bar is a detour bar or a pass-through bar. The wall-attached endpoint is always yellow (4); the
#      free endpoint is magenta (6) for detour and orange (7) for pass-through.
#    - Small canonical trace (start at top, path goes downward): suppose two bars both have blue_count = 2.
#      With free-end 6, the route must avoid the whole radius-2 halo and detour around the bar.
#      With free-end 7, the route may continue straight through one red interior cell of that bar; the crossed cell
#      stays red in the output, while the bar's blue markers still turn red afterward.
#    - The important difference from the D16 solo variant is that pass-through bars are no longer pure red. They still
#      contain blue markers, so "has blue -> must detour" is now false; the endpoint label overrides that assumption.
# b) Complexity calibration
#    - Only two functional endpoint labels are used on the free end: magenta vs orange. This keeps the mapping small
#      enough to infer from a few examples.
#    - Every generated grid contains both obstacle types, and at least one blue-count value appears in both classes.
#      That repeated evidence prevents the puzzle from collapsing back to "count blue cells only".
#    - Bar count is restricted to 3-4 so one of the endpoint labels necessarily repeats inside a single sample.
# c) Implicit-assumption audit against the original main()
#    - Original `bar_mask = (g == BLUE) | (g == RED)` is no longer sufficient: the colored endpoints are part of the
#      same obstacle components and must be included in component detection.
#    - Original `blue_count > 0 => detour obstacle` is invalid now because pass-through bars also contain blue markers.
#      The solver must explicitly decode the free-end color before building blocked regions.
#    - Original `blocked = bar_mask | forbidden` assumed every obstacle cell is unwalkable. That breaks for pass-through
#      bars; only endpoint-coded detour bars and their halos should be blocked.
#    - Original painting logic of only recoloring background to green remains correct and becomes essential, because a
#      traversed pass-through cell must keep its obstacle color instead of turning green.
# d) Placement legality
#    - Bars remain 1-cell-thick side-attached lines, alternating side walls exactly as in the source task.
#    - Detour bars use the original cross-gap legality: the uncovered side must have at least `blue_count + 1` cells.
#      Pass-through bars only need one open-side gap because the path is allowed to cross them directly.
#    - Main-axis spacing is computed from effective radii where pass-through bars contribute radius 0. This keeps detour
#      halos separable without overconstraining pass-through bars.
# e) Visual observability
#    - Humans can compare two bars with the same number of blue markers and notice that one is bypassed while another is
#      crossed. The only stable visual difference between those bars is the free-end endpoint color, which exposes the
#      intended mapping.
#    - Because the crossed bar still shows a red interruption in the route, the pass-through behavior remains visible
#      even after blue markers disappear.
# f) Visual cleanliness
#    - Endpoint metadata is confined to two cells already belonging to the bar, so no extra legend area or side glyphs
#      are needed.
#    - The generator rejects samples where a pass-through crossing lands on a blue or colored endpoint cell; crossings
#      happen on red interior cells only, keeping the output easy to parse.
# g) Change inferability
#    - The palette stays compact: azure background, green path, red/blue bar body, yellow fixed anchor endpoint, and two
#      alternative free-end labels. The color changes are structured: only blue turns red, while endpoint labels persist.
#    - The endpoint-color rule is supported by repeated within-sample evidence and by deliberate same-blue-count bars of
#      opposite types, so the classification can be inferred from structure rather than raw color-id memorization.
# 9) New main() assumptions and generator guarantees
#    - Each obstacle component is a straight 1-cell-thick line with exactly two degree-1 endpoints.
#    - Exactly one endpoint of each bar touches the outer border; that endpoint is yellow and marks the wall-attached end.
#    - The other endpoint is magenta or orange and is the functional label the solver decodes.
#    - Every bar has at least one blue interior marker, even pass-through bars; this is enforced so endpoint decoding is
#      necessary.
#    - The generated route crosses at least one pass-through bar on a red interior cell and detours around at least one
#      magenta-labeled bar, ensuring the fused behavior is actually exercised.

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3
ANCHOR = 4
DETOUR_END = 6
PASS_END = 7

OBSTACLE_COLORS = (BLUE, RED, ANCHOR, DETOUR_END, PASS_END)

# --------------- shared solver helpers ---------------


def _start_and_goal(grid: np.ndarray) -> tuple[int, int, int, str, int]:
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


def _is_goal(goal_kind: str, goal_val: int, r: int, c: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _on_border(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _bar_infos(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    bar_mask = np.isin(grid, OBSTACLE_COLORS)
    infos: list[dict[str, object]] = []

    for comp in components_mask(bar_mask, connectivity=4):
        cells = [tuple(map(int, cell)) for cell in comp]
        cell_set = set(cells)
        endpoints = []
        for r, c in cells:
            degree = sum((rr, cc) in cell_set for rr, cc in neighbors4(r, c))
            if degree == 1:
                endpoints.append((r, c))

        assert len(endpoints) == 2, "every bar must have exactly two endpoints"
        border_endpoints = [cell for cell in endpoints if _on_border((h, w), cell)]
        assert len(border_endpoints) == 1, "every bar must have exactly one wall-attached endpoint"

        attached_endpoint = border_endpoints[0]
        free_endpoint = endpoints[0] if endpoints[1] == attached_endpoint else endpoints[1]
        attached_color = int(grid[attached_endpoint])
        free_color = int(grid[free_endpoint])

        assert attached_color == ANCHOR, "wall-attached endpoint must use the fixed anchor color"
        assert free_color in (DETOUR_END, PASS_END), "free endpoint must encode bar type"

        blue_count = sum(1 for r, c in cells if grid[r, c] == BLUE)
        assert blue_count >= 1, "every bar must contain at least one blue marker"

        infos.append(
            {
                "cells": cells,
                "cell_set": cell_set,
                "attached_endpoint": attached_endpoint,
                "free_endpoint": free_endpoint,
                "blue_count": blue_count,
                "detour": free_color == DETOUR_END,
                "pass_through": free_color == PASS_END,
            }
        )

    return bar_mask, infos


def _build_blocked(
    shape: tuple[int, int],
    infos: list[dict[str, object]],
    start: tuple[int, int],
    treat_pass_as_detour: bool = False,
) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)

    for info in infos:
        if not (bool(info["detour"]) or treat_pass_as_detour):
            continue

        cells = info["cells"]
        radius = int(info["blue_count"])
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in cells:
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            blocked[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked[start] = False
    return blocked


def _shortest_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    goal_kind: str,
    goal_val: int,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
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
        if _is_goal(goal_kind, goal_val, r, c) and (r, c) != (sr, sc):
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

    assert goal_state is not None, "a valid path to the opposite edge must exist"

    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    path.reverse()
    return path


def _solve_details(grid: np.ndarray, treat_pass_as_detour: bool = False) -> dict[str, object]:
    g = np.array(grid, copy=True)
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    bar_mask, infos = _bar_infos(g)
    blocked = _build_blocked(g.shape, infos, (sr, sc), treat_pass_as_detour=treat_pass_as_detour)
    path = _shortest_path(blocked, (sr, sc), init_dir, goal_kind, goal_val)
    path_set = set(path)

    pass_infos = [info for info in infos if bool(info["pass_through"])]
    detour_infos = [info for info in infos if bool(info["detour"])]
    pass_hit_counts = []
    pass_bad_hits = 0
    pass_endpoint_hits = 0
    for info in pass_infos:
        hits = [cell for cell in path if cell in info["cell_set"]]
        pass_hit_counts.append(len(hits))
        for r, c in hits:
            if g[r, c] != RED:
                pass_bad_hits += 1
        if info["attached_endpoint"] in path_set or info["free_endpoint"] in path_set:
            pass_endpoint_hits += 1

    pass_blue_counts = {int(info["blue_count"]) for info in pass_infos}
    detour_blue_counts = {int(info["blue_count"]) for info in detour_infos}

    return {
        "grid": g,
        "bar_mask": bar_mask,
        "infos": infos,
        "path": path,
        "pass_bar_count": len(pass_infos),
        "detour_bar_count": len(detour_infos),
        "pass_bar_hits": sum(1 for n in pass_hit_counts if n > 0),
        "pass_hit_counts": pass_hit_counts,
        "pass_bad_hits": pass_bad_hits,
        "pass_endpoint_hits": pass_endpoint_hits,
        "shared_blue_counts": pass_blue_counts & detour_blue_counts,
    }


def _render_output_from_details(details: dict[str, object]) -> np.ndarray:
    out = np.array(details["grid"], copy=True)
    for r, c in details["path"]:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _baseline_original_output(grid: np.ndarray) -> np.ndarray:
    details = _solve_details(grid, treat_pass_as_detour=True)
    return _render_output_from_details(details)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a border-started green route to the opposite side, but decode each obstacle bar's behavior from its endpoint colors before deciding whether to detour or cross it.

    Input structure:
    - bg = azure (8), with one green (3) start pixel on a border edge.
    - Each obstacle is a 1-cell-thick side-attached bar whose interior still uses red (2) plus blue (1) markers.
    - The wall-touching endpoint is always yellow (4), while the free endpoint is magenta (6) or orange (7) and acts as the bar's type label.

    Rule:
    - Blue count still gives the Manhattan keep-out radius, but only for bars whose free endpoint is magenta.
    - Bars whose free endpoint is orange are pass-through bars: the path may step through them directly, and traversed obstacle cells keep their own colors instead of turning green.
    - After routing the shortest path with fewest turns to the opposite edge, all blue marker cells become red.

    Key insight: Blue still carries the distance number, yet it no longer tells you whether a bar repels the path at all; that decision is encoded by the free-end endpoint color.
    """
    details = _solve_details(grid)
    return _render_output_from_details(details)


# --------------- generator (o5x) ---------------

# Grid size bounds from the source task
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26

# Restrict to 3-4 bars so endpoint-label repetition exists within each sample
BAR_COUNT_LO, BAR_COUNT_HI = 3, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4

BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


def _min_main_len(effective_radii: tuple[int, ...]) -> int:
    last_bar = effective_radii[0] + 2
    for a, b in zip(effective_radii, effective_radii[1:]):
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


def _choose_specs(count: int, main_hi: int) -> list[tuple[bool, int]] | None:
    choices: list[list[tuple[bool, int]]] = []
    for detour_flags in itertools.product((False, True), repeat=count):
        if all(detour_flags) or not any(detour_flags):
            continue
        for blue_counts in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
            pass_counts = {blue_counts[i] for i, flag in enumerate(detour_flags) if not flag}
            detour_counts = {blue_counts[i] for i, flag in enumerate(detour_flags) if flag}
            if not (pass_counts & detour_counts):
                continue
            effective_radii = tuple(blue_counts[i] if detour_flags[i] else 0 for i in range(count))
            if _min_main_len(effective_radii) > main_hi:
                continue
            choices.append([(bool(detour_flags[i]), int(blue_counts[i])) for i in range(count)])
    if not choices:
        return None
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int, detour: bool) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count + 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    if detour:
        hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    else:
        hi = min(BAR_LEN_MAX, cross_size - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int, detour: bool) -> np.ndarray:
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[0] = ANCHOR
    colors[-1] = DETOUR_END if detour else PASS_END
    colors[1 : 2 * blue_count : 2] = BLUE
    placed = colors if side == 0 else colors[::-1]
    if side == 0:
        grid[row, :length] = placed
    else:
        grid[row, w - length :] = placed
    return placed


def _canonical_grid(
    main_size: int,
    cross_size: int,
    specs: list[tuple[bool, int]],
    first_side: int,
) -> np.ndarray | None:
    effective_radii = tuple(blue_count if detour else 0 for detour, blue_count in specs)
    count = len(specs)
    slack = main_size - _min_main_len(effective_radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = effective_radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += effective_radii[i - 1] + effective_radii[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    placed_rows: list[np.ndarray] = []
    lengths: list[int] = []
    for row, (detour, blue_count), side in zip(rows, specs, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count, detour)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        placed_rows.append(_paint_bar(grid, row, side, length, blue_count, detour))

    first_side_now = first_side
    first_len = lengths[0]
    first_row_colors = placed_rows[0]
    red_offsets = [idx for idx, color in enumerate(first_row_colors) if color == RED]
    if not red_offsets:
        return None
    start_offset = random.choice(red_offsets)
    if first_side_now == 0:
        start_col = start_offset
    else:
        start_col = cross_size - first_len + start_offset
    grid[0, start_col] = GREEN
    return grid


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


def _variant_metrics(inp: np.ndarray, out: np.ndarray, details: dict[str, object] | None = None) -> dict[str, object]:
    if details is None:
        details = _solve_details(inp)

    try:
        baseline = _baseline_original_output(inp)
        baseline_diff = int(np.sum(baseline != out))
        baseline_failed = False
    except Exception:
        baseline = None
        baseline_diff = -1
        baseline_failed = True

    return {
        "pass_bar_count": int(details["pass_bar_count"]),
        "detour_bar_count": int(details["detour_bar_count"]),
        "pass_bar_hits": int(details["pass_bar_hits"]),
        "pass_bad_hits": int(details["pass_bad_hits"]),
        "pass_endpoint_hits": int(details["pass_endpoint_hits"]),
        "max_pass_hit_count": max(details["pass_hit_counts"], default=0),
        "shared_blue_count_labels": len(details["shared_blue_counts"]),
        "baseline_failed": baseline_failed,
        "baseline_diff_cells": baseline_diff,
        "baseline": baseline,
    }


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    specs = _choose_specs(count, main_hi)
    if specs is None:
        return None

    effective_radii = tuple(blue_count if detour else 0 for detour, blue_count in specs)
    main_size = random.randint(max(main_lo, _min_main_len(effective_radii)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, specs, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        details = _solve_details(inp)
        out = _render_output_from_details(details)
        metrics = _variant_metrics(inp, out, details=details)
        if metrics["pass_bar_count"] < 1 or metrics["detour_bar_count"] < 1:
            return None
        if metrics["shared_blue_count_labels"] < 1:
            return None
        if metrics["pass_bar_hits"] < 1:
            return None
        if metrics["pass_bad_hits"] != 0 or metrics["pass_endpoint_hits"] != 0:
            return None
        if metrics["max_pass_hit_count"] != 1:
            return None
        if not metrics["baseline_failed"] and metrics["baseline_diff_cells"] <= 0:
            return None
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
