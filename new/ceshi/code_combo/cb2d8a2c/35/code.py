# Combo variant of cb2d8a2c (index 35)
# combo_spec: D9/3+D15/2
# Name: 蓝色像素移动到转弯点+障碍条被绕后变色

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
#    - Original task: each blue count sets a bar's Manhattan keep-out radius, the path
#      detours around every bar, and all blue pixels simply recolor to red in place.
#    - Combo rule here: the route geometry stays the same, but each bypassed bar turns
#      grey and its blue count is transferred onto that bar's routed detour segment.
#      On the generated instances bars carry 1 or 2 blue pixels, so the moved markers
#      land on the first turn only or on both turns of that bar's bypass.
#    - Small trace: if a left-attached bar with blue_count=2 is skirted on its top side,
#      the original output would show a green horizontal detour and a fully red bar.
#      The combo output keeps the same detour geometry, recolors the whole bar grey,
#      and paints both detour corner cells blue instead of leaving them green.
# b) Complexity calibration
#    - Restricting blue counts to {1,2} keeps the "move to turning points" mapping
#      unambiguous because each canonical bypass has exactly two visible corners.
#    - Multi-bar samples are generated with both counts whenever possible, so humans can
#      see repeated evidence for "one blue -> first corner, two blues -> both corners".
# c) Implicit assumptions review
#    - The original `out[out == BLUE] = RED` assumed blue pixels stay on the bars. That
#      is false now: bars lose blue entirely and blue must be repainted on routed cells.
#    - The original solver never needed to identify which turns belong to which bar. The
#      combo solver now extracts path turns and matches them to each bar's radius+1 shell.
#    - The old edge check looked only for green. A moved blue marker could, in principle,
#      occupy a path cell, so validation treats green/blue as path colors on the far edge.
# d) Placement legality
#    - Bars still alternate attached walls and keep the original spacing formula, so the
#      shortest route continues to bypass each bar separately.
#    - The generator additionally rejects any sample where a bar does not own exactly two
#      turn cells on the solved path or where moved blue markers would collide.
# e) Visual observability and cleanliness
#    - Grey bars make the D15 change global and obvious, while blue corner markers make
#      the D9 change local and easy to align with the path.
#    - No extra colors beyond grey are introduced on the bars, and blue appears only on
#      the routed detour cells, so bars, path, and markers remain visually separable.
# f) Change inferability
#    - The color logic is structurally anchored: blue leaves the obstacle and reappears on
#      the same obstacle's detour corners; grey means "this whole obstacle was processed".
#    - There is no hidden ordering by numeric color id. Everything is driven by visible
#      counts, bar placement, and path turns.

BG = 8
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 2
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _start_and_goal(g: np.ndarray) -> tuple[int, int, int, str, int]:
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", g.shape[0] - 1
    elif sr == g.shape[0] - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", g.shape[1] - 1
    elif sc == g.shape[1] - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    return sr, sc, init_dir, goal_kind, goal_val


def _build_bar_infos(g: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    blocked = np.zeros((h, w), dtype=bool)
    bars: list[dict[str, object]] = []
    for comp_arr in bar_comps:
        cells = [(int(r), int(c)) for r, c in comp_arr]
        blue_count = sum(1 for r, c in cells if g[r, c] == BLUE)
        radius = int(blue_count)

        shell = np.zeros((h, w), dtype=bool)
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in cells:
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            if d <= radius:
                blocked[r, c] = True
            elif d == radius + 1:
                shell[r, c] = True

            if d == radius + 1:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

        bars.append(
            {
                "cells": cells,
                "blue_count": blue_count,
                "shell": shell,
            }
        )

    return bars, blocked, bar_mask


def _path_turn_indices(path: list[tuple[int, int]]) -> list[int]:
    turns: list[int] = []
    for idx in range(1, len(path) - 1):
        r0, c0 = path[idx - 1]
        r1, c1 = path[idx]
        r2, c2 = path[idx + 1]
        if (r1 - r0, c1 - c0) != (r2 - r1, c2 - c1):
            turns.append(idx)
    return turns


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    bars, blocked, bar_mask = _build_bar_infos(g)
    blocked[sr, sc] = False

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
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

    turn_indices = _path_turn_indices(path)
    turn_cells = {path[idx] for idx in turn_indices}

    if goal_kind == "row":
        main_key = lambda bar: min(r for r, _ in bar["cells"])
        reverse = sr == h - 1
    else:
        main_key = lambda bar: min(c for _, c in bar["cells"])
        reverse = sc == w - 1
    bars.sort(key=main_key, reverse=reverse)

    for order_idx, bar in enumerate(bars):
        shell = bar["shell"]
        shell_indices = [idx for idx, (r, c) in enumerate(path) if shell[r, c]]
        own_turn_indices = turn_indices[2 * order_idx : 2 * order_idx + 2]
        candidates: list[int] = []
        if len(own_turn_indices) == 2:
            detour_segment = list(range(own_turn_indices[0], own_turn_indices[1] + 1))
            for idx in own_turn_indices + detour_segment:
                if idx not in candidates:
                    candidates.append(idx)
        else:
            for idx in own_turn_indices + shell_indices:
                if idx not in candidates:
                    candidates.append(idx)
        bar["turn_indices"] = own_turn_indices
        bar["marker_indices"] = candidates[: int(bar["blue_count"])]

    return {
        "path": path,
        "bars": bars,
        "bar_mask": bar_mask,
        "turn_cells": turn_cells,
        "start": (sr, sc),
        "goal_kind": goal_kind,
        "goal_val": goal_val,
    }


def _expected_marker_cells(details: dict[str, object]) -> list[tuple[int, int]]:
    path = details["path"]
    cells: list[tuple[int, int]] = []
    for bar in details["bars"]:
        for idx in bar["marker_indices"]:
            cells.append(path[idx])
    return cells


def _variant_metrics(inp: np.ndarray, out: np.ndarray, details: dict[str, object] | None = None) -> dict[str, object]:
    if details is None:
        details = _solve_details(inp)

    bar_mask = details["bar_mask"]
    expected_markers = _expected_marker_cells(details)
    actual_markers = [tuple(map(int, rc)) for rc in np.argwhere(out == BLUE)]

    return {
        "input_blue": int(np.count_nonzero(inp == BLUE)),
        "input_bar_pixels": int(np.count_nonzero(bar_mask)),
        "grey_bar_pixels": int(np.count_nonzero(out[bar_mask] == GRAY)),
        "expected_markers": expected_markers,
        "actual_markers": actual_markers,
        "turn_marker_count": sum(1 for cell in actual_markers if cell in details["turn_cells"]),
        "per_bar_turns": [len(bar["turn_indices"]) for bar in details["bars"]],
        "per_bar_markers": [len(bar["marker_indices"]) for bar in details["bars"]],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the path across the grid, move each bar's blue signal onto the route's bend cells, and mark processed bars in grey.

    Input structure:
    - The background is azure (8). One green (3) border pixel is the unique start.
    - Obstacles are 1-cell-thick red/blue bars (2/1) attached to alternating side walls; a bar's blue count still encodes its keep-out radius.

    Rule:
    - Compute the same shortest orthogonal route as in the base task: every bar forbids all cells at Manhattan distance less than or equal to its blue count.
    - After the path is found, every bypassed bar turns grey (5), recording that the path has already routed around it.
    - The blue pixels no longer stay on the bar. Instead they transfer onto that bar's detour segment: on these generated instances one blue marks the first bend and two blues mark both bends of the bypass.

    Key insight:
    - The output couples obstacle processing with path structure: the obstacle itself fades to grey, while its former blue code reappears exactly where the path had to make its decisive corner detour.
    """
    g = np.array(grid, copy=True)
    details = _solve_details(g)
    out = g.copy()

    for bar in details["bars"]:
        for r, c in bar["cells"]:
            out[r, c] = GRAY

    for r, c in details["path"]:
        if out[r, c] == BG:
            out[r, c] = GREEN

    for r, c in _expected_marker_cells(details):
        out[r, c] = BLUE

    return out


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


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    choices = []
    for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
        if _min_main_len(combo) > main_hi:
            continue
        if count > 1 and len(set(combo)) == 1:
            continue
        choices.append(combo)
    if not choices:
        choices = [
            combo
            for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
            if _min_main_len(combo) <= main_hi
        ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(main_size: int, cross_size: int, blues: tuple[int, ...], first_side: int) -> np.ndarray | None:
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths = []
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
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
        band = out[-1, :]
    elif edge == "bottom":
        band = out[0, :]
    elif edge == "left":
        band = out[:, -1]
    else:
        band = out[:, 0]
    return bool(np.any((band == GREEN) | (band == BLUE)))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        details = _solve_details(inp)
        out = main(inp)
        metrics = _variant_metrics(inp, out, details)
        expected_markers = metrics["expected_markers"]
        actual_markers = metrics["actual_markers"]
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
        if metrics["grey_bar_pixels"] != metrics["input_bar_pixels"]:
            return None
        if len(expected_markers) != metrics["input_blue"]:
            return None
        if len(set(expected_markers)) != len(expected_markers):
            return None
        if sorted(expected_markers) != sorted(actual_markers):
            return None
        if metrics["turn_marker_count"] != metrics["input_blue"]:
            return None
        if any(turns != 2 for turns in metrics["per_bar_turns"]):
            return None
        if any(markers != int(bar["blue_count"]) for markers, bar in zip(metrics["per_bar_markers"], details["bars"])):
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
