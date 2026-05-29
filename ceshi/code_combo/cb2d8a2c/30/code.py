# Combo variant of cb2d8a2c (index 30)
# combo_spec: D7/0+D10/2+D13/1
# Name: 单方向障碍+嵌套排列+大量障碍条

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque

import numpy as np

ROOT = "/home/lixinyu/snap/new"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    The seed task already lets a border-started green route dodge red/blue bars by respecting a
#    Manhattan keep-out radius equal to each bar's blue-pixel count. The fusion changes the input
#    geometry rather than the pathfinding rule itself. In the original generator, a top-start case
#    with horizontal bars might place rows independently as
#      left wall bar, then right wall bar, then left wall bar,
#    all touching the outer frame. The route snakes across the full width each time.
#    In this combo, every instance keeps that one horizontal orientation (D7/0), but the bars are
#    no longer independent. Example with width 30, blues (2,1,2,1,1), and first side = left:
#      chamber [0,29] -> bar 1 spans [0,7], leaves chamber [10,29]
#      chamber [10,29] -> bar 2 spans [23,29], leaves chamber [10,21]
#      chamber [10,21] -> bar 3 spans [10,15], leaves chamber [18,21]
#      chamber [18,21] -> bar 4 spans [19,21], leaves chamber [18,17] is illegal, so the
#      generator instead rejects such lengths and only keeps choices whose remaining chamber can
#      still host the suffix. A valid sample therefore becomes a deep alternating left/right
#      penetration, and the green path forms a comb-like ladder of repeated horizontal excursions
#      while moving downward through five layers.
# b) Complexity calibration:
#    The quantity jump is implemented as exactly five bars. That is the smallest count that visibly
#    realizes D13/1 while still leaving enough width, under the ARC 30x30 regime, for nested inner
#    bars to remain legible. Blue counts are limited to {1,2}; this preserves visible radius
#    variation without making the chamber collapse too often. Width is biased to 29-30 so inner bars
#    are not all forced to the minimum length.
# c) Assumption audit of main():
#    - Border-start detection: unchanged; the fused generator still places exactly one green border
#      pixel.
#    - Goal selection: unchanged; the path still reaches the opposite edge.
#    - Component extraction: still valid because each obstacle is still a disconnected 1-cell-thick
#      line segment, even when inner bars float away from the global walls.
#    - Keep-out construction: unchanged, but the fusion makes spacing constraints stricter; the
#      generator must guarantee both main-axis gaps and chamber shrinkage legality.
#    - The seed solver's single global shortest-path search is too weak here: even on a valid nested
#      input it can legally hug one chamber boundary and skip visible detours. The fused solver
#      therefore still uses Dijkstra locally, but now routes through explicit per-layer waypoints so
#      every nested bar contributes a visible crossing.
# d) Placement legality:
#    The generator explicitly tracks the active chamber [left,right] in canonical top-start layout.
#    A left-attached bar consumes length + blue_count cells from the chamber's left side; a right bar
#    is symmetric. Before choosing each length, the generator reserves the minimum chamber required
#    by the remaining suffix, which prevents overlaps and dead ends. Rows are spaced by the original
#    formula row_gap >= blue_i + blue_{i+1} + 2, so successive forbidden diamonds do not seal the
#    corridor vertically.
# e) Visual observability:
#    Humans can see three persistent cues across examples: all obstacle strips are horizontal, there
#    are five of them, and only the first two reach the outer walls while the remaining inner bars
#    float inside earlier gaps. That visible containment relation distinguishes the combo from the
#    seed task's ordinary left-right snake.
# f) Visual cleanliness:
#    The palette stays at the original four colors. Obstacles remain thin, disconnected, and ordered,
#    so even with five bars the input reads as a clean fence structure rather than clutter.
# g) Color-rule inferability:
#    Blue still means "part of a bar" and also still encodes the radius. The output recolor rule is
#    unchanged: every blue cell turns red after the route is drawn. Because the geometry is repeated
#    five times, the relation between blue prefix size and wider detour remains observable.
# h) Quantitative anti-degeneration target:
#    Every generated sample should have exactly five bars, all horizontal, at least three bars that
#    touch neither global side wall, and an output path with at least nine turns. The first three
#    numbers prove the input is genuinely nested; the turn count proves the solver is actually
#    traversing the layers rather than collapsing them into one outer detour.
# i) New solver assumptions introduced here:
#    The fused path-construction branch assumes: only top/bottom starts are emitted, bars are
#    horizontal, bar count is five, and the active chamber alternates side attachments while staying
#    positive. The generator and the helper metrics below enforce those assumptions. For any input
#    outside that signature, main() falls back to the seed task's global pathfinding behavior.


_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _run_dijkstra(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    is_goal,
) -> tuple[list[tuple[int, int]], int]:
    inf_cost = (10**9, 10**9)
    sr, sc = start
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf_cost):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < blocked.shape[0] and 0 <= cc < blocked.shape[1]):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf_cost):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid path to the target must exist"
    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    path.reverse()
    return path, int(goal_state[2])


def _forced_waypoints(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    if start_pts.shape[0] != 1:
        return None
    sr, _sc = map(int, start_pts[0])
    if sr == 0:
        step = 1
        goal_row = h - 1
    elif sr == h - 1:
        step = -1
        goal_row = 0
    else:
        return None

    bar_mask = (g == BLUE) | (g == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict[str, int]] = []
    for comp in comps:
        rows = {int(r) for r, _c in comp}
        cols = sorted(int(c) for _r, c in comp)
        if len(rows) != 1:
            return None
        bars.append(
            {
                "row": next(iter(rows)),
                "start": cols[0],
                "end": cols[-1],
                "blue": sum(1 for r, c in comp if int(g[r, c]) == BLUE),
            }
        )

    if len(bars) != BAR_COUNT:
        return None
    bars.sort(key=lambda bar: int(bar["row"]), reverse=(step < 0))

    left = 0
    right = w - 1
    waypoints: list[tuple[int, int]] = []
    for bar in bars:
        start = int(bar["start"])
        end = int(bar["end"])
        blue = int(bar["blue"])
        attached_left = start == left
        attached_right = end == right
        if attached_left == attached_right:
            return None

        if attached_left:
            left = end + blue + 1
            exit_col = right
        else:
            right = start - blue - 1
            exit_col = left

        waypoint_row = int(bar["row"]) + step
        if not (0 <= waypoint_row < h and left <= right and left <= exit_col <= right):
            return None
        waypoints.append((waypoint_row, exit_col))

    return waypoints, (goal_row, waypoints[-1][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green route from one border to the opposite through a deep, same-direction fence of nested bars whose blue pixels set each layer's exclusion radius.

    Input structure:
    - bg = azure (8). Red/blue obstacle bars (2/1) are 1-cell-thick segments, and generated instances use one shared orientation for all bars while nesting later bars inside the chamber left by earlier ones.
    - A single green (3) pixel sits on the starting border and marks the entry point.

    Rule:
    - Each disconnected red/blue bar forbids every cell at Manhattan distance less than or equal to its number of blue pixels.
    - The green route grows orthogonally toward the opposite border, repeatedly turning to cross the successive chambers created by the five nested bars.
    - After the path is painted, every blue pixel is recolored red.

    Key insight: Blue pixels still encode obstacle radius, but the obstacles now form a deep single-orientation nested fence, so the shortest legal route becomes a layered comb of repeated detours through shrinking gaps.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    GREEN = 3
    BLUE = 1

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1
    elif sr == h - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", w - 1
    elif sc == w - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    bar_mask = (g == BLUE) | (g == 2)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque(comp)
        for r, c in comp:
            dist[r, c] = 0

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == blue_count:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked = bar_mask | forbidden
    blocked[sr, sc] = False

    path: list[tuple[int, int]]
    forced = _forced_waypoints(g)
    if forced is None:
        path, _last_dir = _run_dijkstra(blocked, (sr, sc), init_dir, is_goal)
    else:
        waypoints, final_goal = forced
        path = [(sr, sc)]
        cur = (sr, sc)
        cur_dir = init_dir
        for tr, tc in [*waypoints, final_goal]:
            segment, cur_dir = _run_dijkstra(
                blocked,
                cur,
                cur_dir,
                lambda r, c, tr=tr, tc=tc: r == tr and c == tc,
            )
            path.extend(segment[1:])
            cur = (tr, tc)

        assert is_goal(*path[-1]), "forced waypoint plan must reach the opposite edge"

    out = g.copy()
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = 2
    return out


BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 25, 30
COLS_LO, COLS_HI = 28, 30
BAR_COUNT = 5
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 2
BAR_LEN_MIN, BAR_LEN_MAX = 4, 14
MAX_GEN_RETRIES = 800


def _min_bar_len(blue_count: int) -> int:
    return max(BAR_LEN_MIN, 2 * blue_count - 1)


def _min_chamber_needed(blues: tuple[int, ...]) -> int:
    need = 1
    for blue_count in reversed(blues):
        need += int(blue_count) + _min_bar_len(int(blue_count))
    return need


def _min_main_len(blues: tuple[int, ...]) -> int:
    last_bar = int(blues[0]) + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += int(a) + int(b) + 2
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


def _choose_blues(cross_size: int, main_hi: int) -> tuple[int, ...] | None:
    combos: list[tuple[int, ...]] = []
    weights: list[int] = []
    for combo in itertools.product(
        range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1),
        repeat=BAR_COUNT,
    ):
        if _min_main_len(combo) > main_hi or _min_chamber_needed(combo) > cross_size:
            continue
        twos = sum(1 for x in combo if int(x) == 2)
        combos.append(combo)
        weights.append({0: 1, 1: 5, 2: 6, 3: 4, 4: 2}.get(twos, 1))
    if not combos:
        return None
    return random.choices(combos, weights=weights, k=1)[0]


def _paint_bar(
    grid: np.ndarray,
    row: int,
    start: int,
    end: int,
    side: int,
    blue_count: int,
) -> None:
    length = end - start + 1
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, start : end + 1] = colors
    else:
        grid[row, start : end + 1] = colors[::-1]


def _choose_nested_spans(
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> list[tuple[int, int, int]] | None:
    left = 0
    right = cross_size - 1
    spans: list[tuple[int, int, int]] = []

    for i, blue_count in enumerate(blues):
        side = (first_side + i) % 2
        chamber = right - left + 1
        min_next = _min_chamber_needed(blues[i + 1 :])
        lo = _min_bar_len(int(blue_count))
        hi = min(BAR_LEN_MAX, chamber - int(blue_count) - min_next)
        if lo > hi:
            return None

        prefer_lo = max(lo, hi - (3 if i < len(blues) - 1 else 2))
        length = random.randint(prefer_lo, hi)
        if side == 0:
            start = left
            end = left + length - 1
            left = end + int(blue_count) + 1
        else:
            end = right
            start = right - length + 1
            right = start - int(blue_count) - 1
        spans.append((start, end, side))

    return spans


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    spans = _choose_nested_spans(cross_size, blues, first_side)
    if spans is None:
        return None

    extras = _random_composition(slack, len(blues) + 1)
    row = int(blues[0]) + 2 + extras[0]
    rows = [row]
    for i in range(1, len(blues)):
        row += int(blues[i - 1]) + int(blues[i]) + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, (start, end, side) in zip(rows, blues, spans):
        _paint_bar(grid, row, start, end, side, int(blue_count))

    start0, end0, _side0 = spans[0]
    if end0 - start0 + 1 < 3:
        return None
    grid[0, random.randint(start0 + 1, end0 - 1)] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    return np.flipud(grid).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    return bool(np.any(out[0, :] == GREEN))


def _layout_metrics(grid: np.ndarray) -> dict[str, int | bool | str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    if start_pts.shape[0] != 1:
        return {
            "bar_count": 0,
            "orientation": "invalid",
            "nested_valid": False,
            "floating_bars": 0,
            "global_touch_bars": 0,
        }

    sr, _sc = map(int, start_pts[0])
    if sr == 0:
        edge = "top"
    elif sr == h - 1:
        edge = "bottom"
    else:
        edge = "side"

    bar_mask = (g == BLUE) | (g == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict[str, int]] = []
    orientations: set[str] = set()
    for comp in comps:
        rows = sorted({int(r) for r, _c in comp})
        cols = sorted({int(c) for _r, c in comp})
        blue_count = sum(1 for r, c in comp if int(g[r, c]) == BLUE)
        if len(rows) == 1:
            orientations.add("horizontal")
            bars.append(
                {
                    "main": rows[0],
                    "start": cols[0],
                    "end": cols[-1],
                    "blue": blue_count,
                }
            )
        elif len(cols) == 1:
            orientations.add("vertical")
        else:
            orientations.add("nonlinear")

    orientation = next(iter(orientations)) if len(orientations) == 1 else "mixed"
    if orientation != "horizontal" or edge == "side":
        return {
            "bar_count": len(bars),
            "orientation": orientation,
            "nested_valid": False,
            "floating_bars": 0,
            "global_touch_bars": 0,
        }

    bars.sort(key=lambda bar: int(bar["main"]), reverse=(edge == "bottom"))

    left = 0
    right = w - 1
    floating_bars = 0
    global_touch_bars = 0
    nested_valid = True
    for i, bar in enumerate(bars):
        start = int(bar["start"])
        end = int(bar["end"])
        blue = int(bar["blue"])

        if start > 0 and end < w - 1:
            floating_bars += 1
        if start == 0 or end == w - 1:
            global_touch_bars += 1

        attached_left = start == left
        attached_right = end == right
        if attached_left == attached_right:
            nested_valid = False
            break

        if attached_left:
            left = end + blue + 1
        else:
            right = start - blue - 1

        if i < len(bars) - 1 and left > right:
            nested_valid = False
            break

    return {
        "bar_count": len(bars),
        "orientation": orientation,
        "nested_valid": nested_valid,
        "floating_bars": floating_bars,
        "global_touch_bars": global_touch_bars,
    }


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom"))
    cross_size = random.choices([28, 29, 30], weights=[1, 3, 4], k=1)[0]
    blues = _choose_blues(cross_size, ROWS_HI)
    if blues is None:
        return None

    main_size = random.randint(max(ROWS_LO, _min_main_len(blues)), ROWS_HI)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    metrics = _layout_metrics(inp)
    if metrics["bar_count"] != BAR_COUNT:
        return None
    if metrics["orientation"] != "horizontal" or not metrics["nested_valid"]:
        return None
    if metrics["floating_bars"] < BAR_COUNT - 2 or metrics["global_touch_bars"] != 2:
        return None

    try:
        out = main(inp)
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
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
